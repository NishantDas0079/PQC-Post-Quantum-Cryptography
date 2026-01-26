"""
Solution 2: Advanced Correlation Power Analysis (CPA) Attack on Kyber PWM
=========================================================================
This solution implements a sophisticated CPA attack with multiple leakage models
and POI (Point of Interest) detection for recovering Kyber secret key coefficients.

Author: IITK SCA Competition Entry
Technique: Multi-Model CPA with Statistical Optimization
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Kyber Parameters
Q = 3329  # Kyber modulus
R = pow(2, 12, Q)  # Montgomery constant
R_INV = pow(R, -1, Q)  # R^-1 mod q

# Target indices
TARGET_INDICES = [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 224, 238, 252]

# Kyber NTT Twiddle factors (zetas) from reference implementation
ZETAS = [
    2285, 2571, 2970, 1812, 1493, 1422, 287, 202, 3158, 622, 1577, 182, 962,
    2127, 1855, 1468, 573, 2004, 264, 383, 2500, 1458, 1727, 3199, 2648, 1017,
    732, 608, 1787, 411, 3124, 1758, 1223, 652, 2777, 1015, 2036, 1491, 3047,
    1785, 516, 3321, 3009, 2663, 1711, 2167, 126, 1469, 2476, 3239, 3058, 830,
    107, 1908, 3082, 2378, 2931, 961, 1821, 2604, 448, 2264, 677, 2054, 2226,
    430, 555, 843, 2078, 871, 1550, 105, 422, 587, 177, 3094, 3038, 2869, 1574,
    1653, 3083, 778, 1159, 3182, 2552, 1483, 2727, 1119, 1739, 644, 2457, 349,
    418, 329, 3173, 3254, 817, 1097, 603, 610, 1322, 2044, 1864, 384, 2114, 3193,
    1218, 1994, 2455, 220, 2142, 1670, 2144, 1799, 2051, 794, 1819, 2475, 2459,
    478, 3221, 3021, 996, 991, 958, 1869, 1522, 1628
]


# ======================= HELPER FUNCTIONS =======================

def montgomery_reduce(a):
    """Montgomery reduction: compute a * R^-1 mod q"""
    return (a * R_INV) % Q

def montgomery_multiply(a, b):
    """Montgomery multiplication: compute a * b * R^-1 mod q"""
    return montgomery_reduce(a * b)

def barrett_reduce(a):
    """Barrett reduction for modular reduction"""
    return a % Q


# ======================= LEAKAGE MODELS =======================

def hamming_weight(x):
    """Compute Hamming weight (number of 1s in binary representation)"""
    x = int(x) & 0xFFFF  # 16-bit
    return bin(x).count('1')

def hamming_distance(x, y):
    """Compute Hamming distance between two values"""
    return hamming_weight(x ^ y)

def identity_model(x):
    """Identity leakage model - value itself"""
    return int(x) & 0xFFFF

def lsb_model(x, n_bits=4):
    """Least significant bits leakage model"""
    return int(x) & ((1 << n_bits) - 1)

def msb_model(x, n_bits=4):
    """Most significant bits leakage model"""
    return (int(x) >> 12) & ((1 << n_bits) - 1)


class LeakageModel:
    """
    Comprehensive leakage model class supporting multiple intermediate values
    and leakage functions for Kyber PWM.
    """

    def __init__(self, model_type='hw'):
        """
        Initialize leakage model.

        Args:
            model_type: 'hw' (Hamming Weight), 'hd' (Hamming Distance),
                       'identity', 'lsb', 'msb'
        """
        self.model_type = model_type
        self.model_funcs = {
            'hw': hamming_weight,
            'identity': identity_model,
            'lsb': lsb_model,
            'msb': msb_model
        }
        self.model_func = self.model_funcs.get(model_type, hamming_weight)

    def compute_intermediate_m0(self, r_i, s_guess):
        """Compute m0 = Montgomery(S[i], r[i])"""
        return montgomery_multiply(s_guess, r_i)

    def compute_intermediate_m1(self, r_i1, s_guess):
        """Compute m1 = Montgomery(S[i+1], r[i+1])"""
        return montgomery_multiply(s_guess, r_i1)

    def compute_intermediate_m3(self, m1, zeta):
        """Compute m3 = Montgomery(m1, zeta)"""
        return montgomery_multiply(m1, zeta)

    def compute_intermediate_out(self, m0, m3):
        """Compute OUT[i] = (m0 + m3) mod q"""
        return (m0 + m3) % Q

    def compute_leakage(self, r_coeff, s_guess, target_idx, intermediate='m0'):
        """
        Compute leakage for a specific intermediate value.

        Args:
            r_coeff: r polynomial coefficient at target index
            s_guess: Secret key guess
            target_idx: Target coefficient index
            intermediate: Which intermediate to target ('m0', 'm1', 'm3', 'out')
        """
        if intermediate == 'm0':
            value = self.compute_intermediate_m0(r_coeff, s_guess)
        elif intermediate == 'm1':
            value = self.compute_intermediate_m1(r_coeff, s_guess)
        elif intermediate == 'm3':
            zeta_idx = target_idx // 2
            zeta = ZETAS[zeta_idx % len(ZETAS)]
            m1 = self.compute_intermediate_m1(r_coeff, s_guess)
            value = self.compute_intermediate_m3(m1, zeta)
        elif intermediate == 'out':
            m0 = self.compute_intermediate_m0(r_coeff, s_guess)
            zeta_idx = target_idx // 2
            zeta = ZETAS[zeta_idx % len(ZETAS)]
            # For OUT, we need both coefficients
            m1 = self.compute_intermediate_m1(r_coeff, s_guess)
            m3 = self.compute_intermediate_m3(m1, zeta)
            value = self.compute_intermediate_out(m0, m3)
        else:
            value = self.compute_intermediate_m0(r_coeff, s_guess)

        return self.model_func(value)

    def compute_hypothetical_leakage_matrix(self, r_coeffs, target_idx, intermediate='m0'):
        """
        Compute hypothetical leakage matrix for all key guesses.

        Args:
            r_coeffs: Array of r coefficients [n_traces]
            target_idx: Target index
            intermediate: Intermediate value to target

        Returns:
            Leakage matrix [n_traces, Q]
        """
        n_traces = len(r_coeffs)
        H = np.zeros((n_traces, Q), dtype=np.float32)

        for s_guess in range(Q):
            for i in range(n_traces):
                H[i, s_guess] = self.compute_leakage(r_coeffs[i], s_guess, target_idx, intermediate)

        return H


# ======================= POI DETECTION =======================

class POIDetector:
    """
    Point of Interest (POI) detector for identifying informative time samples.
    """

    def __init__(self, method='snr'):
        """
        Initialize POI detector.

        Args:
            method: 'snr' (Signal-to-Noise Ratio), 'sost' (Sum of Squared t-statistics),
                   'sosd' (Sum of Squared Differences), 'variance'
        """
        self.method = method

    def detect_snr(self, traces, labels, top_k=500):
        """
        Detect POIs using Signal-to-Noise Ratio.
        SNR = Var(Signal) / Var(Noise)
        """
        unique_labels = np.unique(labels)
        n_samples = traces.shape[1]

        # Compute mean trace for each label
        means = np.zeros((len(unique_labels), n_samples))
        for i, label in enumerate(unique_labels):
            means[i] = traces[labels == label].mean(axis=0)

        # Signal variance (variance of means)
        signal_var = means.var(axis=0)

        # Noise variance (average variance within each class)
        noise_var = np.zeros(n_samples)
        for label in unique_labels:
            noise_var += traces[labels == label].var(axis=0)
        noise_var /= len(unique_labels)

        # SNR
        snr = signal_var / (noise_var + 1e-10)

        # Get top-k POIs
        poi_indices = np.argsort(snr)[-top_k:]

        return poi_indices, snr

    def detect_sost(self, traces, labels, top_k=500):
        """
        Detect POIs using Sum of Squared T-statistics.
        """
        unique_labels = np.unique(labels)
        n_samples = traces.shape[1]

        sost = np.zeros(n_samples)

        # Compute t-test between each pair of classes
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                t_stat, _ = stats.ttest_ind(
                    traces[labels == label1],
                    traces[labels == label2],
                    axis=0
                )
                sost += t_stat ** 2

        poi_indices = np.argsort(sost)[-top_k:]
        return poi_indices, sost

    def detect_variance(self, traces, top_k=500):
        """
        Detect POIs using trace variance (unsupervised).
        """
        variance = traces.var(axis=0)
        poi_indices = np.argsort(variance)[-top_k:]
        return poi_indices, variance

    def detect_difference(self, traces, r_coeffs, target_idx, top_k=500):
        """
        Detect POIs using difference of means based on r coefficient partitioning.
        """
        # Partition traces based on median r value
        median_r = np.median(r_coeffs)
        group1 = traces[r_coeffs < median_r]
        group2 = traces[r_coeffs >= median_r]

        # Difference of means
        diff = np.abs(group1.mean(axis=0) - group2.mean(axis=0))

        poi_indices = np.argsort(diff)[-top_k:]
        return poi_indices, diff


# ======================= CPA ATTACK =======================

class CPAAttack:
    """
    Correlation Power Analysis attack implementation.
    """

    def __init__(self, leakage_model='hw', n_pois=1000):
        """
        Initialize CPA attack.

        Args:
            leakage_model: Type of leakage model ('hw', 'identity', 'lsb', 'msb')
            n_pois: Number of Points of Interest to use
        """
        self.leakage = LeakageModel(leakage_model)
        self.n_pois = n_pois
        self.poi_detector = POIDetector(method='variance')

    def compute_correlation_single_key(self, traces, hypothetical_leakage):
        """
        Compute Pearson correlation between traces and hypothetical leakage.

        Args:
            traces: Power traces [n_traces, n_samples]
            hypothetical_leakage: Hypothetical leakage values [n_traces]

        Returns:
            Correlation values for each time sample [n_samples]
        """
        n_traces, n_samples = traces.shape

        # Center the data
        traces_centered = traces - traces.mean(axis=0)
        leak_centered = hypothetical_leakage - hypothetical_leakage.mean()

        # Compute correlation
        numerator = np.dot(leak_centered, traces_centered)
        denominator = np.sqrt(np.sum(leak_centered ** 2) * np.sum(traces_centered ** 2, axis=0))

        correlation = numerator / (denominator + 1e-10)

        return correlation

    def attack_single_index(self, traces, r_coeffs, target_idx,
                           intermediate='m0', use_pois=True, parallel=True):
        """
        Perform CPA attack on a single target index.

        Args:
            traces: Power traces [n_traces, n_samples]
            r_coeffs: r polynomial coefficients [n_traces]
            target_idx: Target coefficient index
            intermediate: Which intermediate to target
            use_pois: Whether to use POI detection
            parallel: Whether to use parallel processing

        Returns:
            best_key: Recovered key value
            correlations: Correlation matrix [Q, n_samples]
            confidence: Attack confidence metric
        """
        n_traces, n_samples = traces.shape

        # POI detection (optional)
        if use_pois:
            poi_indices, _ = self.poi_detector.detect_variance(traces, top_k=self.n_pois)
            traces_poi = traces[:, poi_indices]
        else:
            traces_poi = traces
            poi_indices = np.arange(n_samples)

        # Compute correlations for all key guesses
        correlations = np.zeros((Q, len(poi_indices)))

        def compute_corr_for_key(s_guess):
            """Compute correlation for a single key guess"""
            hyp_leak = np.array([
                self.leakage.compute_leakage(r_coeffs[i], s_guess, target_idx, intermediate)
                for i in range(n_traces)
            ])
            return self.compute_correlation_single_key(traces_poi, hyp_leak)

        if parallel:
            # Parallel computation
            results = Parallel(n_jobs=-1, verbose=0)(
                delayed(compute_corr_for_key)(s) for s in range(Q)
            )
            correlations = np.array(results)
        else:
            # Sequential computation
            for s_guess in tqdm(range(Q), desc=f"Testing keys for index {target_idx}"):
                correlations[s_guess] = compute_corr_for_key(s_guess)

        # Find best key
        max_corr_per_key = np.max(np.abs(correlations), axis=1)
        best_key = np.argmax(max_corr_per_key)
        best_corr = max_corr_per_key[best_key]

        # Compute confidence (ratio between best and second-best)
        sorted_corrs = np.sort(max_corr_per_key)[::-1]
        confidence = sorted_corrs[0] / (sorted_corrs[1] + 1e-10)

        return best_key, correlations, best_corr, confidence, poi_indices

    def attack_all_indices(self, traces, polynomials, intermediates=['m0']):
        """
        Attack all target indices.

        Args:
            traces: Power traces
            polynomials: Polynomial coefficients
            intermediates: List of intermediates to try

        Returns:
            Dictionary of recovered keys
        """
        recovered_keys = {}

        for target_idx in TARGET_INDICES:
            print(f"\n{'='*60}")
            print(f"Attacking index {target_idx}")
            print(f"{'='*60}")

            r_coeffs = polynomials[:, target_idx].values

            best_overall_key = None
            best_overall_corr = 0

            for intermediate in intermediates:
                print(f"\nTrying intermediate: {intermediate}")

                best_key, correlations, best_corr, confidence, pois = \
                    self.attack_single_index(traces, r_coeffs, target_idx, intermediate)

                print(f"Best key: {best_key}, Correlation: {best_corr:.6f}, Confidence: {confidence:.2f}")

                if best_corr > best_overall_corr:
                    best_overall_corr = best_corr
                    best_overall_key = best_key

            recovered_keys[target_idx] = {
                'key': best_overall_key,
                'correlation': best_overall_corr
            }

            print(f"\n>>> Final key for index {target_idx}: {best_overall_key}")

        return recovered_keys


# ======================= MULTI-MODEL CPA =======================

class MultiModelCPA:
    """
    Multi-model CPA attack that combines multiple leakage models.
    """

    def __init__(self):
        self.models = ['hw', 'identity', 'lsb', 'msb']
        self.intermediates = ['m0', 'm1']
        self.attacks = {m: CPAAttack(leakage_model=m) for m in self.models}

    def attack_index(self, traces, r_coeffs, target_idx):
        """
        Attack a single index using multiple models and combine results.
        """
        votes = {}
        correlation_scores = {}

        for model_name, attack in self.attacks.items():
            for intermediate in self.intermediates:
                key_name = f"{model_name}_{intermediate}"

                best_key, corrs, best_corr, confidence, _ = \
                    attack.attack_single_index(traces, r_coeffs, target_idx,
                                              intermediate, parallel=True)

                if best_key not in votes:
                    votes[best_key] = 0
                    correlation_scores[best_key] = 0

                votes[best_key] += 1
                correlation_scores[best_key] = max(correlation_scores[best_key], best_corr)

        # Combine votes and correlations
        combined_scores = {k: votes[k] * correlation_scores[k] for k in votes}
        best_key = max(combined_scores, key=combined_scores.get)

        return best_key, combined_scores

    def attack_all(self, traces, polynomials):
        """Attack all target indices."""
        recovered_keys = {}

        for target_idx in TARGET_INDICES:
            print(f"\nMulti-Model CPA Attack on index {target_idx}...")
            r_coeffs = polynomials[:, target_idx].values

            best_key, scores = self.attack_index(traces, r_coeffs, target_idx)
            recovered_keys[target_idx] = best_key

            print(f"Recovered key: {best_key}")

        return recovered_keys


# ======================= MAIN EXECUTION =======================

def main():
    """Main execution function"""
    print("="*70)
    print("Solution 2: Advanced CPA Attack on Kyber PWM")
    print("="*70)

    # Load data
    print("\nLoading dataset...")
    traces = np.load('merged_final_dataset_V1.npy')
    polynomials = pd.read_csv('polynomials_final.csv', header=None)

    print(f"Traces shape: {traces.shape}")
    print(f"Polynomials shape: {polynomials.shape}")

    # Normalize traces
    print("\nNormalizing traces...")
    scaler = StandardScaler()
    traces_normalized = scaler.fit_transform(traces)

    # Run CPA Attack
    print("\n" + "="*70)
    print("Starting Correlation Power Analysis Attack")
    print("="*70)

    # Single model CPA
    cpa_attack = CPAAttack(leakage_model='hw', n_pois=2000)
    recovered_keys = cpa_attack.attack_all_indices(
        traces_normalized, polynomials,
        intermediates=['m0', 'm1']
    )

    # Print results
    print("\n" + "="*70)
    print("RECOVERED SECRET KEY COEFFICIENTS (CPA)")
    print("="*70)
    print("\nIndex\t|\tRecovered Value\t|\tCorrelation")
    print("-"*60)
    for idx in TARGET_INDICES:
        key_info = recovered_keys[idx]
        print(f"{idx}\t|\t{key_info['key']}\t\t|\t{key_info['correlation']:.6f}")

    # Save results
    results_df = pd.DataFrame({
        'Index': TARGET_INDICES,
        'Recovered_Key': [recovered_keys[idx]['key'] for idx in TARGET_INDICES],
        'Correlation': [recovered_keys[idx]['correlation'] for idx in TARGET_INDICES]
    })
    results_df.to_csv('recovered_keys_solution2.csv', index=False)
    print("\nResults saved to 'recovered_keys_solution2.csv'")

    return recovered_keys


if __name__ == "__main__":
    main()
