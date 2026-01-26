"""
Solution 2 HYBRID (VECTORIZED): THE COMPLETE PIPELINE
=====================================================
1. CONFIG: User-friendly Index Selector at the top.
2. PREPROCESSING: Auto-Aligns traces & Calibrates timing windows.
3. ATTACK: High-Speed Parallel Multi-Model Vectorized Attack.
4. MATH: Strictly complies with README (R=2^12).
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import Utils (Ensure kyber_helpers.py has R_INV=2704)
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.kyber_helpers import (
    Q, R_INV, 
    hamming_weight
)

# ==========================================
# [1] USER CONFIGURATION (EDIT HERE)
# ==========================================
class Config:
    # --- FILES ---
    DATA_DIR = 'data'
    INPUTS = 'polynomials_final.csv'
    RAW_SOURCE_FILE = 'merged_final_dataset_V1.npy'
    ALIGNED_TRACES = 'aligned_dataset_universal.npy'
    
    # --- TARGET INDICES ---
    # Edit this list to control exactly which indices the script attacks.
    # (Note: Indices > 40 will likely be SKIPPED due to short trace length)
    TARGET_INDICES = [
        0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 
        140, 154, 168, 182, 196, 210, 224, 238, 252
    ]
    
    # --- ADVANCED SETTINGS ---
    CALIB_WINDOW_IDX0 = (200, 600)   # Where to look for Index 0
    MIN_TRACE_VARIANCE = 1e-10       # Threshold to detect "Dead Traces"
    ATTACK_WINDOW_WIDTH = 400        # How many samples to attack per index
    N_JOBS = -1                      # CPU Cores (-1 = All Cores)

# ==========================================
# [2] ROBUST PREPROCESSING ENGINE
# ==========================================
class RobustDataEngine:
    def __init__(self, config):
        self.cfg = config
        self.traces = None
        self.inputs = None
        self.valid_mask = None
        self.timing_base = 0
        self.timing_stride = 0

    def _align_traces(self, raw_traces):
        """Aligns traces using Sum of Absolute Difference (SAD)."""
        print(f"    [!] Aligned file not found. Starting Auto-Alignment (SAD)...")
        
        # Find first healthy trace to use as reference
        ref_idx = 0
        while np.var(raw_traces[ref_idx]) < self.cfg.MIN_TRACE_VARIANCE: 
            ref_idx += 1
        
        ref_trace = raw_traces[ref_idx]
        ref_pattern = ref_trace[100:600] # Use the first peak as reference
        aligned = np.zeros_like(raw_traces)
        aligned[ref_idx] = ref_trace
        
        # Process alignment
        for i in tqdm(range(len(raw_traces)), desc="    Aligning"):
            if i == ref_idx: continue
            target = raw_traces[i]
            
            # Skip dead traces during alignment
            if np.var(target) < self.cfg.MIN_TRACE_VARIANCE: 
                aligned[i] = target; continue
            
            # Fast SAD
            search_region = target[0:700] 
            best_sad = float('inf')
            best_offset = 0
            len_ref = len(ref_pattern)
            
            # Slide reference over target
            for offset in range(len(search_region) - len_ref + 1):
                diff = ref_pattern - search_region[offset:offset+len_ref]
                sad = np.sum(np.abs(diff))
                if sad < best_sad: 
                    best_sad = sad; best_offset = offset
            
            shift = (0 + best_offset) - 100
            aligned[i] = np.roll(target, -shift)
            
        print(f"    [+] Saving to {self.cfg.ALIGNED_TRACES}...")
        np.save(os.path.join(self.cfg.DATA_DIR, self.cfg.ALIGNED_TRACES), aligned)
        return aligned

    def load_data(self):
        print("\n[1] ROBUST DATA LOADING")
        try:
            # Load Inputs
            input_path = os.path.join(self.cfg.DATA_DIR, self.cfg.INPUTS)
            self.inputs = pd.read_csv(input_path, header=None).values
            
            # Load or Create Aligned Traces
            a_path = os.path.join(self.cfg.DATA_DIR, self.cfg.ALIGNED_TRACES)
            if os.path.exists(a_path):
                print(f"    [+] Found Aligned Cache: {self.cfg.ALIGNED_TRACES}")
                self.traces = np.load(a_path)
            else:
                r_path = os.path.join(self.cfg.DATA_DIR, self.cfg.RAW_SOURCE_FILE)
                print(f"    [i] Loading Raw: {self.cfg.RAW_SOURCE_FILE}")
                if r_path.endswith('.csv'):
                    raw = pd.read_csv(r_path, header=None).values
                else:
                    raw = np.load(r_path)
                self.traces = self._align_traces(raw)
            
            # Dead Trace Removal
            vars = np.var(self.traces, axis=1)
            self.valid_mask = vars > self.cfg.MIN_TRACE_VARIANCE
            print(f"    [+] Valid Traces: {np.sum(self.valid_mask)} / {len(self.traces)}")
            
            return self.traces[self.valid_mask], self.inputs[self.valid_mask]
            
        except Exception as e:
            print(f"[CRITICAL] Data Load Error: {e}"); sys.exit(1)

    def calibrate(self, valid_traces, valid_inputs):
        print("\n[2] AUTO-CALIBRATION (T-Test)")
        
        def t_test(idx, start, end):
            # Sort traces into High/Low groups based on HW of input (Proxy)
            hw = np.array([bin(int(x)).count('1') for x in valid_inputs[:, idx]])
            t_low, t_high = np.percentile(hw, 20), np.percentile(hw, 80)
            
            roi = valid_traces[:, start:end]
            g1, g2 = roi[hw <= t_low], roi[hw >= t_high]
            
            m1, m2 = g1.mean(0), g2.mean(0)
            v1, v2 = g1.var(0)+1e-20, g2.var(0)+1e-20
            
            # Compute T-Statistic
            t = np.abs((m1-m2)/np.sqrt(v1/len(g1)+v2/len(g2)))
            return np.argmax(t)+start
            
        w0s, w0e = self.cfg.CALIB_WINDOW_IDX0
        l0 = t_test(0, w0s, w0e)
        
        # Look for Index 14 roughly 3500 samples later
        l14 = t_test(14, l0+3000, l0+4000)
        
        self.timing_base = l0
        self.timing_stride = (l14 - l0) / 7.0
        print(f"    [+] Calibration Locked: Base={l0}, Stride={self.timing_stride:.2f}")

# ==========================================
# [3] VECTORIZED ATTACK ENGINE
# ==========================================
def vec_identity(x): return x
def vec_lsb(x): return x & 0x0F
def vec_msb(x): return (x >> 12) & 0x0F

def compute_leakage_matrix(inputs, model_func):
    """Generates leakage matrix for ALL 3329 keys at once."""
    n_traces = len(inputs)
    H = np.zeros((n_traces, Q), dtype=np.float32)
    
    guesses = np.arange(Q).reshape(1, -1)
    inputs_col = inputs.reshape(-1, 1)
    
    # Vectorized Montgomery Multiplication (Uses correct R_INV)
    intermediates = (inputs_col * guesses * R_INV) % Q
    
    if model_func == hamming_weight:
        for k in range(Q):
            H[:, k] = np.array([bin(int(x)).count('1') for x in intermediates[:, k]])
    else:
        H = model_func(intermediates)
    return H

def fast_correlation_matrix(O, H):
    """Computes correlation for thousands of keys instantly."""
    O_centered = O - np.mean(O, axis=0, keepdims=True)
    H_centered = H - np.mean(H, axis=0, keepdims=True)
    
    numerator = np.dot(H_centered.T, O_centered)
    o_ss = np.sum(O_centered ** 2, axis=0)
    h_ss = np.sum(H_centered ** 2, axis=0)
    denominator = np.sqrt(np.outer(h_ss, o_ss))
    
    return numerator / (denominator + 1e-10)

# ==========================================
# [4] PARALLEL WORKER
# ==========================================
def process_single_index(idx, traces_full, inputs_full, engine_info):
    timing_base, timing_stride, win_width = engine_info
    
    # 1. Calculate Window
    loop = idx / 2
    center = int(timing_base + (loop * timing_stride))
    if idx == 56 and center > 9800: center = 9303 # Fold-back fix
    
    start = max(0, center - win_width // 2)
    end = min(traces_full.shape[1], center + win_width // 2)
    
    # 2. Check Bounds
    if start >= traces_full.shape[1] - 10 or end > traces_full.shape[1]:
        return (idx, 0, 0.0, "SKIPPED (Out of Bounds)")
    
    traces_roi = traces_full[:, start:end]
    inputs_roi = inputs_full[:, idx]
    
    # 3. Multi-Model Attack
    models = {'hw': hamming_weight, 'identity': vec_identity, 'lsb': vec_lsb, 'msb': vec_msb}
    votes = np.zeros(Q)
    best_overall_corr = 0
    
    for name, func in models.items():
        H = compute_leakage_matrix(inputs_roi, func)
        corr_matrix = fast_correlation_matrix(traces_roi, H)
        max_corrs = np.max(np.abs(corr_matrix), axis=1)
        votes += max_corrs
        
        current_best = np.max(max_corrs)
        if current_best > best_overall_corr:
            best_overall_corr = current_best
            
    final_key = np.argmax(votes)
    status = "CONFIRMED" if best_overall_corr > 0.028 else "LOW_CONF"
    
    return (idx, final_key, best_overall_corr, status)

# ==========================================
# [5] MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("HYBRID VECTORIZED SOLVER (FULL PIPELINE)")
    print(f"[INFO] Using R_INV={R_INV} (Matches README R=2^12)")
    print("="*60)
    
    cfg = Config()
    
    # 1. Initialize & Run Robust Preprocessing
    engine = RobustDataEngine(cfg)
    valid_traces, valid_inputs = engine.load_data()
    engine.calibrate(valid_traces, valid_inputs)
    
    # 2. Prepare for Parallel Execution
    engine_info = (engine.timing_base, engine.timing_stride, cfg.ATTACK_WINDOW_WIDTH)
    
    print(f"\n[3] ATTACKING {len(cfg.TARGET_INDICES)} INDICES (Parallel)")
    results = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(process_single_index)(idx, valid_traces, valid_inputs, engine_info) 
        for idx in tqdm(cfg.TARGET_INDICES, desc="Processing")
    )
    
    # 3. Report & Save
    print("\n" + "-"*65)
    print(f"{'Idx':<5} | {'Key':<6} | {'Corr':<8} | {'Status'}")
    print("-" * 65)
    
    clean_results = []
    for res in results:
        idx, key, corr, status = res
        print(f"{idx:<5} | {key:<6} | {corr:.4f}   | {status}")
        clean_results.append({'Index': idx, 'Recovered_Key': key, 'Correlation': corr})
        
    pd.DataFrame(clean_results).to_csv('final_recovered_keys_vectorized.csv', index=False)
    print("\n[+] SUCCESS. Results saved to final_recovered_keys_vectorized.csv")