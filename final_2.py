import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

# ==========================================
# [1] CONFIGURATION
# ==========================================
class Config:
    # --- FILE PATHS ---
    DATA_DIR = 'data'
    INPUTS = 'polynomials_final.csv'
    
    # 1. Output file (The script creates this after alignment)
    ALIGNED_TRACES = 'aligned_dataset_universal1.npy'
    
    # 2. Input Source (Can be .npy OR .csv)
    # The script looks for this file to load raw data.
    RAW_SOURCE_FILE = 'merged_final_dataset_V1.npy' 
    
    # --- TARGETS ---
    # Add indices to solve here
    TARGET_INDICES = [0, 14, 28, 56, 70, 84, 98] 
    
    # --- KYBER CONSTANTS ---
    KYBER_Q = 3329
    R_INV = 2704 
    
    # --- AUTO-CALIBRATION ---
    CALIB_WINDOW_IDX0 = (200, 600)  
    MIN_T_SCORE = 3.0              
    
    # --- DATA HYGIENE ---
    MIN_TRACE_VARIANCE = 1e-10 
    
    # --- ATTACK SETTINGS ---
    ATTACK_WINDOW_WIDTH = 400 
    CORRELATION_THRESHOLD = 0.028 

# ==========================================
# [2] THE UNIVERSAL ENGINE
# ==========================================

class KyberUniversalSolver:
    def __init__(self, config):
        self.cfg = config
        self.traces = None
        self.inputs = None
        self.num_samples = 0
        self.timing_base = 0
        self.timing_stride = 0
        self.valid_trace_mask = None 

    def _load_raw_data(self):
        """Smart loader that handles both .npy and .csv files."""
        filepath = os.path.join(self.cfg.DATA_DIR, self.cfg.RAW_SOURCE_FILE)
        
        if not os.path.exists(filepath):
            print(f"[ERROR] Source file not found: {filepath}")
            sys.exit(1)
            
        print(f"    [i] Loading Raw Data from: {self.cfg.RAW_SOURCE_FILE}...")
        
        # Check extension
        if filepath.endswith('.csv'):
            print("    [!] CSV detected. This may take a moment to parse...")
            # Header=None assumes no titles, just numbers. Change if needed.
            df = pd.read_csv(filepath, header=None)
            
            # Sanitize CSV: Remove brackets or strings if present
            # (Checks first cell to see if cleaning is needed)
            first_val = df.iloc[0,0]
            if isinstance(first_val, str) and '[' in first_val:
                print("    [!] Cleaning bracketed strings (e.g. '[123')...")
                # Fast cleaning using pandas apply
                clean_func = lambda x: int(str(x).replace('[','').replace(']','').replace("'",'').strip())
                df = df.applymap(clean_func)
            
            return df.values # Return as numpy array
            
        elif filepath.endswith('.npy'):
            return np.load(filepath)
            
        else:
            print("[ERROR] Unsupported file format. Use .csv or .npy")
            sys.exit(1)

    def _align_traces(self, raw_traces):
        """Aligns traces using SAD to remove jitter."""
        print(f"    [!] Aligned file not found. Starting Auto-Alignment...")
        
        # 1. Select Reference (First Non-Dead Trace)
        ref_idx = 0
        while np.var(raw_traces[ref_idx]) < self.cfg.MIN_TRACE_VARIANCE:
            ref_idx += 1
            if ref_idx >= len(raw_traces):
                print("[CRITICAL] All traces appear to be empty/dead!")
                return raw_traces

        ref_trace = raw_traces[ref_idx]
        win_start, win_end = 100, 600
        ref_pattern = ref_trace[win_start:win_end]
        
        aligned = np.zeros_like(raw_traces)
        aligned[ref_idx] = ref_trace
        
        print(f"    [i] Reference Trace: {ref_idx} | Pattern Window: {win_start}-{win_end}")
        
        # 2. Align remaining traces
        max_shift = 100
        
        for i in tqdm(range(len(raw_traces)), desc="    Aligning"):
            if i == ref_idx: continue
            
            target = raw_traces[i]
            if np.var(target) < self.cfg.MIN_TRACE_VARIANCE:
                aligned[i] = target 
                continue
                
            search_start = max(0, win_start - max_shift)
            search_end = min(len(target), win_end + max_shift)
            search_region = target[search_start:search_end]
            
            best_sad = float('inf')
            best_offset = 0
            
            len_ref = len(ref_pattern)
            for offset in range(len(search_region) - len_ref + 1):
                slice_candidate = search_region[offset : offset + len_ref]
                sad = np.sum(np.abs(ref_pattern - slice_candidate))
                if sad < best_sad:
                    best_sad = sad
                    best_offset = offset
            
            current_pos = search_start + best_offset
            shift = current_pos - win_start
            aligned[i] = np.roll(target, -shift)
            
        print(f"    [+] Alignment Complete. Saving to {self.cfg.ALIGNED_TRACES}...")
        np.save(os.path.join(self.cfg.DATA_DIR, self.cfg.ALIGNED_TRACES), aligned)
        return aligned

    def load_and_curate(self):
        """Main Data Ingestion."""
        print(f"\n[1] DATA INGESTION & HEALTH CHECK")
        try:
            # Load Inputs (CSVs usually need robust cleaning too)
            input_path = os.path.join(self.cfg.DATA_DIR, self.cfg.INPUTS)
            if not os.path.exists(input_path):
                print(f"[ERROR] Input file not found: {input_path}")
                sys.exit(1)
            
            # Robust Input Loading
            df_inputs = pd.read_csv(input_path, header=None)
            # Check for bracket cleanup
            if isinstance(df_inputs.iloc[0,0], str) and '[' in str(df_inputs.iloc[0,0]):
                 clean_func = lambda x: int(str(x).replace('[','').replace(']','').replace("'",'').strip())
                 # Apply only if needed to avoid overhead
                 self.inputs = df_inputs.applymap(clean_func).values
            else:
                 self.inputs = df_inputs.values

            # Load Traces
            aligned_path = os.path.join(self.cfg.DATA_DIR, self.cfg.ALIGNED_TRACES)
            if os.path.exists(aligned_path):
                print(f"    [+] Found Pre-Aligned Data: {self.cfg.ALIGNED_TRACES}")
                self.traces = np.load(aligned_path)
            else:
                # Load Raw (CSV or NPY) and Align
                raw_traces = self._load_raw_data()
                self.traces = self._align_traces(raw_traces)

            self.num_samples = self.traces.shape[1]
            
            # Health Check
            variances = np.var(self.traces, axis=1)
            self.valid_trace_mask = variances > self.cfg.MIN_TRACE_VARIANCE
            
            dead_count = len(self.traces) - np.sum(self.valid_trace_mask)
            if dead_count > 0:
                print(f"    [!] WARNING: {dead_count} Dead Traces detected (Masked).")
            else:
                print(f"    [+] Dataset Clean (0 Dead Traces).")
                
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            sys.exit(1)

    def _montgomery(self, a, b):
        return (a.astype(np.int64) * b * self.cfg.R_INV) % self.cfg.KYBER_Q

    def _hw(self, val):
        return np.array([bin(int(x)).count('1') for x in val])

    def _calculate_t_stat(self, target_idx, w_start, w_end):
        if w_start < 0: w_start = 0
        if w_end > self.num_samples: w_end = self.num_samples
        
        valid_traces = self.traces[self.valid_trace_mask]
        valid_inputs = self.inputs[self.valid_trace_mask, target_idx]
        
        hw = self._hw(valid_inputs)
        t_low = np.percentile(hw, 20)
        t_high = np.percentile(hw, 80)
        
        traces_roi = valid_traces[:, w_start:w_end]
        g1 = traces_roi[hw <= t_low]
        g2 = traces_roi[hw >= t_high]
        
        if len(g1) < 5 or len(g2) < 5: return -1, 0
        
        mean1 = np.mean(g1, axis=0)
        mean2 = np.mean(g2, axis=0)
        var1 = np.var(g1, axis=0) + 1e-20
        var2 = np.var(g2, axis=0) + 1e-20
        
        t_stats = np.abs((mean1 - mean2) / np.sqrt(var1/len(g1) + var2/len(g2)))
        return np.argmax(t_stats) + w_start, np.max(t_stats)

    def calibrate(self):
        print(f"\n[2] AUTO-CALIBRATION")
        w0_start, w0_end = self.cfg.CALIB_WINDOW_IDX0
        loc0, sc0 = self._calculate_t_stat(0, w0_start, w0_end)
        
        if sc0 < self.cfg.MIN_T_SCORE:
            print("[!] Calibration Failed: Index 0 signal weak.")
            sys.exit(1)
            
        print(f"    [+] Locked Index 0 at Sample {loc0} (Score: {sc0:.2f})")
        
        loc14, sc14 = self._calculate_t_stat(14, loc0+3000, loc0+4000)
        if sc14 < self.cfg.MIN_T_SCORE: 
             loc14, sc14 = self._calculate_t_stat(14, loc0+2000, loc0+5000)
             
        if sc14 < self.cfg.MIN_T_SCORE:
            print("[!] Calibration Failed: Index 14 not found.")
            sys.exit(1)

        print(f"    [+] Locked Index 14 at Sample {loc14} (Score: {sc14:.2f})")

        self.timing_base = loc0
        self.timing_stride = (loc14 - loc0) / 7.0
        
        print("-" * 50)
        print(f"    [CALIBRATION COMPLETE] Base: {self.timing_base} | Stride: {self.timing_stride:.2f}")
        print("-" * 50)

    def attack(self):
        print(f"\n[3] UNIVERSAL ATTACK SEQUENCE")
        print(f"{'Idx':<5} | {'Loc':<10} | {'Found Key':<10} | {'Corr':<8} | {'Status'}")
        print("-" * 65)
        
        for idx in self.cfg.TARGET_INDICES:
            loop_count = idx / 2
            center = int(self.timing_base + (loop_count * self.timing_stride))
            if idx == 56 and center > 9800: center = 9303
                
            if center > self.num_samples - 100:
                print(f"{idx:<5} | {center:<10} | {'OUT_BOUNDS':<10} | {'-':<8} | SKIPPED")
                continue
                
            r_loc, r_sc = self._calculate_t_stat(idx, center-200, center+200)
            final_loc = r_loc if r_sc > 3.0 else center
            
            w_start = max(0, final_loc - self.cfg.ATTACK_WINDOW_WIDTH // 2)
            w_end = min(self.num_samples, final_loc + self.cfg.ATTACK_WINDOW_WIDTH // 2)
            
            traces_roi = self.traces[self.valid_trace_mask][:, w_start:w_end]
            inputs_roi = self.inputs[self.valid_trace_mask, idx]
            
            t_mean = np.mean(traces_roi, axis=0)
            t_centered = traces_roi - t_mean
            t_den = np.sqrt(np.sum(t_centered**2, axis=0))
            
            best_corr = 0
            best_key = -1
            
            for guess in range(self.cfg.KYBER_Q):
                iv = self._montgomery(inputs_roi, guess)
                h = self._hw(iv)
                h_centered = h - np.mean(h)
                
                num = np.dot(h_centered, t_centered)
                den = np.sqrt(np.sum(h_centered**2))
                if den == 0: continue
                
                corr = np.max(np.abs(num / (den * t_den + 1e-10)))
                if corr > best_corr:
                    best_corr = corr
                    best_key = guess
            
            status = "CONFIRMED" if best_corr > self.cfg.CORRELATION_THRESHOLD else "LOW_CONF"
            print(f"{idx:<5} | {final_loc:<10} | {best_key:<10} | {best_corr:.4f}   | {status}")

if __name__ == "__main__":
    solver = KyberUniversalSolver(Config)
    solver.load_and_curate()
    solver.calibrate()
    solver.attack()