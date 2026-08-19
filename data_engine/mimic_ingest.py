"""MIMIC-IV Ingestion Pipeline — Relational EHR to (N, K×3) Tensor.

Translates MIMIC-IV hosp/icu module tables into the Institutional Grid's
tensor format, handling:
  - Thousands of raw variables with irregular timestamps
  - Asynchronous lab charting (12h+ delays)
  - Missing data patterns unique to critical care
  - Multi-modal data (vitals, labs, meds, procedures)

Schema (MIMIC-IV v3.0+):
  hosp: patients, admissions, labevents, d_labitems
  icu:  icustays, chartevents, d_items, inputevents, outputevents

Output: Per-stay (W, K×3) tensors + z-score stats + patient metadata.
"""
import os
import json
import glob
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

# ─── MIMIC-IV ItemID → Feature Mapping ───────────────────────────────────────

# Maps MIMIC-IV itemids to our 39 clinical features.
# Sources: MIMIC-IV d_items (ICU) + d_labitems (Hosp) tables.
# Itemids from: https://mimic.mit.edu/docs/IV/modules/hosp/labevents/
#               https://mimic.mit.edu/docs/IV/modules/icu/chartevents/

MIMIC_TO_FEATURE = {
    # ── Heart Rate ──
    220045: "HR",   # ICU: Heart Rate (MetaVision)
    211:    "HR",   # ICU: Heart Rate (CareVue legacy)

    # ── O2 Saturation ──
    220277: "O2Sat",  # ICU: O2 saturation pulseoxymetry
    646:    "O2Sat",  # ICU: O2 Saturation (CareVue)
    834:    "O2Sat",  # ICU: O2 sat (MetaVision)

    # ── Temperature ──
    223761: "Temp",  # ICU: Temperature Fahrenheit
    223762: "Temp",  # ICU: Temperature Celsius
    678:    "Temp",  # ICU: Temperature (CareVue)
    676:    "Temp",  # ICU: Temperature (CareVue)
    677:    "Temp",  # ICU: Temperature (CareVue)

    # ── Systolic BP ──
    220050: "SBP",  # ICU: Arterial BP Systolic
    220179: "SBP",  # ICU: Non Invasive BP Systolic
    51:     "SBP",  # ICU: Arterial BP Systolic (CareVue)
    442:    "SBP",  # ICU: Manual BP Systolic (CareVue)
    455:    "SBP",  # ICU: NBP Systolic (CareVue)

    # ── Diastolic BP ──
    220051: "DBP",  # ICU: Arterial BP Diastolic
    220180: "DBP",  # ICU: Non Invasive BP Diastolic
    8368:   "DBP",  # ICU: Arterial BP Diastolic (CareVue)
    8441:   "DBP",  # ICU: NBP Diastolic (CareVue)
    52:     "DBP",  # ICU: Arterial BP Diastolic (CareVue)
    8555:   "DBP",  # ICU: NBP Diastolic (CareVue)

    # ── MAP (Mean Arterial Pressure) ──
    220052: "MAP",  # ICU: Arterial BP Mean
    220181: "MAP",  # ICU: Non Invasive BP Mean
    224:    "MAP",  # ICU: MAP (CareVue)
    8440:   "MAP",  # ICU: NBP Mean (CareVue)
    456:    "MAP",  # ICU: NBP Mean (CareVue)

    # ── Respiratory Rate ──
    220210: "Resp",  # ICU: Respiratory Rate
    615:    "Resp",  # ICU: Respiratory Rate (CareVue)
    618:    "Resp",  # ICU: Respiratory Rate (CareVue)
    614:    "Resp",  # ICU: Respiratory Rate (CareVue)

    # ── Lactate (Lab) ──
    50813:  "Lactate",   # Hosp: Lactate
    816:    "Lactate",   # ICU: Lactate (point of care)
    1483:   "Lactate",   # ICU: Lactate (lab)
    1664:   "Lactate",   # ICU: Lactate
    3826:   "Lactate",   # ICU: Lactate
    3871:   "Lactate",   # ICU: Lactate
    3908:   "Lactate",   # ICU: Lactate (calc)
    7808:   "Lactate",   # ICU: Lactate (whole blood)
    16732:  "Lactate",   # ICU: Lactate
    84129:  "Lactate",   # ICU: Lactate (whole blood)

    # ── WBC (Lab) ──
    51300:  "WBC",   # Hosp: WBC
    51301:  "WBC",   # Hosp: WBC (automated)
    861:    "WBC",   # ICU: WBC (lab)
    1127:   "WBC",   # ICU: WBC
    34482:  "WBC",   # ICU: WBC

    # ── Glucose (Lab) ──
    50931:  "Glucose",  # Hosp: Glucose
    807:    "Glucose",  # ICU: Glucose (whole blood)
    811:    "Glucose",  # ICU: Glucose (lab)
    1529:   "Glucose",  # ICU: Glucose
    225664: "Glucose",  # ICU: Glucose finger stick
    226537: "Glucose",  # ICU: Glucose (whole blood)
    220621: "Glucose",  # ICU: Glucose (calc)

    # ── Creatinine (Lab) ──
    50912:  "Creatinine",  # Hosp: Creatinine
    791:    "Creatinine",  # ICU: Creatinine (lab)
    1162:   "Creatinine",  # ICU: Creatinine
    5067:   "Creatinine",  # ICU: Creatinine (lab)

    # ── Potassium (Lab) ──
    50971:  "Potassium",  # Hosp: Potassium
    817:    "Potassium",  # ICU: Potassium (lab)
    1524:   "Potassium",  # ICU: Potassium
    4136:   "Potassium",  # ICU: Potassium (whole blood)
    84123:  "Potassium",  # ICU: Potassium

    # ── Magnesium (Lab) ──
    50960:  "Magnesium",  # Hosp: Magnesium
    821:    "Magnesium",  # ICU: Magnesium (lab)

    # ── Chloride (Lab) ──
    50902:  "Chloride",  # Hosp: Chloride
    813:    "Chloride",  # ICU: Chloride (lab)
    84117:  "Chloride",  # ICU: Chloride (whole blood)

    # ── Hemoglobin (Lab) ──
    51221:  "Hgb",   # Hosp: Hemoglobin
    849:    "Hgb",   # ICU: Hemoglobin (lab)
    220228: "Hgb",   # ICU: Hemoglobin (calc)
    11682:  "Hgb",   # ICU: Hemoglobin

    # ── Hematocrit (Lab) ──
    51222:  "Hct",   # Hosp: Hematocrit (automated)
    851:    "Hct",   # ICU: Hematocrit (lab)
    220229: "Hct",   # ICU: Hematocrit (calc)
    11633:  "Hct",   # ICU: Hematocrit

    # ── Platelets (Lab) ──
    51265:  "Platelets",  # Hosp: Platelets
    850:    "Platelets",  # ICU: Platelets (lab)
    1536:   "Platelets",  # ICU: Platelets

    # ── BUN (Lab) ──
    51006:  "BUN",   # Hosp: BUN
    845:    "BUN",   # ICU: BUN (lab)

    # ── pH (Lab) ──
    50820:  "pH",    # Hosp: pH (arterial)
    50821:  "pH",    # Hosp: pH (venous)
    780:    "pH",    # ICU: pH (lab)
    4753:   "pH",    # ICU: pH (whole blood)
    3683:   "pH",    # ICU: pH

    # ── PaCO2 (Lab) ──
    50818:  "PaCO2",  # Hosp: PaCO2 (arterial)
    50819:  "PaCO2",  # Hosp: PaCO2 (venous)
    779:    "PaCO2",  # ICU: PaCO2 (lab)
    4751:   "PaCO2",  # ICU: PaCO2 (whole blood)

    # ── FiO2 ──
    223835: "FiO2",  # ICU: Inspired O2 Fraction
    3420:   "FiO2",  # ICU: FiO2 (set)
    3421:   "FiO2",  # ICU: FiO2 (measured)
    3422:   "FiO2",  # ICU: FiO2 (ventilator)
    189:    "FiO2",  # ICU: FiO2 (CareVue)
    190:    "FiO2",  # ICU: FiO2 (CareVue)
    1279:   "FiO2",  # ICU: FiO2 (CareVue)
    1979:   "FiO2",  # ICU: FiO2 (CareVue)
    7249:   "FiO2",  # ICU: FiO2 (CareVue)
    7511:   "FiO2",  # ICU: FiO2 (CareVue)
    7704:   "FiO2",  # ICU: FiO2 (CareVue)
    16547:  "FiO2",  # ICU: FiO2

    # ── SaO2 (Lab) ──
    50817:  "SaO2",   # Hosp: O2 sat (arterial)
    50862:  "SaO2",   # Hosp: O2 sat (venous)
    834:    "SaO2",   # ICU: O2 sat
    220227: "SaO2",   # ICU: O2 saturation (calc)

    # ── Base Excess (Lab) ──
    50802:  "BaseExcess",  # Hosp: Base Excess (arterial)
    50803:  "BaseExcess",  # Hosp: Base Excess (venous)
    778:    "BaseExcess",  # ICU: Base Excess (lab)

    # ── Bicarbonate (Lab) ──
    50806:  "HCO3",   # Hosp: Bicarbonate (arterial)
    50807:  "HCO3",   # Hosp: Bicarbonate (venous)
    50882:  "HCO3",   # Hosp: Bicarbonate (serum)
    786:    "HCO3",   # ICU: Bicarbonate (lab)
    4749:   "HCO3",   # ICU: Bicarbonate (whole blood)

    # ── AST (Lab) ──
    50861:  "AST",   # Hosp: AST

    # ── Alkaline Phosphatase (Lab) ──
    50863:  "Alkalinephos",  # Hosp: Alkaline Phosphate

    # ── Calcium (Lab) ──
    50808:  "Calcium",  # Hosp: Calcium (total)
    50893:  "Calcium",  # Hosp: Calcium (ionized)
    84112:  "Calcium",  # ICU: Calcium (ionized, whole blood)

    # ── Bilirubin Direct (Lab) ──
    50885:  "Bilirubin_direct",  # Hosp: Bilirubin, Direct

    # ── Bilirubin Total (Lab) ──
    50883:  "Bilirubin_total",  # Hosp: Bilirubin, Total

    # ── Troponin I (Lab) ──
    51003:  "TroponinI",  # Hosp: Troponin I

    # ── PTT (Lab) ──
    51274:  "PTT",   # Hosp: PTT
    51275:  "PTT",   # Hosp: PTT (PTT)

    # ── Fibrinogen (Lab) ──
    51214:  "Fibrinogen",  # Hosp: Fibrinogen

    # ── Phosphate (Lab) ──
    50970:  "Phosphate",  # Hosp: Phosphate
    829:    "Phosphate",  # ICU: Phosphate (lab)

    # ── EtCO2 (ICU) ──
    224689: "EtCO2",  # ICU: End Tidal CO2
    19249:  "EtCO2",  # ICU: ETCO2
}

# Feature name → index (must match the Grid's feature order)
FEATURE_NAMES = [
    'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3',
    'FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride',
    'Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate',
    'Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC',
    'Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime'
]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
K = len(FEATURE_NAMES)  # 39
K3 = K * 3  # 117

# ─── Cohort Selection ────────────────────────────────────────────────────────

def select_icu_stays(
    icustays_path: str,
    admissions_path: str,
    patients_path: str,
    min_hours: int = 12,
    max_hours: int = 168,  # 7 days
) -> list[dict]:
    """Select ICU stays for cohort.

    Filters:
    - Stay duration between min_hours and max_hours
    - Has at least some charted data
    """
    # This is a stub — actual implementation depends on data format
    # MIMIC-IV CSV files are available from PhysioNet
    # Parquet files from BigQuery export are more efficient
    return []


# ─── Temporal Assembly ───────────────────────────────────────────────────────

class MIMICStayAssembler:
    """Assembles hourly (W, K×3) tensors from irregular MIMIC events."""

    def __init__(self, window_size: int = 14):
        self.window_size = window_size
        self.events: list[tuple] = []  # (datetime, feature_idx, value)
        self.stay_id: Optional[int] = None
        self.subject_id: Optional[int] = None
        self.start_time: Optional[datetime] = None

    def add_event(self, charttime, itemid: int, value: float):
        """Add a single charted event.

        charttime can be a datetime or an ISO format string.
        """
        if isinstance(charttime, str):
            # Parse ISO format string
            charttime = datetime.fromisoformat(charttime.replace("Z", "+00:00"))

        if itemid in MIMIC_TO_FEATURE:
            feature_name = MIMIC_TO_FEATURE[itemid]
            if feature_name in FEATURE_INDEX:
                self.events.append((charttime, FEATURE_INDEX[feature_name], value))

    def assemble(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble into (T, K), (T, K), (T, K) arrays.

        Returns:
            values: (T, K) z-scored values
            mask: (T, K) observation mask (1=observed, 0=missing)
            delta: (T, K) hours since last observation
        """
        if not self.events:
            return (
                np.zeros((0, K), dtype=np.float32),
                np.zeros((0, K), dtype=np.float32),
                np.zeros((0, K), dtype=np.float32),
            )

        # Sort by time
        self.events.sort(key=lambda x: x[0])
        self.start_time = self.events[0][0]

        # Hourly bins
        max_hours = int((self.events[-1][0] - self.start_time).total_seconds() / 3600) + 1
        T = max_hours

        values = np.zeros((T, K), dtype=np.float32)
        mask = np.zeros((T, K), dtype=np.float32)
        delta = np.full((T, K), 48.0, dtype=np.float32)  # Max cap

        for charttime, feat_idx, value in self.events:
            hour = int((charttime - self.start_time).total_seconds() / 3600)
            hour = min(hour, T - 1)

            # Multiple observations in same hour — take the last one
            values[hour, feat_idx] = value
            mask[hour, feat_idx] = 1.0

        # Forward fill + compute delta
        for c in range(K):
            last_val = 0.0
            last_seen = None  # Track last observed hour in this forward pass
            for t in range(T):
                if mask[t, c]:
                    last_val = values[t, c]
                    last_seen = t
                else:
                    # Forward fill
                    values[t, c] = last_val
                    if last_seen is not None:
                        delta[t, c] = min(t - last_seen, 48.0)

        return values, mask, delta

    def get_windows(self, values: np.ndarray, mask: np.ndarray,
                    delta: np.ndarray) -> list[np.ndarray]:
        """Extract overlapping (W, K×3) windows."""
        T = len(values)
        W = self.window_size
        windows = []

        for t in range(T):
            # Look back W hours (or less at start)
            start = max(0, t - W + 1)
            n = t - start + 1

            window = np.zeros((W, K3), dtype=np.float32)

            for i in range(n):
                src_t = start + i
                dst_t = W - n + i  # Right-aligned

                for c in range(K):
                    window[dst_t, c] = values[src_t, c]  # value
                    window[dst_t, K + c] = mask[src_t, c]  # mask
                    window[dst_t, 2*K + c] = delta[src_t, c]  # delta

            windows.append(window)

        return windows


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_mimic_csv(
    chartevents_path: str,
    itemid_filter: set = None,
    stay_id: int = None,
    chunksize: int = 500_000,
) -> list[dict]:
    """Load MIMIC-IV chartevents from CSV.

    Handles large files by chunking.
    Returns list of {stay_id, charttime, itemid, valuenum}.
    """
    import pandas as pd

    events = []
    for chunk in pd.read_csv(
        chartevents_path,
        chunksize=chunksize,
        dtype={'stay_id': 'Int64', 'itemid': 'Int64', 'valuenum': 'Float64'},
    ):
        # Filter by stay_id if specified
        if stay_id is not None:
            chunk = chunk[chunk['stay_id'] == stay_id]

        # Filter by relevant itemids
        if itemid_filter is not None:
            chunk = chunk[chunk['itemid'].isin(itemid_filter)]

        # Drop rows without values
        chunk = chunk.dropna(subset=['valuenum'])

        for _, row in chunk.iterrows():
            events.append({
                'stay_id': int(row['stay_id']),
                'charttime': pd.to_datetime(row['charttime']),
                'itemid': int(row['itemid']),
                'valuenum': float(row['valuenum']),
            })

    return events


def load_mimic_labevents(
    labevents_path: str,
    itemid_filter: set = None,
    hadm_id: int = None,
    chunksize: int = 500_000,
) -> list[dict]:
    """Load MIMIC-IV labevents from CSV."""
    import pandas as pd

    events = []
    for chunk in pd.read_csv(
        labevents_path,
        chunksize=chunksize,
        dtype={'hadm_id': 'Int64', 'itemid': 'Int64', 'valuenum': 'Float64'},
    ):
        if hadm_id is not None:
            chunk = chunk[chunk['hadm_id'] == hadm_id]

        if itemid_filter is not None:
            chunk = chunk[chunk['itemid'].isin(itemid_filter)]

        chunk = chunk.dropna(subset=['valuenum'])

        for _, row in chunk.iterrows():
            events.append({
                'hadm_id': int(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
                'charttime': pd.to_datetime(row['charttime']),
                'itemid': int(row['itemid']),
                'valuenum': float(row['valuenum']),
                'subject_id': int(row['subject_id']),
            })

    return events


# ─── Tensor Export ────────────────────────────────────────────────────────────

def export_stay_tensors(
    assembler: MIMICStayAssembler,
    z_mu: np.ndarray,
    z_sd: np.ndarray,
    stay_id: int,
) -> dict:
    """Export all windows for one ICU stay.

    Returns:
        {
            'stay_id': int,
            'n_hours': int,
            'n_windows': int,
            'windows': np.ndarray,  # (n_windows, W, K3)
            'completeness': float,  # fraction of non-missing cells
        }
    """
    values, mask, delta = assembler.assemble()
    windows = assembler.get_windows(values, mask, delta)

    if not windows:
        return {'stay_id': stay_id, 'n_hours': 0, 'n_windows': 0, 'windows': [], 'completeness': 0.0}

    # Z-score
    windows_arr = np.stack(windows, axis=0)  # (n_windows, W, K3)

    # Apply z-score to value columns only
    for c in range(K):
        if z_sd[c] > 1e-6:
            windows_arr[:, :, c] = (windows_arr[:, :, c] - z_mu[c]) / z_sd[c]
        else:
            windows_arr[:, :, c] = windows_arr[:, :, c] - z_mu[c]

    completeness = float(mask.sum()) / (len(values) * K) if len(values) > 0 else 0.0

    return {
        'stay_id': stay_id,
        'n_hours': len(values),
        'n_windows': len(windows),
        'windows': windows_arr,
        'completeness': completeness,
    }


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: str,
    output_dir: str,
    max_stays: int = None,
    window_size: int = 14,
):
    """Run the full MIMIC-IV → tensor pipeline.

    data_dir: directory containing MIMIC-IV CSV/parquet files
    output_dir: directory to save tensors + metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all relevant itemids
    all_itemids = set(MIMIC_TO_FEATURE.keys())

    # Load chartevents for stays
    chartevents_path = os.path.join(data_dir, 'icu', 'chartevents.csv.gz')
    if not os.path.exists(chartevents_path):
        chartevents_path = os.path.join(data_dir, 'icu', 'chartevents.csv')
    if not os.path.exists(chartevents_path):
        raise FileNotFoundError(f"chartevents not found at {chartevents_path}")

    print(f"Loading chartevents from {chartevents_path}...", flush=True)
    events = load_mimic_csv(chartevents_path, itemid_filter=all_itemids)
    print(f"Loaded {len(events)} events", flush=True)

    # Group by stay_id
    by_stay = defaultdict(list)
    for e in events:
        by_stay[e['stay_id']].append(e)

    # Process stays
    stays_processed = 0
    all_windows = []
    all_completeness = []
    all_values = []  # For computing global z-score stats

    for stay_id, stay_events in by_stay.items():
        if max_stays and stays_processed >= max_stays:
            break

        assembler = MIMICStayAssembler(window_size=window_size)
        for e in stay_events:
            assembler.add_event(e['charttime'], e['itemid'], e['valuenum'])

        result = export_stay_tensors(assembler, np.zeros(K), np.ones(K), stay_id)

        if result['n_windows'] > 0:
            stays_processed += 1
            all_windows.append(result)
            all_completeness.append(result['completeness'])

            # Accumulate values for stats (mask=1 only)
            for w in result['windows']:
                for t in range(window_size):
                    for c in range(K):
                        if w[t, K + c] > 0:  # mask = 1
                            all_values.append(w[t, c])

        if stays_processed % 100 == 0:
            print(f"Processed {stays_processed} stays...", flush=True)

    # Compute global z-score stats
    all_values = np.array(all_values, dtype=np.float32)
    z_mu = all_values.mean(axis=0).astype(np.float32)
    z_sd = all_values.std(axis=0).astype(np.float32)
    z_sd[z_sd < 1e-6] = 1.0

    # Re-z-score all windows with correct stats
    for result in all_windows:
        for c in range(K):
            if z_sd[c] > 1e-6:
                result['windows'][:, :, c] = (result['windows'][:, :, c] - z_mu[c]) / z_sd[c]
            else:
                result['windows'][:, :, c] = result['windows'][:, :, c] - z_mu[c]

    # Save
    meta = {
        'n_stays': stays_processed,
        'total_windows': sum(r['n_windows'] for r in all_windows),
        'mean_completeness': float(np.mean(all_completeness)) if all_completeness else 0.0,
        'z_mu': z_mu.tolist(),
        'z_sd': z_sd.tolist(),
        'window_size': window_size,
        'n_features': K,
    }

    with open(os.path.join(output_dir, 'mimic_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save windows as memmap for efficiency
    n_total = sum(r['n_windows'] for r in all_windows)
    memmap_path = os.path.join(output_dir, 'mimic_windows.dat')
    mmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(n_total, window_size, K3))

    offset = 0
    for result in all_windows:
        n = result['n_windows']
        mmap[offset:offset+n] = result['windows']
        offset += n

    mmap.flush()

    print(f"\nPipeline complete:", flush=True)
    print(f"  Stays: {stays_processed}", flush=True)
    print(f"  Windows: {n_total}", flush=True)
    print(f"  Mean completeness: {meta['mean_completeness']:.3f}", flush=True)
    print(f"  Saved to {output_dir}/", flush=True)

    return meta


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python mimic_ingest.py <data_dir> <output_dir> [--max-stays N]")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    max_stays = None
    if "--max-stays" in sys.argv:
        idx = sys.argv.index("--max-stays")
        max_stays = int(sys.argv[idx + 1])

    run_pipeline(data_dir, output_dir, max_stays=max_stays)
