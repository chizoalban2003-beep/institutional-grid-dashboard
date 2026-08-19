"""MIMIC Data Loader — streams MIMIC-IV tensors to the FastAPI backend.

Provides an interface compatible with the existing backend's patient_data
format, allowing MIMIC ICU stays to be served through the same /predict
and /stream endpoints.

Usage:
    from backend.mimic.loader import MIMICLoader

    loader = MIMICLoader()
    loader.load_synthetic(n_stays=50)  # or loader.load_real('/path/to/mimic')
    patient_data = loader.get_patient_dict()  # compat with backend
"""
import os
import sys
import json
import numpy as np
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_engine.mimic_ingest import (
    MIMICStayAssembler,
    FEATURE_NAMES,
    FEATURE_INDEX,
    K,
    K3,
)
from data_engine.synthetic_mimic import generate_stay


K = 39
K3 = K * 3
WINDOW_SIZE = 14


class MIMICLoader:
    """Loads MIMIC-IV data and converts to backend-compatible format."""

    def __init__(self):
        self.patients = {}  # stay_id -> patient_data dict
        self.z_mu = None
        self.z_sd = None
        self._loaded = False

    def load_synthetic(self, n_stays: int = 50, seed: int = 42):
        """Generate synthetic MIMIC data and load into patient format.

        Creates realistic ICU stays with stable/deteriorating/recovering
        trajectories, then converts them to the backend's expected format.
        """
        import tempfile
        from data_engine.synthetic_mimic import generate_cohort
        from data_engine.mimic_ingest import MIMIC_TO_FEATURE

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate synthetic cohort
            generate_cohort(tmpdir, n_stays=n_stays, seed=seed)

            # Load chartevents
            import csv
            chartevents_path = os.path.join(tmpdir, 'chartevents.csv')

            # Group events by stay_id
            by_stay = {}
            with open(chartevents_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stay_id = int(row['stay_id'])
                    if stay_id not in by_stay:
                        by_stay[stay_id] = []
                    by_stay[stay_id].append({
                        'charttime': row['charttime'],
                        'itemid': int(row['itemid']),
                        'valuenum': float(row['valuenum']),
                    })

            # Process each stay
            all_values = [[] for _ in range(K)]  # Per-feature lists
            for stay_id, events in by_stay.items():
                assembler = MIMICStayAssembler(window_size=WINDOW_SIZE)
                for e in events:
                    assembler.add_event(e['charttime'], e['itemid'], e['valuenum'])

                values, mask, delta = assembler.assemble()
                if len(values) < WINDOW_SIZE:
                    continue  # Skip stays shorter than window

                # Accumulate for z-score stats (per feature)
                for c in range(K):
                    obs = values[mask[:, c] > 0, c]
                    if len(obs) > 0:
                        all_values[c].extend(obs.tolist())

                # Store raw data for later z-scoring
                self.patients[f"mimic_{stay_id}"] = {
                    'values': values,
                    'mask': mask,
                    'delta': delta,
                    'n_hours': len(values),
                    'n_positive': 0,  # No labels for synthetic
                    'scenario': self._detect_scenario(values, mask),
                }

            # Compute z-score stats (per feature)
            self.z_mu = np.array([np.mean(v) if v else 0.0 for v in all_values], dtype=np.float32)
            self.z_sd = np.array([np.std(v) if v else 1.0 for v in all_values], dtype=np.float32)
            self.z_sd = np.where(self.z_sd < 1e-6, 1.0, self.z_sd)

            # Z-score all patients
            for pid, pdata in self.patients.items():
                for c in range(K):
                    pdata['values'][:, c] = (pdata['values'][:, c] - self.z_mu[c]) / self.z_sd[c]

            self._loaded = True
            print(f"Loaded {len(self.patients)} synthetic MIMIC stays", flush=True)

    def load_real(self, data_dir: str, max_stays: int = None):
        """Load real MIMIC-IV data from CSV/parquet files.

        data_dir should contain:
        - icu/chartevents.csv (or .csv.gz)
        - icu/icustays.csv
        - hosp/patients.csv (optional, for demographics)
        """
        import csv
        from data_engine.mimic_ingest import MIMIC_TO_FEATURE

        chartevents_path = os.path.join(data_dir, 'icu', 'chartevents.csv')
        if not os.path.exists(chartevents_path):
            chartevents_path = os.path.join(data_dir, 'icu', 'chartevents.csv.gz')
        if not os.path.exists(chartevents_path):
            # Try flat structure
            chartevents_path = os.path.join(data_dir, 'chartevents.csv')
            if not os.path.exists(chartevents_path):
                raise FileNotFoundError(f"chartevents not found in {data_dir}")

        print(f"Loading MIMIC-IV from {chartevents_path}...", flush=True)

        # Group events by stay_id
        by_stay = {}
        n_events = 0

        # Handle both gzip and plain CSV
        open_func = open
        if chartevents_path.endswith('.gz'):
            import gzip
            open_func = lambda p: gzip.open(p, 'rt')

        with open_func(chartevents_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                stay_id = int(row['stay_id'])
                if stay_id not in by_stay:
                    by_stay[stay_id] = []

                try:
                    valuenum = float(row.get('valuenum', ''))
                    itemid = int(row['itemid'])
                except (ValueError, TypeError):
                    continue

                by_stay[stay_id].append({
                    'charttime': row['charttime'],
                    'itemid': itemid,
                    'valuenum': valuenum,
                })
                n_events += 1

                if max_stays and len(by_stay) >= max_stays * 10:
                    break  # Rough limit

        print(f"Loaded {n_events:,} events for {len(by_stay)} stays", flush=True)

        # Process stays
        all_values = []
        stays_processed = 0

        for stay_id, events in by_stay.items():
            if max_stays and stays_processed >= max_stays:
                break

            assembler = MIMICStayAssembler(window_size=WINDOW_SIZE)
            for e in events:
                assembler.add_event(e['charttime'], e['itemid'], e['valuenum'])

            values, mask, delta = assembler.assemble()
            if len(values) < WINDOW_SIZE:
                continue

            for t in range(len(values)):
                for c in range(K):
                    if mask[t, c] > 0:
                        all_values.append(values[t, c])

            self.patients[f"mimic_{stay_id}"] = {
                'values': values,
                'mask': mask,
                'delta': delta,
                'n_hours': len(values),
                'n_positive': 0,
                'scenario': 'real',
            }
            stays_processed += 1

        # Compute z-score stats
        all_values = np.array(all_values, dtype=np.float32)
        self.z_mu = all_values.mean(axis=0).astype(np.float32)
        self.z_sd = all_values.std(axis=0).astype(np.float32)
        self.z_sd[self.z_sd < 1e-6] = 1.0

        # Z-score all patients
        for pid, pdata in self.patients.items():
            for c in range(K):
                pdata['values'][:, c] = (pdata['values'][:, c] - self.z_mu[c]) / self.z_sd[c]

        self._loaded = True
        print(f"Loaded {stays_processed} MIMIC-IV stays ({len(all_values):,} observations)", flush=True)

    def _detect_scenario(self, values: np.ndarray, mask: np.ndarray) -> str:
        """Detect if this stay shows deterioration, recovery, or stability."""
        if len(values) < 6:
            return 'stable'

        # Check HR trend
        hr_idx = FEATURE_INDEX.get('HR', 0)
        hr_mask = mask[:, hr_idx] > 0
        if hr_mask.sum() < 3:
            return 'stable'

        hr_values = values[hr_mask, hr_idx]
        if len(hr_values) < 3:
            return 'stable'

        # Simple trend detection
        first_half = hr_values[:len(hr_values)//2].mean()
        second_half = hr_values[len(hr_values)//2:].mean()

        if second_half > first_half + 10:
            return 'deteriorating'
        elif second_half < first_half - 10:
            return 'recovering'
        return 'stable'

    def get_patient_dict(self) -> dict:
        """Convert to backend-compatible patient_data format.

        Returns dict compatible with the existing backend's /predict endpoint.
        """
        if not self._loaded:
            return {}

        result = {}
        for pid, pdata in self.patients.items():
            n_hours = pdata['n_hours']

            # Build ffilled arrays for compatibility
            ffilled = pdata['values'].tolist()
            mask_list = pdata['mask'].tolist()
            delta_list = pdata['delta'].tolist()

            result[pid] = {
                'n_hours': n_hours,
                'n_positive': pdata.get('n_positive', 0),
                'labels': [0] * n_hours,  # No labels for MIMIC
                'ffilled': ffilled,
                'mask': mask_list,
                'delta': delta_list,
                'scenario': pdata.get('scenario', 'unknown'),
            }

        return result

    def get_z_stats(self) -> tuple:
        """Return (z_mu, z_sd) for z-scoring."""
        return self.z_mu, self.z_sd

    def summary(self) -> dict:
        """Get summary statistics of loaded data."""
        if not self._loaded:
            return {'loaded': False}

        scenarios = {}
        n_hours_list = []
        completeness_list = []

        for pid, pdata in self.patients.items():
            scenario = pdata.get('scenario', 'unknown')
            scenarios[scenario] = scenarios.get(scenario, 0) + 1
            n_hours_list.append(pdata['n_hours'])

            mask = pdata['mask']
            if mask.size > 0:
                completeness_list.append(float(mask.sum()) / mask.size * 100)

        return {
            'loaded': True,
            'n_patients': len(self.patients),
            'scenarios': scenarios,
            'mean_hours': float(np.mean(n_hours_list)) if n_hours_list else 0,
            'median_hours': float(np.median(n_hours_list)) if n_hours_list else 0,
            'mean_completeness': float(np.mean(completeness_list)) if completeness_list else 0,
        }


# Global instance for backend integration
_mimic_loader: Optional[MIMICLoader] = None


def get_mimic_loader() -> MIMICLoader:
    """Get or create the global MIMIC loader."""
    global _mimic_loader
    if _mimic_loader is None:
        _mimic_loader = MIMICLoader()
    return _mimic_loader


def load_mimic_into_backend(backend_globals: dict, n_stays: int = 50):
    """Load MIMIC data into the backend's global state.

    This integrates MIMIC patients into the existing /predict and /stream
    endpoints alongside the PhysioNet demo patients.

    Usage in backend/main.py:
        from .mimic.loader import load_mimic_into_backend
        load_mimic_into_backend(globals(), n_stays=50)
    """
    loader = get_mimic_loader()
    loader.load_synthetic(n_stays=n_stays)

    # Merge with existing patient_data
    patient_data = backend_globals.get('patient_data', {})
    mimic_patients = loader.get_patient_dict()
    patient_data.update(mimic_patients)
    backend_globals['patient_data'] = patient_data

    # Update z-score stats if needed
    z_mu = backend_globals.get('z_mu')
    z_sd = backend_globals.get('z_sd')
    if z_mu is None or z_sd is None:
        mim_mu, mim_sd = loader.get_z_stats()
        backend_globals['z_mu'] = mim_mu
        backend_globals['z_sd'] = mim_sd

    print(f"MIMIC loaded: {len(mimic_patients)} stays merged into backend", flush=True)
    return loader.summary()
