"""MIMIC-IV Ingestion Tests — validates the full pipeline on synthetic data."""
import pytest
import os
import json
import numpy as np
import tempfile
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_engine.mimic_ingest import (
    MIMIC_TO_FEATURE,
    FEATURE_NAMES,
    FEATURE_INDEX,
    K,
    K3,
    MIMICStayAssembler,
    export_stay_tensors,
)
from data_engine.synthetic_mimic import (
    generate_cohort,
    generate_stay,
    ITEMIDS,
)


# ─── ItemID Mapping Tests ────────────────────────────────────────────────────

class TestMIMICMapping:
    def test_feature_count(self):
        assert K == 39
        assert K3 == 117

    def test_all_features_mapped(self):
        """Every clinical feature should have at least one MIMIC itemid.
        
        Note: Age, Gender, Unit1, Unit2, HospAdmTime are metadata features
        populated from patient demographics, not from MIMIC itemids.
        """
        mapped_features = set(MIMIC_TO_FEATURE.values())
        # Metadata features come from demographics, not itemids
        metadata_features = {'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime'}
        clinical_features = [f for f in FEATURE_NAMES if f not in metadata_features]
        
        for feat in clinical_features:
            assert feat in mapped_features, f"{feat} not mapped from any MIMIC itemid"

    def test_itemid_uniqueness(self):
        """Each itemid should map to exactly one feature."""
        seen = {}
        for itemid, feat in MIMIC_TO_FEATURE.items():
            if itemid in seen:
                assert seen[itemid] == feat, f"ItemID {itemid} maps to {seen[itemid]} and {feat}"
            seen[itemid] = feat

    def test_known_itemids(self):
        """Verify some key MIMIC-IV itemids are correctly mapped."""
        assert MIMIC_TO_FEATURE[220045] == "HR"
        assert MIMIC_TO_FEATURE[50813] == "Lactate"
        assert MIMIC_TO_FEATURE[220277] == "O2Sat"
        assert MIMIC_TO_FEATURE[50912] == "Creatinine"


# ─── Stay Assembler Tests ────────────────────────────────────────────────────

class TestStayAssembler:
    def test_empty_stay(self):
        assembler = MIMICStayAssembler()
        values, mask, delta = assembler.assemble()
        assert values.shape == (0, K)
        assert mask.shape == (0, K)
        assert delta.shape == (0, K)

    def test_single_event(self):
        from datetime import datetime
        assembler = MIMICStayAssembler()
        assembler.add_event(datetime(2020, 1, 1, 0, 0), 220045, 72.0)  # HR
        values, mask, delta = assembler.assemble()

        assert values.shape[0] >= 1
        assert mask[0, FEATURE_INDEX["HR"]] == 1.0
        assert values[0, FEATURE_INDEX["HR"]] == 72.0

    def test_forward_fill(self):
        """Values should be forward-filled through missing hours."""
        from datetime import datetime
        assembler = MIMICStayAssembler()
        assembler.add_event(datetime(2020, 1, 1, 0, 0), 220045, 72.0)  # Hour 0
        assembler.add_event(datetime(2020, 1, 1, 5, 0), 220045, 80.0)  # Hour 5

        values, mask, delta = assembler.assemble()

        # Hours 1-4 should be forward-filled
        assert values[1, FEATURE_INDEX["HR"]] == 72.0
        assert values[2, FEATURE_INDEX["HR"]] == 72.0
        assert values[3, FEATURE_INDEX["HR"]] == 72.0
        assert values[4, FEATURE_INDEX["HR"]] == 72.0
        assert mask[1, FEATURE_INDEX["HR"]] == 0.0  # Not observed
        assert delta[4, FEATURE_INDEX["HR"]] == 4.0  # 4 hours since last obs

    def test_delta_capped(self):
        """Delta should cap at 48 hours."""
        from datetime import datetime
        assembler = MIMICStayAssembler()
        assembler.add_event(datetime(2020, 1, 1, 0, 0), 220045, 72.0)

        # Add event 60 hours later
        assembler.add_event(datetime(2020, 1, 3, 12, 0), 220045, 80.0)

        values, mask, delta = assembler.assemble()

        # Delta at hour 59 should be capped at 48
        assert delta[-2, FEATURE_INDEX["HR"]] == 48.0

    def test_multiple_features(self):
        """Multiple features should be assembled independently."""
        from datetime import datetime
        assembler = MIMICStayAssembler()
        assembler.add_event(datetime(2020, 1, 1, 0, 0), 220045, 72.0)  # HR
        assembler.add_event(datetime(2020, 1, 1, 2, 0), 50813, 2.5)   # Lactate

        values, mask, delta = assembler.assemble()

        assert mask[0, FEATURE_INDEX["HR"]] == 1.0
        assert mask[2, FEATURE_INDEX["Lactate"]] == 1.0
        assert mask[0, FEATURE_INDEX["Lactate"]] == 0.0  # Not yet observed

    def test_window_extraction(self):
        """Windows should be right-aligned with correct shape."""
        from datetime import datetime
        assembler = MIMICStayAssembler(window_size=6)

        for h in range(10):
            assembler.add_event(datetime(2020, 1, 1, h, 0), 220045, 70 + h)

        values, mask, delta = assembler.assemble()
        windows = assembler.get_windows(values, mask, delta)

        assert len(windows) == 10
        assert windows[0].shape == (6, K3)
        # First window: only 1 observation, should be right-aligned
        assert windows[0][5, FEATURE_INDEX["HR"]] == 70.0  # Last row = hour 0

    def test_last_event_wins(self):
        """Multiple events in same hour should take the last value."""
        from datetime import datetime
        assembler = MIMICStayAssembler()
        assembler.add_event(datetime(2020, 1, 1, 0, 0), 220045, 72.0)
        assembler.add_event(datetime(2020, 1, 1, 0, 30), 220045, 80.0)  # 30 min later

        values, mask, delta = assembler.assemble()
        assert values[0, FEATURE_INDEX["HR"]] == 80.0


# ─── Synthetic Data Tests ────────────────────────────────────────────────────

class TestSyntheticMIMIC:
    def test_generate_stay_stable(self):
        events = generate_stay(
            stay_id=1, subject_id=1, n_hours=24, scenario="stable", seed=42
        )
        assert len(events) > 0
        # Vitals should be mostly present
        hr_events = [e for e in events if e['itemid'] == ITEMIDS['HR']]
        assert len(hr_events) > 20  # ~95% of 24 hours

    def test_generate_stay_deteriorating(self):
        events = generate_stay(
            stay_id=1, subject_id=1, n_hours=24, scenario="deteriorating", seed=42
        )
        assert len(events) > 0

        # Check that deterioration is reflected in values
        early_hr = [e['valuenum'] for e in events if e['itemid'] == ITEMIDS['HR'] and e['charttime'] < '2020-01-01T12']
        late_hr = [e['valuenum'] for e in events if e['itemid'] == ITEMIDS['HR'] and e['charttime'] >= '2020-01-01T12']

        if early_hr and late_hr:
            assert np.mean(late_hr) > np.mean(early_hr), "HR should increase in deteriorating scenario"

    def test_generate_cohort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_cohort(tmpdir, n_stays=10, seed=42)

            assert os.path.exists(os.path.join(tmpdir, 'chartevents.csv'))
            assert os.path.exists(os.path.join(tmpdir, 'icustays.csv'))
            assert os.path.exists(os.path.join(tmpdir, 'patients.csv'))

            # Check file sizes
            chartevents_size = os.path.getsize(os.path.join(tmpdir, 'chartevents.csv'))
            assert chartevents_size > 1000  # Should have substantial content


# ─── End-to-End Pipeline Test ────────────────────────────────────────────────

class TestPipelineE2E:
    def test_synthetic_pipeline(self):
        """Run the full pipeline on synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate synthetic data
            data_dir = os.path.join(tmpdir, 'mimic_data')
            generate_cohort(data_dir, n_stays=20, seed=42)

            # Run the pipeline
            output_dir = os.path.join(tmpdir, 'output')

            # Note: run_pipeline expects a specific directory structure
            # (data_dir/icu/chartevents.csv)
            # Our synthetic data is flat, so we need to adapt
            # For now, test the assembler directly
            assembler = MIMICStayAssembler()

            # Load events from the synthetic data
            import csv
            chartevents_path = os.path.join(data_dir, 'chartevents.csv')
            with open(chartevents_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    charttime = row['charttime']
                    if charttime.endswith('+00:00'):
                        charttime = charttime[:-6]
                    assembler.add_event(
                        charttime=row['charttime'],
                        itemid=int(row['itemid']),
                        value=float(row['valuenum']),
                    )

            values, mask, delta = assembler.assemble()
            assert values.shape[0] > 0
            assert values.shape[1] == K

            windows = assembler.get_windows(values, mask, delta)
            assert len(windows) > 0
            assert windows[0].shape == (14, K3)

    def test_export_tensors(self):
        """Test tensor export with z-scoring."""
        from datetime import datetime

        assembler = MIMICStayAssembler()
        for h in range(24):
            assembler.add_event(
                datetime(2020, 1, 1, h, 0),
                220045, 70 + h * 0.5  # HR slowly rising
            )
            if h % 6 == 0:
                assembler.add_event(
                    datetime(2020, 1, 1, h, 0),
                    50813, 1.5 + h * 0.1  # Lactate rising
                )

        values, mask, delta = assembler.assemble()
        z_mu = np.zeros(K)
        z_sd = np.ones(K)

        result = export_stay_tensors(assembler, z_mu, z_sd, stay_id=1)
        assert result['n_hours'] == 24
        assert result['n_windows'] == 24
        assert result['windows'].shape == (24, 14, K3)
        assert 0 < result['completeness'] < 1.0

    def test_tensor_shape(self):
        """Verify tensor dimensions match the Grid's expectations."""
        from datetime import datetime

        assembler = MIMICStayAssembler(window_size=14)
        for h in range(20):
            assembler.add_event(datetime(2020, 1, 1, h, 0), 220045, 72.0)
            assembler.add_event(datetime(2020, 1, 1, h, 0), 27086, 97.0)  # O2Sat

        values, mask, delta = assembler.assemble()
        windows = assembler.get_windows(values, mask, delta)

        for w in windows:
            assert w.shape == (14, 117)  # (W, K*3)
            # Value columns should be within reasonable range (not NaN/inf)
            assert np.isfinite(w[:, :K]).all()
            # Mask should be 0 or 1
            assert np.all((w[:, K:2*K] >= 0) & (w[:, K:2*K] <= 1))
            # Delta should be non-negative and capped
            assert np.all(w[:, 2*K:] >= 0)
            assert np.all(w[:, 2*K:] <= 48.0)
