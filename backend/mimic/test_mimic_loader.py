"""MIMIC Backend Loader Tests — validates integration with FastAPI."""
import pytest
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.mimic.loader import (
    MIMICLoader,
    get_mimic_loader,
    load_mimic_into_backend,
)


# ─── MIMICLoader Tests ───────────────────────────────────────────────────────

class TestMIMICLoader:
    def test_init(self):
        loader = MIMICLoader()
        assert loader.patients == {}
        assert loader.z_mu is None
        assert loader.z_sd is None
        assert not loader._loaded

    def test_load_synthetic(self):
        loader = MIMICLoader()
        loader.load_synthetic(n_stays=10, seed=42)

        assert loader._loaded
        assert len(loader.patients) > 0
        assert loader.z_mu is not None
        assert loader.z_sd is not None
        assert len(loader.z_mu) == 39
        assert len(loader.z_sd) == 39

    def test_patient_format(self):
        """Patients should be in backend-compatible format."""
        loader = MIMICLoader()
        loader.load_synthetic(n_stays=5, seed=42)

        patient_dict = loader.get_patient_dict()
        assert len(patient_dict) > 0

        for pid, pdata in patient_dict.items():
            assert 'n_hours' in pdata
            assert 'ffilled' in pdata
            assert 'mask' in pdata
            assert 'delta' in pdata
            assert 'labels' in pdata
            assert pdata['n_hours'] >= 14  # At least window size

    def test_z_score_applied(self):
        """Z-scored values should have mean ~0, std ~1."""
        loader = MIMICLoader()
        loader.load_synthetic(n_stays=20, seed=42)

        # Collect all observed values
        all_z = []
        for pid, pdata in loader.patients.items():
            values = pdata['values']
            mask = pdata['mask']
            for c in range(39):
                observed = values[mask[:, c] > 0, c]
                if len(observed) > 0:
                    all_z.extend(observed.tolist())

        all_z = np.array(all_z)
        assert abs(all_z.mean()) < 0.5  # Should be close to 0
        assert 0.5 < all_z.std() < 2.0  # Should be around 1

    def test_summary(self):
        """Summary should return meaningful statistics."""
        loader = MIMICLoader()
        loader.load_synthetic(n_stays=10, seed=42)

        summary = loader.summary()
        assert summary['loaded'] is True
        assert summary['n_patients'] > 0
        assert 'scenarios' in summary
        assert summary['mean_hours'] > 0
        assert 0 < summary['mean_completeness'] <= 100

    def test_scenario_detection(self):
        """Should detect deteriorating/recovering/stable scenarios."""
        loader = MIMICLoader()
        loader.load_synthetic(n_stays=30, seed=42)

        scenarios = loader.summary()['scenarios']
        assert len(scenarios) > 0
        # Should have a mix of scenarios
        total = sum(scenarios.values())
        assert total == loader.summary()['n_patients']


# ─── Backend Integration Tests ────────────────────────────────────────────────

class TestBackendIntegration:
    def test_load_into_backend(self):
        """Should merge MIMIC patients into backend globals."""
        # Simulate backend globals
        backend_globals = {
            'patient_data': {'physio_001': {'n_hours': 20}},
            'z_mu': None,
            'z_sd': None,
        }

        summary = load_mimic_into_backend(backend_globals, n_stays=10)

        # Should have merged
        assert len(backend_globals['patient_data']) > 1
        assert backend_globals['z_mu'] is not None
        assert backend_globals['z_sd'] is not None
        assert summary['loaded'] is True

    def test_patient_keys_prefixed(self):
        """MIMIC patient keys should be prefixed with 'mimic_'."""
        loader = MIMICLoader()
        loader.load_synthetic(n_stays=5, seed=42)

        patient_dict = loader.get_patient_dict()
        for pid in patient_dict:
            assert pid.startswith('mimic_')

    def test_global_loader_singleton(self):
        """get_mimic_loader should return the same instance."""
        loader1 = get_mimic_loader()
        loader2 = get_mimic_loader()
        assert loader1 is loader2


# ─── Tensor Shape Tests ──────────────────────────────────────────────────────

class TestTensorShapes:
    def test_window_compatibility(self):
        """MIMIC windows should be compatible with backend's zscore_window."""
        loader = MIMICLoader()
        loader.load_synthetic(n_stays=5, seed=42)

        for pid, pdata in loader.patients.items():
            values = pdata['values']
            mask = pdata['mask']
            delta = pdata['delta']

            n_hours = len(values)
            assert values.shape == (n_hours, 39)
            assert mask.shape == (n_hours, 39)
            assert delta.shape == (n_hours, 39)

            # All values should be finite
            assert np.isfinite(values).all()
            assert np.isfinite(mask).all()
            assert np.isfinite(delta).all()

            # Mask should be 0 or 1
            assert np.all((mask >= 0) & (mask <= 1))

            # Delta should be non-negative and capped
            assert np.all(delta >= 0)
            assert np.all(delta <= 48.0)
