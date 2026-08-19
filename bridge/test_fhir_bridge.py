"""FHIR Bridge Tests — validates the FHIR-to-tensor translation pipeline."""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bridge.fhir_bridge import (
    PatientState,
    FeatureState,
    FHIR_TO_FEATURE,
    FEATURE_NAMES,
    K,
    K3,
    ingest_fhir_bundle,
    build_diagnostic_report,
)


# ─── FeatureState Tests ──────────────────────────────────────────────────────

class TestFeatureState:
    def test_initial_state(self):
        fs = FeatureState()
        assert fs.last_value is None
        assert fs.last_observed_at is None
        assert fs.n_missing == 0
        assert fs.time_since_observation == float('inf')

    def test_update_value(self):
        import time
        fs = FeatureState()
        fs.last_value = 72.0
        fs.last_observed_at = time.time()
        assert fs.time_since_observation < 1.0

    def test_missing_tracking(self):
        fs = FeatureState()
        fs.n_missing = 5
        assert fs.n_missing == 5


# ─── PatientState Tests ──────────────────────────────────────────────────────

class TestPatientState:
    def test_init(self):
        ps = PatientState(patient_id="P001")
        assert ps.patient_id == "P001"
        assert len(ps.features) == K
        assert ps.hour == 0

    def test_ingest_observation(self):
        ps = PatientState(patient_id="P001")
        obs = {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
            "valueQuantity": {"value": 72, "unit": "bpm"},
            "effectiveDateTime": "2024-01-15T10:30:00Z",
            "status": "final",
        }
        ps.ingest_observation(obs)
        assert ps.features["HR"].last_value == 72.0
        assert ps.features["HR"].total_observed == 1

    def test_ingest_missing_observation(self):
        ps = PatientState(patient_id="P001")
        obs = {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
            "status": "pending",
            "effectiveDateTime": "2024-01-15T10:30:00Z",
        }
        ps.ingest_observation(obs)
        assert ps.features["HR"].last_value is None
        assert ps.features["HR"].n_missing == 1

    def test_ingest_unknown_code(self):
        ps = PatientState(patient_id="P001")
        obs = {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "99999-9"}]},
            "valueQuantity": {"value": 42},
        }
        ps.ingest_observation(obs)
        # Should not crash, just skip
        assert all(fs.last_value is None for fs in ps.features.values())

    def test_get_current_vector(self):
        ps = PatientState(patient_id="P001")
        # Set some values
        ps.features["HR"].last_value = 72.0
        ps.features["HR"].last_observed_at = 1700000000
        ps.features["O2Sat"].last_value = 97.0
        ps.features["O2Sat"].last_observed_at = 1700000000

        values, mask, delta = ps.get_current_vector()
        assert values.shape == (K,)
        assert mask.shape == (K,)
        assert delta.shape == (K,)
        assert mask[FEATURE_NAMES.index("HR")] == 1.0
        assert mask[FEATURE_NAMES.index("O2Sat")] == 1.0
        assert mask[FEATURE_NAMES.index("Lactate")] == 0.0  # Not observed

    def test_get_window_tensor(self):
        ps = PatientState(patient_id="P001")
        ps.features["HR"].last_value = 72.0
        ps.features["HR"].last_observed_at = 1700000000

        z_mu = np.zeros(K, dtype=np.float32)
        z_sd = np.ones(K, dtype=np.float32)

        tensor = ps.get_window_tensor(z_mu, z_sd)
        assert tensor.shape == (ps.window_size, K3)
        # Last row should have the observed values
        assert tensor[-1, FEATURE_NAMES.index("HR")] == 72.0  # z-scored with mu=0, sd=1


# ─── Bundle Ingestion Tests ──────────────────────────────────────────────────

class TestIngestFHIRBundle:
    def test_simple_bundle(self):
        ps = PatientState(patient_id="P001")
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                    "valueQuantity": {"value": 72},
                    "status": "final",
                }},
                {"resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2708-6"}]},
                    "valueQuantity": {"value": 97},
                    "status": "final",
                }},
            ]
        }
        result = ingest_fhir_bundle(ps, bundle)
        assert result["n_processed"] == 2
        assert result["n_skipped"] == 0
        assert result["current_hour"] == 1
        assert "HR" in result["features_updated"]
        assert "O2Sat" in result["features_updated"]

    def test_bundle_with_condition(self):
        ps = PatientState(patient_id="P001")
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {
                    "resourceType": "Condition",
                    "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "A41.9"}]},
                }},
            ]
        }
        result = ingest_fhir_bundle(ps, bundle)
        assert result["n_processed"] == 1

    def test_bundle_mixed_types(self):
        ps = PatientState(patient_id="P001")
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
                    "valueQuantity": {"value": 72},
                    "status": "final",
                }},
                {"resource": {
                    "resourceType": "Patient",  # Not an Observation or Condition
                    "name": [{"text": "Test"}],
                }},
            ]
        }
        result = ingest_fhir_bundle(ps, bundle)
        assert result["n_processed"] == 1
        assert result["n_skipped"] == 1


# ─── DiagnosticReport Tests ──────────────────────────────────────────────────

class TestDiagnosticReport:
    def test_build_report_clear(self):
        report = build_diagnostic_report(
            patient_id="P001",
            prediction=0.1234,
            governance={"cost_scale": 1.0, "risk_scale": 1.0},
            jurisdiction_pies={"VITALS": {"cost": 0.33, "risk": 0.33, "neutrality": 0.33}},
            top_cells=[{"cell_id": 0, "bid": 0.5}],
            alert_triggered=False,
        )
        assert report["resourceType"] == "DiagnosticReport"
        assert report["status"] == "final"
        assert report["subject"]["reference"] == "Patient/P001"
        assert report["conclusion"] == "NO_ALERT"
        # Should have 3 result observations
        assert len(report["result"]) == 3

    def test_build_report_alert(self):
        report = build_diagnostic_report(
            patient_id="P001",
            prediction=0.8765,
            governance={},
            jurisdiction_pies={},
            top_cells=[],
            alert_triggered=True,
        )
        assert report["conclusion"] == "SEPSIS_ALERT"
        # Check the prediction observation
        pred_obs = report["result"][0]
        assert abs(pred_obs["valueQuantity"]["value"] - 0.8765) < 0.0001
        assert pred_obs["interpretation"][0]["coding"][0]["code"] == "H"


# ─── Schema Tests ────────────────────────────────────────────────────────────

class TestSchema:
    def test_feature_count(self):
        assert K == 39
        assert K3 == 117
        assert len(FEATURE_NAMES) == K

    def test_fhir_mapping_coverage(self):
        # All mapped codes should exist
        for code, name in FHIR_TO_FEATURE.items():
            if code not in ("AGE", "GENDER", "UNIT1", "UNIT2", "HOSP_ADM_TIME"):
                assert name in FEATURE_NAMES

    def test_feature_index(self):
        from bridge.fhir_bridge import FEATURE_INDEX
        assert len(FEATURE_INDEX) == K
        for i, name in enumerate(FEATURE_NAMES):
            assert FEATURE_INDEX[name] == i
