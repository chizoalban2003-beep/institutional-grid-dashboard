.PHONY: setup download-ckpt build run stop clean test test-fhir test-mimic test-loader fhir-demo

# Download trained checkpoint from Kaggle
download-ckpt:
	mkdir -p backend/checkpoints data
	~/.venvs/kaggle-tools/bin/kaggle kernels output albanchigozirim/grid-physionet-pie-chart-loss -p backend/checkpoints/
	@echo "Checkpoint downloaded to backend/checkpoints/"

# Generate demo patient data from local PhysioNet files
generate-data:
	python3 -c "\
	import sys; sys.path.insert(0, 'backend'); \
	from main import generate_demo_patients; \
	generate_demo_patients(); \
	print('Demo patients generated in data/')"

# Build Docker images
build:
	docker compose build

# Run the dashboard
run:
	docker compose up -d
	@echo ""
	@echo "Dashboard: http://localhost:8501"
	@echo "API:       http://localhost:8000/docs"
	@echo "FHIR API:  http://localhost:8000/fhir/audit"
	@echo ""
	@echo "Wait ~10s for the backend to load the model..."

# Run backend only (for FHIR development)
run-api:
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Stop the dashboard
stop:
	docker compose down

# Clean up
clean:
	docker compose down --rmi local
	rm -rf backend/checkpoints/* data/*

# Run all tests
test:
	python3 -m pytest bridge/test_fhir_bridge.py data_engine/test_mimic_ingest.py backend/mimic/test_mimic_loader.py -v

# Run FHIR bridge tests only
test-fhir:
	python3 -m pytest bridge/test_fhir_bridge.py -v --tb=short

# Run MIMIC ingestion tests only
test-mimic:
	python3 -m pytest data_engine/test_mimic_ingest.py -v --tb=short

# Run MIMIC backend loader tests
test-loader:
	python3 -m pytest backend/mimic/test_mimic_loader.py -v --tb=short

# Generate synthetic MIMIC data for pipeline testing
mimic-generate:
	python3 data_engine/synthetic_mimic.py --output /tmp/synthetic_mimic --n-stays 100

# Run the FHIR demo (mock EHR client → live backend)
fhir-demo:
	@echo "Starting FHIR demo — ensure backend is running on :8000"
	python3 bridge/mock_ehr_client.py --scenario deteriorating --n-hours 24 --interval 0.5

# Run the FHIR healthy demo
fhir-demo-healthy:
	@echo "Starting FHIR healthy patient demo"
	python3 bridge/mock_ehr_client.py --scenario healthy --n-hours 24 --interval 0.5

# Generate FHIR example bundles for documentation
fhir-examples:
	python3 -c "\
	import json; \
	from bridge.mock_ehr_client import scenario_healthy, scenario_deteriorating; \
	print(json.dumps(scenario_healthy(0), indent=2)); \
	print('---'); \
	print(json.dumps(scenario_deteriorating(0), indent=2))"
