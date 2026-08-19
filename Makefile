.PHONY: setup download-ckpt build run stop clean

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
	@echo ""
	@echo "Wait ~10s for the backend to load the model..."

# Stop the dashboard
stop:
	docker compose down

# Clean up
clean:
	docker compose down --rmi local
	rm -rf backend/checkpoints/* data/*
