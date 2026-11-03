.PHONY: all data preprocess test run diagrams app clean

PYTHON ?= python3

all: data run diagrams test

data:
	$(PYTHON) scripts/download_data.py --dest-dir data/raw

preprocess:
	$(PYTHON) -c "import yaml; from pathlib import Path; from src.amr.features import process_and_save_features; c = yaml.safe_load(open('configs/config.yaml')); process_and_save_features(Path('data/raw/PDG000000012.2550.amr.metadata.tsv'), Path('data/processed'), c)"

test:
	$(PYTHON) -m pytest -v tests/

run:
	$(PYTHON) scripts/run_pipeline.py --config configs/config.yaml

diagrams:
	$(PYTHON) scripts/generate_diagrams.py

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/interim/*
	rm -rf __pycache__ src/amr/__pycache__ src/amr/kernels/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
