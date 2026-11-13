PY ?= .venv/bin/python
SF := $(PY) -m salesforecast

.PHONY: setup data validate features train backtest predict monitor pipeline serve dashboard test lint format docker-build docker-up clean

setup:            ## Create a virtualenv and install the package with all extras
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[serve,dashboard,dev]"

data:             ## Download the raw dataset
	$(SF) download

validate:         ## Data-quality checks
	$(SF) validate

features:         ## Build the feature store
	$(SF) features

train:            ## Train, evaluate and register a model
	$(SF) train

backtest:         ## Rolling-origin backtest
	$(SF) backtest

predict:          ## Score the forecast month
	$(SF) predict

monitor:          ## Feature drift report
	$(SF) monitor

pipeline:         ## Run every stage end to end (with backtest)
	$(SF) all --backtest

serve:            ## Start the prediction API on :8000
	$(PY) -m uvicorn salesforecast.serving.api:app --host 0.0.0.0 --port 8000

dashboard:        ## Start the Streamlit dashboard on :8501
	$(PY) -m streamlit run app/dashboard.py

test:             ## Run the test suite
	$(PY) -m pytest

lint:             ## Static checks
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:           ## Auto-format
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

docker-build:
	docker compose build

docker-up:        ## Serve API + dashboard from containers (run the pipeline first)
	docker compose up -d api dashboard

clean:            ## Remove generated data, models and caches
	rm -rf data/processed artifacts .pytest_cache .ruff_cache
