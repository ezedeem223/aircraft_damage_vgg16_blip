PYTHON ?= python
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

format:
	black app scripts src tests

lint:
	ruff check app scripts src tests

test:
	pytest

run-demo:
	$(PYTHON) scripts/run_demo.py
