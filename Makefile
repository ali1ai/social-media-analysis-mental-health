.PHONY: install install-all test lint smoke train notebook notebook-build clean

# Always use the active interpreter, so a stray system pytest/jupyter can't pick up the wrong Python.
PY ?= python

install:            ## core + Kaggle download + dev tools
	$(PY) -m pip install -e ".[data,dev]"

install-all:        ## everything incl. the optional transformer baseline
	$(PY) -m pip install -e ".[data,dev,transformer]"

lint:
	$(PY) -m ruff check src tests scripts

test: lint
	$(PY) -m pytest -q

smoke:              ## full pipeline on synthetic data, no Kaggle needed (~1-2 min)
	$(PY) scripts/make_synthetic_fixture.py data/synthetic_fixture.csv
	$(PY) -m socialmediamind.train --data data/synthetic_fixture.csv --fast
	cd notebooks && SMM_FAST=1 SMM_TRANSFORMER=0 SMM_DATA_PATH=../data/synthetic_fixture.csv \
		$(PY) -m jupyter nbconvert --to notebook --execute SocialMediaMind.ipynb --output /tmp/smm_executed.ipynb

train:              ## reproducible training on the real Kaggle dataset
	$(PY) -m socialmediamind.train

notebook:           ## run the full analysis notebook in place on the real dataset
	cd notebooks && $(PY) -m jupyter nbconvert --to notebook --execute --inplace SocialMediaMind.ipynb \
		--ExecutePreprocessor.timeout=-1

notebook-build:     ## regenerate the notebook after editing src/ or scripts/build_notebook.py
	$(PY) scripts/build_notebook.py

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
