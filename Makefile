# AI_Biomed: Cross-Platform Research Makefile
.PHONY: help install test benchmark nephro figures app docker-build docker-run clean

PYTHON := python
VENV_PYTHON := .venv/Scripts/python

help:
	@echo "AI_Biomed Research & Clinical Decision Support Automation"
	@echo "Targets:"
	@echo "  make install     Install requirements and package in editable mode"
	@echo "  make test        Execute automated unit & integration test suite"
	@echo "  make benchmark   Reproduce historical experimental benchmarks"
	@echo "  make nephro      Run drug nephrotoxicity and CKD progression pipeline"
	@echo "  make figures     Generate publication-ready figures (ROC, PR, SHAP)"
	@echo "  make app         Launch interactive Streamlit CDSS web interface"
	@echo "  make docker-build Build multi-stage Docker container"
	@echo "  make docker-run   Run Streamlit CDSS in Docker container"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-build-isolation --no-deps

test:
	$(PYTHON) -m pytest tests/ -v

benchmark:
	$(PYTHON) main.py --mode benchmark

nephro:
	$(PYTHON) main.py --mode nephro

figures:
	$(PYTHON) main.py --mode figures

app:
	streamlit run app.py

docker-build:
	docker build -t ai_biomed_cdss .

docker-run:
	docker run -p 8501:8501 ai_biomed_cdss

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
