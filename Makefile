.PHONY: install download train evaluate dashboard api test lint docker-build clean

install:
	pip install -r requirements.txt

download:
	python main.py download

train:
	python main.py train --model all --task both

train-cv:
	python -c "from src.training.trainer import train_models; train_models('all', 'both', run_cv=True)"

evaluate:
	python main.py evaluate

simulate:
	python main.py simulate --n 500 --output data/simulated_traffic.csv

detect:
	python main.py detect --input data/simulated_traffic.csv --model rf

dashboard:
	streamlit run app/dashboard.py

api:
	uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ app/ --ignore E501,F401

docker-build:
	docker build -t ai-ids:latest .

docker-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf results/plots/*.png results/*.json

# Full pipeline: download → train → evaluate → launch dashboard
all: download train evaluate dashboard
