FROM python:3.11-slim

WORKDIR /app

# System deps for torch + seaborn + matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn fastapi

COPY . .

# Create required directories
RUN mkdir -p data/raw data/processed models/saved results/plots

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
