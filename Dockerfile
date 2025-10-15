FROM python:3.12-slim

WORKDIR /app

# Install system dependencies, Python packages, and spaCy model in single layer
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libopenblas-dev \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir setuptools wheel

# Copy requirements and install Python dependencies + spaCy model
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download es_core_news_lg

# Copy application code
COPY . .

# Default command
CMD ["python", "orchestrator.py", "--help"]
