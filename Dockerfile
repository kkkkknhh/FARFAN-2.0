FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema y compiladores, luego setuptools
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libopenblas-dev \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir setuptools wheel

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelo spaCy
RUN python -m spacy download es_core_news_lg

# Copiar código
COPY . .

# Comando por defecto
CMD ["python", "orchestrator.py", "--help"]
