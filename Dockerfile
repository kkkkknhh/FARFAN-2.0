FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema y compiladores
RUN apt-get update && apt-get install -y \
    gcc g++ make \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# CRÍTICO: Instalar setuptools primero
RUN pip install --no-cache-dir setuptools wheel

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
