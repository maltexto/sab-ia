FROM python:3.10-slim

WORKDIR /app

# Instalar ffmpeg e dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Configurar variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copiar o projeto
COPY . /app/

# Instalar o projeto e suas dependências usando pip diretamente do pyproject.toml
RUN pip install --no-cache-dir .

EXPOSE 8000

# Comando para executar a aplicação
CMD ["uvicorn", "src.sab_ia.main:app", "--host", "0.0.0.0", "--port", "8000"]