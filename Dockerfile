FROM python:3.11-slim

WORKDIR /app

# System deps (faiss, etc.)
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm

COPY . .

ENV UVICORN_HOST=0.0.0.0
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

