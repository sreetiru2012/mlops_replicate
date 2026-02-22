FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV MODEL_PATH=/app/artifacts/model.pt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]