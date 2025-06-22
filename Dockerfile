# Dockerfile
FROM bikram2docker/face-api-base

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9010"]
