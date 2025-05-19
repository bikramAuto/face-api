# Dockerfile
FROM face-api-base

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9010

CMD ["python", "main.py"]
