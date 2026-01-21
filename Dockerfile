# 1. Python base image
FROM python:3.10-slim

# 2. Logs output immediately
ENV PYTHONUNBUFFERED=1

# 3. Working directory
WORKDIR /app

# 4. Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy source code
COPY . .

# 6. Run server on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
