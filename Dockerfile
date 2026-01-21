FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# UV 설치
RUN pip install uv

# 의존성 파일 복사
COPY pyproject.toml .

# 의존성 설치
RUN uv pip install --system -e .

# 소스 코드 복사
COPY server.py .

# 포트 설정
ENV PORT=8080
EXPOSE 8080

# 서버 실행
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
