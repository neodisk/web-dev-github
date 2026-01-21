# Gemini MCP Server Dockerfile for Cloud Run
# Python 3.13 + uv package manager

FROM python:3.13-slim

# Install uv (빠른 패키지 매니저)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 파일 복사
COPY pyproject.toml .
COPY server.py .

# Python 로그 즉시 출력
ENV PYTHONUNBUFFERED=1

# 의존성 설치
RUN uv sync

# Cloud Run은 PORT 환경변수를 자동 설정
EXPOSE $PORT

# uvicorn으로 FastAPI 앱 실행
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
