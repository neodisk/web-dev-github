# 1. 파이썬 버전
FROM python:3.10-slim

# 2. 로그 즉시 출력 설정 (디버깅용)
ENV PYTHONUNBUFFERED=1

# 3. 작업 폴더
WORKDIR /app

# 4. [핵심 수정] sse-starlette 및 필수 라이브러리 모두 설치
# mcp를 쓸 때 sse-starlette가 없으면 에러가 나는 경우가 많습니다.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" mcp google-generativeai sse-starlette httpx

# 5. 소스 코드 복사
COPY . .

# 6. 서버 실행 (포트 8080)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]