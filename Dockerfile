# 1. 파이썬 버전
FROM python:3.10-slim

# 2. 로그 즉시 출력 (에러 확인용 필수 설정)
ENV PYTHONUNBUFFERED=1

# 3. 작업 폴더
WORKDIR /app

# 4. [핵심] 모든 가능성에 대비해 필요한 라이브러리를 전부 설치합니다.
# main.py가 옛날 코드여도, 최신 코드여도 다 돌아가게 만듭니다.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" mcp google-generativeai sse-starlette httpx

# 5. 소스 코드 복사
COPY . .

# 6. 서버 실행 (포트 8080)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]