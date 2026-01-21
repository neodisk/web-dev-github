# 1. 파이썬 버전 (3.10)
FROM python:3.10-slim

# 2. 로그가 버퍼링 없이 즉시 출력되도록 설정 (로그 확인에 필수!)
ENV PYTHONUNBUFFERED=1

# 3. 작업 폴더
WORKDIR /app

# 4. 필수 라이브러리 설치
# (팁: mcp 라이브러리가 최신인지 확인하기 위해 업그레이드 옵션을 붙입니다)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" mcp google-generativeai

# 5. 소스 코드 복사
COPY . .

# 6. 서버 실행
# --workers 1 옵션을 추가해서 가볍게 실행합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
