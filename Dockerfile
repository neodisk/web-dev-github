# 1. 파이썬 버전
FROM python:3.10-slim

# 2. 로그가 버퍼링 없이 즉시 출력되도록 설정 (가장 중요!)
# 이 설정이 있어야 에러가 났을 때 로그에 바로 찍힙니다.
ENV PYTHONUNBUFFERED=1

# 3. 작업 폴더
WORKDIR /app

# 4. 필수 라이브러리 설치 (일단 mcp는 뺍니다)
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

# 5. 소스 코드 복사
COPY . .

# 6. 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]