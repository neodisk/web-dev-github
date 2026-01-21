# 1. 파이썬 버전 지정
FROM python:3.10-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. [중요] requirements.txt 파일 없이 직접 설치
# (COPY requirements.txt ... 이 부분을 아예 없앰)
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

# 4. 소스 코드 복사 (이제 requirements.txt가 없어도 에러 안 남)
COPY . .

# 5. 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
