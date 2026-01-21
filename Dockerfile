# 1. Python 버전 지정 (가볍고 안정적인 버전)
FROM python:3.10-slim

# 2. 컨테이너 내 작업 폴더 설정
WORKDIR /app

# 3. 필요한 파일 복사 (requirements.txt가 있다면)
# COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"  # FastAPI와 Uvicorn 설치

# 4. 소스 코드 전체 복사
COPY . .

# 5. 서버 실행 (Cloud Run은 기본적으로 8080 포트를 사용합니다)
# main.py 파일에 app이 있다고 가정할 때:
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]