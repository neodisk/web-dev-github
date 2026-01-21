# 1. 파이썬 기본 이미지 설정 (가볍고 안정적인 버전)
FROM python:3.10-slim

# 2. 컨테이너 내 작업 폴더 설정
WORKDIR /app

# =====================================================================
# [핵심 수정] requirements.txt 파일을 복사하지 않고 직접 설치합니다.
# 이 방식을 쓰면 requirements.txt 파일이 있든 없든 상관없이 빌드가 성공합니다.
# =====================================================================
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" mcp google-generativeai

# 3. 나머지 소스 코드 복사
COPY . .

# 4. 서버 실행 명령어 설정 (Cloud Run은 기본 8080 포트 사용)
# main.py 파일 안에 app 객체가 있다고 가정합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]