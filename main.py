from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Cloud Run is working!"}

# 서버 실행 (이 부분은 Docker가 CMD로 실행하지만, 로컬 테스트용으로 남겨둡니다)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)