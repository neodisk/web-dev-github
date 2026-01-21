from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Alive", "message": "Cloud Run is working correctly!"}

# Docker에서 실행될 때 uvicorn이 이 파일을 로드합니다.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)