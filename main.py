from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Alive", "message": "Cloud Run is working!"}

# Docker가 실행할 때 이 부분이 호출됩니다.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)