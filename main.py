from fastapi import FastAPI
from mcp.server.fastapi import McpServer
import uvicorn

# 1. FastAPI 앱 생성
app = FastAPI()

# 2. MCP 서버 생성 (이름은 원하는 대로)
mcp = McpServer("gemini-mcp-server")

# 3. 간단한 툴 예시 (Gemini가 사용할 도구)
@mcp.tool()
def calculate_sum(a: int, b: int) -> int:
    """두 숫자의 합을 계산합니다."""
    return a + b

# 4. SSE(Server-Sent Events) 엔드포인트 연결 (중요!)
mcp.mount_sse_messages(app, "/sse")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)