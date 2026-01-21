# Gemini MCP Server

Google Cloud Run에서 실행되는 Gemini AI 연동 MCP (Model Context Protocol) 서버입니다.

## 기능

- **gemini_generate**: Gemini AI를 사용한 텍스트 생성
- **gemini_summarize**: 텍스트 요약
- **gemini_translate**: 다국어 번역
- **gemini_analyze**: 텍스트 분석 및 질문 답변
- **gemini_code_review**: 코드 리뷰 및 개선 제안

## 배포

### 환경 변수 설정

Cloud Run 서비스에 다음 환경 변수를 설정하세요:

```
GEMINI_API_KEY=your-gemini-api-key
```

### Cloud Run 배포

```bash
# 소스에서 직접 배포
gcloud run deploy mcp-server-gemini \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your-api-key
```

## 로컬 테스트

```bash
# 의존성 설치
uv sync

# 환경 변수 설정
export GEMINI_API_KEY=your-api-key

# 서버 실행
uv run server.py
```

## MCP 클라이언트 연결

```python
from fastmcp import Client

async def test():
    async with Client("http://your-cloud-run-url/mcp") as client:
        result = await client.call_tool(
            "gemini_generate",
            {"prompt": "안녕하세요, 자기소개 해주세요"}
        )
        print(result)
```

## Transport

이 서버는 **Streamable HTTP** transport를 사용합니다. 엔드포인트: `/mcp`

## 라이선스

MIT License
