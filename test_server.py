"""
Gemini MCP Server 테스트 클라이언트
Cloud Run 배포 후 테스트용
"""

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def test_server(server_url: str):
    """MCP 서버 테스트"""
    print(f"🔗 Connecting to: {server_url}")
    
    async with streamablehttp_client(server_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            # 초기화
            await session.initialize()
            print("✅ Connected to MCP server")
            
            # 사용 가능한 도구 목록
            tools = await session.list_tools()
            print(f"\n📋 Available tools ({len(tools.tools)}):")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:50]}...")
            
            # gemini_generate 테스트
            print("\n🧪 Testing gemini_generate...")
            result = await session.call_tool(
                "gemini_generate",
                arguments={
                    "prompt": "파이썬으로 'Hello, World!'를 출력하는 코드를 작성해주세요.",
                    "max_tokens": 500
                }
            )
            print(f"Result: {result.content[0].text[:500]}...")
            
            # gemini_summarize 테스트
            print("\n🧪 Testing gemini_summarize...")
            test_text = """
            인공지능(AI)은 기계가 인간의 지능을 모방하여 학습, 문제 해결, 
            패턴 인식 등의 작업을 수행할 수 있게 하는 기술입니다. 
            AI는 머신러닝, 딥러닝, 자연어 처리 등 다양한 분야를 포함합니다.
            최근에는 ChatGPT, Gemini 등의 대규모 언어 모델이 주목받고 있으며,
            이들은 자연스러운 대화와 텍스트 생성이 가능합니다.
            """
            result = await session.call_tool(
                "gemini_summarize",
                arguments={
                    "text": test_text,
                    "style": "bullet_points",
                    "language": "ko"
                }
            )
            print(f"Result: {result.content[0].text}")
            
            # gemini_translate 테스트
            print("\n🧪 Testing gemini_translate...")
            result = await session.call_tool(
                "gemini_translate",
                arguments={
                    "text": "안녕하세요, 만나서 반갑습니다!",
                    "source_language": "ko",
                    "target_language": "en"
                }
            )
            print(f"Result: {result.content[0].text}")
            
            print("\n✅ All tests completed!")


if __name__ == "__main__":
    import sys
    
    # 기본 URL 또는 명령줄 인자에서 URL 받기
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "http://localhost:8080/mcp"
    
    print("=" * 60)
    print("Gemini MCP Server Test Client")
    print("=" * 60)
    
    asyncio.run(test_server(url))
