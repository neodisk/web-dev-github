"""
Gemini AI MCP Server for Cloud Run
FastAPI + MCP integration for Cloud Run compatibility
"""

import asyncio
import logging
import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
import google.generativeai as genai

# MCP imports
from mcp.server.fastmcp import FastMCP

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(message)s",
    level=logging.INFO
)

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured")
else:
    logger.warning("⚠️ GEMINI_API_KEY not set - Gemini tools will not work")

# MCP 서버 초기화
mcp = FastMCP(
    name="gemini_mcp",
    instructions="""
    This is a Gemini AI MCP server that provides various AI-powered tools.
    Available tools:
    - gemini_generate: Generate text using Gemini AI
    - gemini_summarize: Summarize text
    - gemini_translate: Translate text between languages
    - gemini_analyze: Analyze and answer questions about text
    - gemini_code_review: Review and improve code
    """
)


# ============================================================
# Pydantic 입력 모델 정의
# ============================================================

class GenerateInput(BaseModel):
    """텍스트 생성 입력 모델"""
    prompt: str = Field(
        ...,
        description="생성할 텍스트에 대한 프롬프트 (예: '파이썬으로 피보나치 함수 작성해줘')",
        min_length=1,
        max_length=10000
    )
    model: str = Field(
        default="gemini-2.0-flash",
        description="사용할 Gemini 모델 (gemini-2.0-flash, gemini-1.5-pro 등)"
    )
    max_tokens: Optional[int] = Field(
        default=2048,
        description="생성할 최대 토큰 수",
        ge=1,
        le=8192
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="창의성 조절 (0.0=결정적, 1.0=창의적)",
        ge=0.0,
        le=1.0
    )


class SummarizeInput(BaseModel):
    """텍스트 요약 입력 모델"""
    text: str = Field(
        ...,
        description="요약할 텍스트",
        min_length=10,
        max_length=50000
    )
    style: str = Field(
        default="concise",
        description="요약 스타일: 'concise' (간결), 'detailed' (상세), 'bullet_points' (글머리 기호)"
    )
    language: str = Field(
        default="ko",
        description="출력 언어 코드 (ko=한국어, en=영어, ja=일본어 등)"
    )


class TranslateInput(BaseModel):
    """번역 입력 모델"""
    text: str = Field(
        ...,
        description="번역할 텍스트",
        min_length=1,
        max_length=10000
    )
    source_language: str = Field(
        default="auto",
        description="원본 언어 (auto=자동 감지, ko=한국어, en=영어, ja=일본어 등)"
    )
    target_language: str = Field(
        ...,
        description="번역할 대상 언어 (ko=한국어, en=영어, ja=일본어 등)"
    )


class AnalyzeInput(BaseModel):
    """텍스트 분석 입력 모델"""
    text: str = Field(
        ...,
        description="분석할 텍스트 또는 문서",
        min_length=1,
        max_length=50000
    )
    question: str = Field(
        ...,
        description="텍스트에 대해 물어볼 질문",
        min_length=1,
        max_length=1000
    )


class CodeReviewInput(BaseModel):
    """코드 리뷰 입력 모델"""
    code: str = Field(
        ...,
        description="리뷰할 코드",
        min_length=1,
        max_length=20000
    )
    language: str = Field(
        default="auto",
        description="프로그래밍 언어 (auto=자동 감지, python, javascript, typescript 등)"
    )
    focus: str = Field(
        default="all",
        description="리뷰 초점: 'all' (전체), 'security' (보안), 'performance' (성능), 'readability' (가독성)"
    )


# ============================================================
# Gemini API 헬퍼 함수
# ============================================================

def get_gemini_model(model_name: str = "gemini-2.0-flash"):
    """Gemini 모델 인스턴스 반환"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    return genai.GenerativeModel(model_name)


async def generate_content(
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> str:
    """Gemini API를 사용하여 콘텐츠 생성"""
    try:
        model = get_gemini_model(model_name)
        
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        logger.error(f"Gemini API 오류: {e}")
        raise


# ============================================================
# MCP 도구 정의
# ============================================================

@mcp.tool(
    name="gemini_generate",
    annotations={
        "title": "Gemini 텍스트 생성",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def gemini_generate(params: GenerateInput) -> str:
    """
    Gemini AI를 사용하여 텍스트를 생성합니다.
    
    다양한 용도로 사용 가능:
    - 창작 글쓰기
    - 코드 생성
    - 아이디어 브레인스토밍
    - 질문 답변
    
    Args:
        params: GenerateInput - 생성 설정
            - prompt: 생성할 내용에 대한 프롬프트
            - model: 사용할 모델 (기본: gemini-2.0-flash)
            - max_tokens: 최대 토큰 수 (기본: 2048)
            - temperature: 창의성 (0.0-1.0, 기본: 0.7)
    
    Returns:
        str: 생성된 텍스트
    """
    logger.info(f">>> 🛠️ Tool: 'gemini_generate' called with prompt length: {len(params.prompt)}")
    
    result = await generate_content(
        prompt=params.prompt,
        model_name=params.model,
        max_tokens=params.max_tokens,
        temperature=params.temperature
    )
    
    return result


@mcp.tool(
    name="gemini_summarize",
    annotations={
        "title": "텍스트 요약",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def gemini_summarize(params: SummarizeInput) -> str:
    """
    Gemini AI를 사용하여 텍스트를 요약합니다.
    
    Args:
        params: SummarizeInput - 요약 설정
            - text: 요약할 텍스트
            - style: 요약 스타일 (concise/detailed/bullet_points)
            - language: 출력 언어
    
    Returns:
        str: 요약된 텍스트
    """
    logger.info(f">>> 🛠️ Tool: 'gemini_summarize' called with text length: {len(params.text)}")
    
    style_prompts = {
        "concise": "간결하게 핵심만",
        "detailed": "상세하게",
        "bullet_points": "글머리 기호를 사용하여"
    }
    
    language_names = {
        "ko": "한국어",
        "en": "영어",
        "ja": "일본어",
        "zh": "중국어"
    }
    
    style_desc = style_prompts.get(params.style, "간결하게")
    lang_name = language_names.get(params.language, params.language)
    
    prompt = f"""다음 텍스트를 {style_desc} {lang_name}로 요약해주세요:

---
{params.text}
---

요약:"""
    
    result = await generate_content(prompt=prompt, temperature=0.3)
    return result


@mcp.tool(
    name="gemini_translate",
    annotations={
        "title": "텍스트 번역",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def gemini_translate(params: TranslateInput) -> str:
    """
    Gemini AI를 사용하여 텍스트를 번역합니다.
    
    Args:
        params: TranslateInput - 번역 설정
            - text: 번역할 텍스트
            - source_language: 원본 언어 (auto=자동 감지)
            - target_language: 대상 언어
    
    Returns:
        str: 번역된 텍스트
    """
    logger.info(f">>> 🛠️ Tool: 'gemini_translate' called: {params.source_language} -> {params.target_language}")
    
    language_names = {
        "ko": "한국어",
        "en": "영어",
        "ja": "일본어",
        "zh": "중국어",
        "es": "스페인어",
        "fr": "프랑스어",
        "de": "독일어",
        "auto": "자동 감지된 언어"
    }
    
    source_name = language_names.get(params.source_language, params.source_language)
    target_name = language_names.get(params.target_language, params.target_language)
    
    if params.source_language == "auto":
        prompt = f"""다음 텍스트를 {target_name}로 번역해주세요. 원문의 뉘앙스와 톤을 유지하세요.

원문:
{params.text}

{target_name} 번역:"""
    else:
        prompt = f"""다음 {source_name} 텍스트를 {target_name}로 번역해주세요. 원문의 뉘앙스와 톤을 유지하세요.

원문:
{params.text}

{target_name} 번역:"""
    
    result = await generate_content(prompt=prompt, temperature=0.2)
    return result


@mcp.tool(
    name="gemini_analyze",
    annotations={
        "title": "텍스트 분석 및 질문 답변",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def gemini_analyze(params: AnalyzeInput) -> str:
    """
    Gemini AI를 사용하여 텍스트를 분석하고 질문에 답변합니다.
    
    Args:
        params: AnalyzeInput - 분석 설정
            - text: 분석할 텍스트/문서
            - question: 질문
    
    Returns:
        str: 질문에 대한 답변
    """
    logger.info(f">>> 🛠️ Tool: 'gemini_analyze' called with question: {params.question[:50]}...")
    
    prompt = f"""다음 텍스트를 분석하고 질문에 답변해주세요.

텍스트:
---
{params.text}
---

질문: {params.question}

답변:"""
    
    result = await generate_content(prompt=prompt, temperature=0.3)
    return result


@mcp.tool(
    name="gemini_code_review",
    annotations={
        "title": "코드 리뷰",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def gemini_code_review(params: CodeReviewInput) -> str:
    """
    Gemini AI를 사용하여 코드를 리뷰합니다.
    
    Args:
        params: CodeReviewInput - 코드 리뷰 설정
            - code: 리뷰할 코드
            - language: 프로그래밍 언어
            - focus: 리뷰 초점 (all/security/performance/readability)
    
    Returns:
        str: 코드 리뷰 결과
    """
    logger.info(f">>> 🛠️ Tool: 'gemini_code_review' called with focus: {params.focus}")
    
    focus_prompts = {
        "all": "전반적인 품질, 버그, 보안, 성능, 가독성",
        "security": "보안 취약점과 잠재적 보안 문제",
        "performance": "성능 최적화 가능성과 병목 현상",
        "readability": "코드 가독성, 명명 규칙, 구조"
    }
    
    focus_desc = focus_prompts.get(params.focus, focus_prompts["all"])
    lang_hint = f"({params.language})" if params.language != "auto" else ""
    
    prompt = f"""다음 코드{lang_hint}를 리뷰해주세요. 특히 {focus_desc}에 초점을 맞춰주세요.

```
{params.code}
```

다음 형식으로 리뷰해주세요:

## 요약
(코드의 전반적인 평가)

## 발견된 문제점
(문제점과 개선 제안)

## 개선된 코드 (필요시)
(수정된 코드 예시)

## 추가 권장사항
(베스트 프랙티스 등)
"""
    
    result = await generate_content(prompt=prompt, temperature=0.3, max_tokens=4096)
    return result


# ============================================================
# 헬스 체크용 리소스
# ============================================================

@mcp.resource("health://status")
def health_status() -> str:
    """서버 상태 확인"""
    return "OK - Gemini MCP Server is running"


# ============================================================
# FastAPI 앱 설정
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan for MCP setup"""
    logger.info("🚀 Starting Gemini MCP Server...")
    yield
    logger.info("👋 Shutting down Gemini MCP Server...")


# FastAPI 앱 생성
app = FastAPI(
    title="Gemini MCP Server",
    description="Gemini AI MCP Server for Cloud Run",
    version="1.0.0",
    lifespan=lifespan
)


# Health check 엔드포인트 (Cloud Run용)
@app.get("/")
async def root():
    """Root health check endpoint"""
    return {"status": "ok", "service": "gemini-mcp-server"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_configured": GEMINI_API_KEY is not None
    }


# MCP SSE 앱 마운트
# FastMCP의 sse_app() 메서드로 SSE 엔드포인트 생성
app.mount("/mcp", mcp.sse_app())


# ============================================================
# 서버 실행
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    logger.info(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
