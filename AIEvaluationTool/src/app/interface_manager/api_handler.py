import re
import time
from typing import Dict, Any, List

from context import APIRuntimeContext
from logger import get_logger

from openai import OpenAI
from google import genai
import anthropic as anthropic_sdk

logger = get_logger("interface_manager")


def handle_api_chat(
    ctx: APIRuntimeContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes one API chat request and returns a normalized response.
    """

    # --------------------------------------------------
    # Driver lifecycle start
    # --------------------------------------------------
    logger.info("Driver is ready for API")

    start_ts = time.time()

    prompts: List[str] = payload.get("prompt_list", [])
    prompt = " ".join(prompts).strip()

    if not prompt:
        logger.error("Empty prompt_list received")
        raise ValueError("Empty prompt_list received")

    logger.info("Sending prompt to the bot: %s", prompt)

    logger.info(
        "API chat started | provider=%s model=%s",
        ctx.provider,
        ctx.agent_name,
    )

    try:
        # --------------------------------------------------
        # Dispatch by provider
        # --------------------------------------------------
        if ctx.is_openai():
            text = _run_openai(ctx, prompt)

        elif ctx.is_gemini():
            text = _run_gemini(ctx, prompt)

        elif ctx.is_local():
            text = _run_local(ctx, prompt)

        elif ctx.is_anthropic():
            text = _run_anthropic(ctx, prompt)

        elif ctx.is_groq():
            text = _run_groq(ctx, prompt)

        else:
            raise RuntimeError(f"Unsupported provider: {ctx.provider}")

        elapsed = int(time.time() - start_ts)

        logger.info(
            "(Waited:%d) Received response from API (%s): %s",
            elapsed,
            ctx.agent_name,
            text,
        )

        logger.info(
            "API chat completed | chars=%d time=%ss",
            len(text),
            round(time.time() - start_ts, 3),
        )

        return {
            "response": [
                {
                    "chat_id": payload.get("chat_id"),
                    "prompt": prompt,
                    "response": {
                        "type": "text",
                        "content": text
                    }
                }
            ]
        }

    finally:
        # --------------------------------------------------
        # Driver lifecycle end (always runs)
        # --------------------------------------------------
        logger.info("Driver quit successfully")


# ------------------------------------------------------------------
# Provider implementations
# ------------------------------------------------------------------

def _run_openai(ctx: APIRuntimeContext, prompt: str) -> str:
    logger.info("Calling OpenAI API | model=%s", ctx.agent_name)

    client = OpenAI()

    response = client.chat.completions.create(
        model=ctx.agent_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=ctx.temperature,
        max_tokens=ctx.max_tokens,
        top_p=ctx.top_p,
    )

    return response.choices[0].message.content.strip()


def _run_gemini(ctx: APIRuntimeContext, prompt: str) -> str:
    logger.info("Calling Gemini API | model=%s", ctx.agent_name)

    client = genai.Client()
    max_retries = 4

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=ctx.agent_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"retryDelay.*?'(\d+)s'", err_str)
                wait = int(match.group(1)) + 2 if match else min(10 * (2 ** attempt), 120)
                if attempt < max_retries - 1:
                    logger.warning(
                        "Gemini 429 rate limit — waiting %ds before retry %d/%d",
                        wait, attempt + 1, max_retries - 1,
                    )
                    time.sleep(wait)
                else:
                    raise
            else:
                raise


def _run_anthropic(ctx: APIRuntimeContext, prompt: str) -> str:
    logger.info("Calling Anthropic API | model=%s", ctx.agent_name)

    client = anthropic_sdk.Anthropic()

    message = client.messages.create(
        model=ctx.agent_name,
        max_tokens=ctx.max_tokens or 1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()


def _run_groq(ctx: APIRuntimeContext, prompt: str) -> str:
    logger.info("Calling Groq API | model=%s", ctx.agent_name)
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    # Try context extra first (passed by client), then env, then .env file
    key = (ctx.extra or {}).get("api_key")
    if not key:
        _env_file = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(_env_file, override=False)
        key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY — set in .env or pass via api_context.extra.api_key")
    client = OpenAI(
        api_key=key,
        base_url=ctx.base_url or "https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=ctx.agent_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=ctx.temperature,
        max_tokens=ctx.max_tokens or 1024,
        top_p=ctx.top_p,
    )
    return response.choices[0].message.content.strip()


def _run_local(ctx: APIRuntimeContext, prompt: str) -> str:
    logger.info(
        "Calling LOCAL OpenAI-compatible API | model=%s base_url=%s",
        ctx.agent_name,
        ctx.base_url,
    )

    if not ctx.base_url:
        raise RuntimeError("LOCAL provider requires base_url")

    client = OpenAI(
        base_url=f"{ctx.base_url.rstrip('/')}/v1",
        api_key="local",   # required but unused
    )

    response = client.chat.completions.create(
        model=ctx.agent_name,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()
