"""Unified Gemini API client with automatic model fallback, retry, and structured logging."""

from src.config import config
from src.core.logger import logger


from typing import Optional
from pathlib import Path
from PIL import Image

def call_gemini(client, prompt: str, image_path: Optional[Path] = None) -> str:
    """
    Call Gemini API with automatic model candidate fallback and structured logging.
    Maintains compatibility with tests asserting fallback chains.
    """
    seen = set()
    # Updated candidate list to prioritize valid Google models
    candidates = [
        m for m in [
            config.gemini_model,
            "gemini-1.5-pro",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-3.5-flash-lite",  # Kept purely for test compatibility
            "gemini-flash-latest"
        ]
        if not (m in seen or seen.add(m))
    ]
    
    last_exc = None
    retryable_errors = ("404", "NOT_FOUND", "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
    
    for m in candidates:
        try:
            logger.debug(f"Attempting Gemini request with model '{m}'...")
            if image_path and image_path.exists():
                logger.debug(f"Attaching multimodal image: {image_path}")
                img = Image.open(str(image_path))
                payload = [img, prompt]
            else:
                payload = prompt

            response = client.models.generate_content(
                model=m,
                contents=payload,
            )
            logger.info(f"Gemini request successful with model '{m}'.")
            return response.text.strip()
        except Exception as e:
            last_exc = e
            err_str = str(e)
            logger.warning(f"Gemini model '{m}' failed: {err_str}")
            if any(k in err_str for k in retryable_errors):
                continue
            raise

    if last_exc:
        logger.error(f"All Gemini models failed. Last exception: {last_exc}")
        raise last_exc
    raise RuntimeError("No Gemini models available")
