"""LLM utility functions with retry logic and error handling."""

from __future__ import annotations

import time
import logging
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMCallError(Exception):
    """Exception raised when LLM call fails after retries."""
    pass


def call_with_retry(
    func: Callable[..., T],
    max_retries: int = 3,
    retry_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None,
) -> T:
    """Call a function with exponential backoff retry logic.
    
    Args:
        func: Function to call
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiply delay by this factor after each retry
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called on each retry
        
    Returns:
        Result from the function call
        
    Raises:
        LLMCallError: If all retries fail
    """
    last_exception = None
    delay = retry_delay
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                if on_retry:
                    try:
                        on_retry(attempt, e)
                    except Exception as callback_error:
                        logger.warning(f"Retry callback failed: {callback_error}")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise LLMCallError(f"Failed after {max_retries + 1} attempts: {last_exception}")


def llm_retry_decorator(
    max_retries: int = 3,
    retry_delay: float = 2.0,
    backoff_factor: float = 2.0,
    fallback_return: Any = None,
):
    """Decorator to add retry logic to LLM calls.
    
    Args:
        max_retries: Maximum retry attempts
        retry_delay: Initial delay between retries
        backoff_factor: Exponential backoff multiplier
        fallback_return: Value to return if all retries fail (instead of raising)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return call_with_retry(
                    lambda: func(*args, **kwargs),
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    backoff_factor=backoff_factor,
                )
            except LLMCallError as e:
                if fallback_return is not None:
                    logger.warning(
                        f"LLM call failed, using fallback. Error: {e}"
                    )
                    return fallback_return
                raise
        return wrapper
    return decorator


def safe_llm_generate(
    backend: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    max_retries: int = 3,
    fallback_response: str = "",
) -> str:
    """Safely generate text from LLM with retry logic.
    
    Args:
        backend: LLM backend object with generate method
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        fallback_response: Response to return if all retries fail
        
    Returns:
        Generated text or fallback response
    """
    def generate():
        return backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    
    try:
        return call_with_retry(
            generate,
            max_retries=max_retries,
            retry_delay=2.0,
            backoff_factor=2.0,
        )
    except LLMCallError:
        logger.warning(f"LLM generation failed after retries, using fallback")
        return fallback_response


def parse_json_with_fallback(
    text: str,
    fallback: Optional[Any] = None,
) -> Any:
    """Parse JSON from text with fallback on failure.
    
    Args:
        text: Text containing JSON
        fallback: Value to return if parsing fails
        
    Returns:
        Parsed JSON or fallback value
    """
    import json
    import re
    
    # Try to find JSON in text
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return fallback
