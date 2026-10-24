"""
One place for the LLM call, so you can switch providers without touching the rest of the code.

Choose the provider by setting a key in the terminal:
    $env:GEMINI_API_KEY="..."     -> uses Gemini
    $env:GROQ_API_KEY="..."       -> uses Groq
If both are set, pick one with:  $env:LLM_PROVIDER="gemini"  (or "groq")
Choose the model with:           $env:LLM_MODEL="exact-model-name"
"""
import os
import time
try:  # read GEMINI_API_KEY / LLM_MODEL from a .env file in the project folder, if there is one
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Defaults are a best guess. If you get "model not found", copy the exact
# model name from Google AI Studio / console.groq.com into LLM_MODEL.
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash-lite", "groq": "llama-3.3-70b-versatile"}
TRANSIENT = ("502", "503", "504", "429", "UNAVAILABLE", "DEADLINE_EXCEEDED", "imed out", "imeout")


def provider():
    choice = os.environ.get("LLM_PROVIDER", "").lower()
    if choice in DEFAULT_MODELS:
        return choice
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return None


def model_name():
    return os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider(), "")


def key_help():
    return ("No API key found. Set GEMINI_API_KEY (Gemini) or GROQ_API_KEY (Groq) "
            "in the terminal, then restart.")


def _call(prov, system, user, timeout):
    if prov == "gemini":
        from google import genai
        from google.genai import types
        key = os.environ["GEMINI_API_KEY"]
        try:
            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        except Exception:
            client = genai.Client(api_key=key)
        r = client.models.generate_content(
            model=model_name(),
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return r.text or ""
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=timeout, max_retries=0)
    r = client.chat.completions.create(
        model=model_name(),
        temperature=0.1,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content or ""


def generate(system, user, timeout=40.0, attempts=3):
    """Ask the model. Retries when the service is busy, rate-limited or slow."""
    prov = provider()
    if prov is None:
        raise RuntimeError(key_help())
    for attempt in range(attempts):
        try:
            return _call(prov, system, user, timeout)
        except Exception as e:
            if attempt == attempts - 1 or not any(code in str(e) for code in TRANSIENT):
                raise
            time.sleep(3 * (attempt + 1))