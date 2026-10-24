"""Prints the Gemini model names your API key can use.

    python list_models.py
"""
import os

from google import genai

if "GEMINI_API_KEY" not in os.environ:
    raise SystemExit('Set your key first:  $env:GEMINI_API_KEY="your-key"')

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
names = []
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if not actions or "generateContent" in actions:  # only models that can answer questions
        names.append(m.name.replace("models/", ""))

for name in sorted(names):
    print(name)
print(f"\n{len(names)} models. Look for one with 'lite' in the name.")