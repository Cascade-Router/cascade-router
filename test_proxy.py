import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key or not api_key.strip() or "your_openai_api_key" in api_key:
    print(
        "Error: OPENAI_API_KEY is missing or still set to the placeholder value.\n"
        "Add your key to a `.env` file in the project root:\n\n"
        "  OPENAI_API_KEY=sk-your-real-key-here\n\n"
        "Copy `.env.example` to `.env` and replace the placeholder. "
        "Never commit `.env` to git."
    )
    sys.exit(1)

# Point the official OpenAI SDK at our local C++ proxy.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=api_key,
)

print("Sending request through Cascade...")
try:
    response = client.chat.completions.create(
        model="cascade-auto",  # The proxy intercepts this
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a one-line python function to reverse a string."},
        ],
    )
    print("\n--- Upstream Response ---")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"\nError: {e}")
