import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MODEL_NAME = os.getenv(
    "FIREWORKS_MODEL_NAME",
    "accounts/fireworks/models/gemma2-9b-it",
).strip()

client = AsyncOpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY", "").strip() or "EMPTY",
    base_url=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
)


async def main() -> None:
    if not os.getenv("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is not set. Add it to your environment or .env file.")
        return

    print(f"Calling model: {MODEL_NAME}")
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        temperature=0.2,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())