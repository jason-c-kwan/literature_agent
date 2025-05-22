# sanity_check_gemini.py
import os
import asyncio
from dotenv import load_dotenv          # pip install python-dotenv
from openai import AsyncOpenAI

# 1) pull variables out of .env into os.environ
load_dotenv()

# 2) grab what we need, with defaults
API_KEY   = os.getenv("OPENAI_API_KEY")          # must exist
BASE_URL  = os.getenv("OPENAI_API_BASE",
                      "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL     = os.getenv("GEMINI_MODEL",
                      "gemini-2.5-flash-preview-04-17")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set (check your .env file)")

# 3) create the async client and send a test chat completion
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def main() -> None:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Ping?"}],
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
