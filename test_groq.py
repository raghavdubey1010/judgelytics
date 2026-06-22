import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

async def test():
    try:
        from groq import AsyncGroq
        key = os.getenv("GROQ_API_KEY", "")
        print(f"Using key: {key[:20]}...")
        client = AsyncGroq(api_key=key)
        completion = await client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=30
        )
        print("SUCCESS:", completion.choices[0].message.content)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

asyncio.run(test())
