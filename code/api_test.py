import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ API key not found!")
else:
    print("✅ API key loaded")

try:
    client = genai.Client(api_key=api_key)
    
    # Use gemini-2.5-flash (current free tier model)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one sentence."
    )
    print("✅ Gemini API working!")
    print("Response:", response.text)

except Exception as e:
    print("❌ API call failed:", e)