import os
from dotenv import load_dotenv
from huggingface_hub import login

def get_huggingface_token():
    """Get Hugging Face token from environment variables."""
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
    return token

def login_to_huggingface():
    """Login to Hugging Face using token from environment variables."""
    token = get_huggingface_token()
    login(token=token) 