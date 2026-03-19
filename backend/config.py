import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DATA_FILE = os.path.join(os.path.dirname(__file__), "data/android_weekly_classified.json")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "stepfun/step-3.5-flash:free"
TOP_K = 8

SKIP_DOMAIN_TYPES = {"medium", "medium_pub"}
