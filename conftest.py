import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Тесты герметичны: не зависят от пользовательского .env
# (пользователь мог включить sentence_transformers/реальный LLM API).
from app.config import settings  # noqa: E402

settings.embeddings_provider = "hash"
settings.llm_provider = "mock"
settings.bot_token = ""

