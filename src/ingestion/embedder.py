from langchain_openai import OpenAIEmbeddings
from src.config import settings

def get_embedder():
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

