import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    DATA_FOLDER: str = "data"
    VECTOR_DB_PATH: str = "vectorstore"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-large"

settings = Settings()
