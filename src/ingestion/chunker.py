from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    return splitter.split_documents(documents)

