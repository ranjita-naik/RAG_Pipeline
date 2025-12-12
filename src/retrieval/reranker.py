from typing import List
from langchain_openai import ChatOpenAI

class Reranker:
    def __init__(self, model="gpt-4o-mini", top_k=4):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.top_k = top_k

    def rerank(self, query: str, docs: List):
        """
        Uses the LLM to score documents by relevance to query.
        """

        scored = []
        for d in docs:
            prompt = f"""
Score relevance between 0 and 1.
Query: {query}
Document:
{d.page_content[:500]}
            """
            score = float(self.llm.invoke(prompt).content.strip())
            scored.append((score, d))

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        return [d for _, d in scored_sorted[:self.top_k]]

