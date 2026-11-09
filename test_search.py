"""
השוואת החיפוש: לפני ואחרי הרחבת השאלה.
הרצה: python test_search.py
"""
from dotenv import load_dotenv
load_dotenv()

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from expand import expand

# --- חיבור למאגר ---
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("hilchot_brachot")  # השם חייב להיות כמו ב-ingest.py
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

retriever = index.as_retriever(similarity_top_k=4)
RELEVANCE_THRESHOLD = 0.70

def search(query):
    nodes = retriever.retrieve(query)
    results = []
    for n in nodes:
        label = n.metadata.get("header_path", "") + " | " + n.text[:40]
        results.append((label, n.score))
    return results


def print_results(title, results):
    print(f"  {title}:")
    for label, score in results:
        print(f"    {score:.3f}  {label}")


QUESTIONS = [
    "מה מברכים על עוגה?",
    # "שכחתי לברך אחרי שאכלתי, מה עושים?",
    # "מתי מברכים הגומל?",
    # "איך מתקינים פייתון?",
    #     "מה הבירה של צרפת?",
]

if __name__ == "__main__":
    for q in QUESTIONS:
        before = search(q)
        top_score = before[0][1]

        print("=" * 60)
        print("שאלה:", q)
        print_results("לפני", before)

        if top_score < RELEVANCE_THRESHOLD:
            print(f"  ציון עליון {top_score:.3f} מתחת לסף. לא נמצא בחומר.")
            continue

        expanded = expand(q)
        after = search(q + " " + expanded)
        print("מורחבת:", expanded)
        print_results("אחרי", after)

        before_set = {label for label, score in before}
        after_set = {label for label, score in after}
        print("  נוספו:", after_set - before_set or "אין")
        print("  נעלמו:", before_set - after_set or "אין")