"""
הלב של הבוט: חיפוש, הוספת הסעיף הקודם, ויצירת תשובה.
הרצה לבדיקה: python bot.py
"""
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from dotenv import load_dotenv
load_dotenv()
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from expand import expand,llm

CHROMA_DIR = "chroma_db"
STORAGE_DIR = "storage"
COLLECTION_NAME = "hilchot_brachot"  # אותו שם כמו ב-ingest.py
TOP_K = 6
RELEVANCE_THRESHOLD = 0.70

NOT_FOUND = "לא מצאתי תשובה לשאלה זו בחומר. כדאי לשאול רב."

ANSWER_PROMPT = """את עוזרת שעונה על שאלות בהלכות ברכות, רק לפי הקטעים מקיצור שולחן ערוך שלמטה.
כללים:
- עני רק לפי הקטעים. אל תוסיפי ידע מבחוץ.
- בסוף התשובה כתבי את המקורות: סימן וסעיף.
- אם התשובה לא נמצאת בקטעים, כתבי בדיוק: "{not_found}"
- עני בעברית, בקצרה ובבהירות.
- אם בקטעים יש כמה מקרים שונים, התחילי מהמקרה שמתאים לשאלה כפי שנשאלה, בלי להניח פרטים שהשואל לא אמר (למשל, אם לא הזכיר סעודה, אל תניחי שהוא בתוך סעודה). אחר כך פרטי בקצרה את המקרים האחרים.
- כתבי את כל המקורות שהשתמשת בהם, לא רק אחד.

קטעים:
{context}

שאלה: {question}
תשובה:"""

def docstore_from_chroma(collection):
    data = collection.get(include=["metadatas", "documents"])
    nodes = [
        metadata_dict_to_node(metadata, text=text)
        for metadata, text in zip(data["metadatas"], data["documents"])
    ]
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    return docstore
# --- טעינה ---
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=TOP_K)

docstore = docstore_from_chroma(collection)
def add_previous(nodes):
    result = {}
    for n in nodes:
        prev_info = n.node.prev_node
        if prev_info and docstore.document_exists(prev_info.node_id):
            prev = docstore.get_node(prev_info.node_id)
            result.setdefault(prev.node_id, NodeWithScore(node=prev, score=None))
        result[n.node.node_id] = n
    return list(result.values())


def retrieve(question):
    before = retriever.retrieve(question)
    if not before or before[0].score < RELEVANCE_THRESHOLD:
        return None

    expanded = expand(question)
    after = retriever.retrieve(question + " " + expanded)

    # איחוד בלי כפילויות, שומרים את הציון הגבוה מבין השניים
    merged = {}
    for n in before + after:
        node_id = n.node.node_id
        if node_id not in merged or n.score > merged[node_id].score:
            merged[node_id] = n
    nodes = sorted(merged.values(), key=lambda n: n.score, reverse=True)

    return add_previous(nodes)

def build_context(nodes):
    parts = []
    for n in nodes:
        header = n.node.metadata.get("header_path", "")
        parts.append(f"[{header}]\n{n.node.get_content()}")
    return "\n\n---\n\n".join(parts)


def answer(question):
    nodes = retrieve(question)
    if nodes is None:
        return NOT_FOUND, []

    prompt = ANSWER_PROMPT.format(
        not_found=NOT_FOUND,
        context=build_context(nodes),
        question=question,
    )
    response = llm.complete(prompt)
    return response.text.strip(), nodes


if __name__ == "__main__":
    for q in ["מה מברכים על עוגה?", "מותר לבשל בשבת?"]:
        text, nodes = answer(q)
        print("=" * 60)
        print("שאלה:", q)
        print("תשובה:", text)
        print("סעיפים שנשלחו למודל:", len(nodes))