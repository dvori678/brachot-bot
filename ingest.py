"""
טעינת קובץ ההלכות ל-ChromaDB.
הרצה: python ingest.py
"""
from dotenv import load_dotenv
import chromadb
import time

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import MetadataMode   


load_dotenv()

# ---------- הגדרות ----------
DATA_DIR = "data"              # התיקייה שבה נמצא hilchot-brachot.md
CHROMA_DIR = "chroma_db"       # איפה Chroma שומרת את הווקטורים
COLLECTION_NAME = "hilchot_brachot"
EMBED_MODEL = "gemini-embedding-2"

DRY_RUN = False   # True = רק חלוקה והדפסה, בלי לשלוח כלום ל-Gemini


def load_documents():
    return SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".md"],
    ).load_data()



def split_to_nodes(documents):
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    nodes = [n for n in nodes if n.text.startswith("### סעיף")]

    for n in nodes:
        n.excluded_embed_metadata_keys.append("file_path")
        n.excluded_llm_metadata_keys.append("file_path")

    return nodes


def inspect(nodes):
    print(f"נוצרו {len(nodes)} סעיפים")

    sample = nodes[10]
    print("--- טקסט ---")
    print(sample.text)
    print("--- מטא-דאטה ---")
    print(sample.metadata)
    print("--- מה שנשלח ל-embedding ---")
    print(sample.get_content(metadata_mode=MetadataMode.EMBED))


def build_index(nodes):
    # 1. חיבור ל-ChromaDB (שומר לתיקייה במחשב)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 2. מחיקת collection קודם, אם קיים, כדי שלא יהיו כפילויות
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # 3. collection חדש עם מדד cosine
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # 4. StorageContext שמחבר את ה-index ל-Chroma
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    # 5. יצירת הווקטורים בקבוצות, בגלל מגבלה של 100 בקשות לדקה
    embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL)
    index = VectorStoreIndex(
        [],
        storage_context=storage_context,
        embed_model=embed_model,
    )

    BATCH_SIZE = 90
    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i + BATCH_SIZE]
        index.insert_nodes(batch)
        print(f"נשמרו {i + len(batch)}/{len(nodes)} סעיפים")

        if i + BATCH_SIZE < len(nodes):
            print("מחכה דקה בגלל מגבלת הבקשות...")
            time.sleep(61)
            
    return index


def main():
    documents = load_documents()
    nodes = split_to_nodes(documents)
    inspect(nodes)

    if DRY_RUN:
        print("DRY_RUN: לא נשלח כלום ל-Gemini")
        return

    build_index(nodes)
    print("הטעינה הסתיימה")


if __name__ == "__main__":
    main()