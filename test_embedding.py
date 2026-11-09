from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")
vector = embed_model.get_text_embedding("מה מברכים על תפוח?")

print(len(vector))
print(vector[:5])