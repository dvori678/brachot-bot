import re
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

llm = GoogleGenAI(model="gemini-3.5-flash-lite", temperature=0)

EXPAND_PROMPT = """את עוזרת חיפוש בספר קיצור שולחן ערוך, הלכות ברכות.
נסחי מחדש את השאלה בלשון הספר, והוסיפי מונחים הלכתיים שמתאימים לה.
כללים:
- אל תעני על השאלה.
- אל תוסיפי נושאים שלא נשאלו. למשל, אם שואלים על ברכה ראשונה, אל תוסיפי ברכה אחרונה.
- אם השאלה לא קשורה להלכות ברכות, החזירי אותה בדיוק כמו שהיא.
- החזירי שורה אחת בלבד.

שאלה: {question}
שאלה מנוסחת מחדש:"""


def expand(question):
    prompt = EXPAND_PROMPT.format(question=question)
    response = llm.complete(prompt)
    text = re.sub(r"[\u0600-\u06FF]", "", response.text)  # מחיקת אותיות ערביות
    return text.strip()

QUESTIONS = [
    "מה מברכים על עוגה?",
    "שכחתי לברך אחרי שאכלתי, מה עושים?",
    "מתי מברכים הגומל?",

]



if __name__ == "__main__":
    for q in QUESTIONS:
        print("מקורית:", q)
        print("מורחבת:", expand(q))
        print()