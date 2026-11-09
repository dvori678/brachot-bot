"""
ממשק הבוט.
הרצה: streamlit run app.py
"""
import streamlit as st

from bot import answer

st.set_page_config(page_title="בוט הלכות ברכות", page_icon="📖")

# כיוון מימין לשמאל
st.markdown(
    """
    <style>
    .stApp { direction: rtl; text-align: right; }
    .stChatInput textarea { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("בוט הלכות ברכות")
st.caption("עונה לפי קיצור שולחן ערוך, הלכות ברכות. אינו תחליף לשאלת רב.")

# היסטוריית השיחה נשמרת בין הרצות של הדף
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("שאלה בהלכות ברכות...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפשת בחומר..."):
            try:
                text, nodes = answer(question)
            except Exception as e:
                text, nodes = f"אירעה שגיאה: {e}", []

        st.markdown(text)

        if nodes:
            with st.expander("הסעיפים שנמצאו"):
                for n in nodes:
                    header = n.node.metadata.get("header_path", "")
                    tag = f"{n.score:.3f}" if n.score is not None else "סעיף קודם"
                    st.markdown(f"**{header}** ({tag})")
                    st.text(n.node.get_content())

    st.session_state.messages.append({"role": "assistant", "content": text})