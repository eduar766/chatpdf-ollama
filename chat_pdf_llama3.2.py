import os
import tempfile
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import message

def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {"provider": "ollama", "config": {"model": "llama3.2:latest", "max_tokens": 250, "temperature": 0.5, "stream": True, "base_url": 'http://localhost:11434'}},
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {"provider": "ollama", "config": {"model": "llama3.2:latest", "base_url": 'http://localhost:11434'}},
        }
    )

def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.title("Chat with PDF using Llama 3.2")
st.caption("This app allows you to chat with a PDF using Llama 3.2 running locally with Ollama!")

db_path = tempfile.mkdtemp()

# Asegurar que el estado inicial está correctamente configurado
if 'app' not in st.session_state:
    st.session_state.app = embedchain_bot(db_path)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'clearing_chat' not in st.session_state:
    st.session_state.clearing_chat = False
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""  # Inicializamos el prompt vacío

with st.sidebar:
    st.header("PDF upload")
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)

if st.button("Add to Knowledge Base"):
    with st.spinner("Adding PDF to knowledge base ..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            st.session_state.app.add(f.name, data_type="pdf_file")
        os.remove(f.name)
    st.success(f"Added {pdf_file.name} to knowledge base!")

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

# Verificamos si estamos en el proceso de limpiar el historial
if not st.session_state.clearing_chat:
    prompt = st.chat_input("Ask a question about the PDF")
    
    # Solo procesar el nuevo prompt si hay input del usuario
    if prompt:
        st.session_state.prompt = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        message(prompt, is_user=True)

        with st.spinner("Thinking..."):
            response = st.session_state.app.chat(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response)
        
        # Limpiar el prompt después de procesarlo
        st.session_state.prompt = ""

# Acción para limpiar el historial del chat
if st.button("Clear Chat History"):
    st.session_state.clearing_chat = True
    st.session_state.messages = []
    st.rerun()

# Resetear la variable clearing_chat después del reinicio
st.session_state.clearing_chat = False
