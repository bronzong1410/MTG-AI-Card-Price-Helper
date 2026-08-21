from doc_helper import extract_text
import streamlit as st
import ollama
import chromadb
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Chroma setup (persists to disk so you don't re-embed every rerun) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("mtg_prices")

def get_all_documents(batch_size=1000):
    all_docs = []
    all_metas = []
    offset = 0
    while True:
        batch = collection.get(limit=batch_size, offset=offset)
        docs = batch["documents"]
        if not docs:
            break
        all_docs.extend(docs)
        all_metas.extend(batch["metadatas"])
        offset += batch_size
    return all_docs, all_metas

def load_text(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text, max_chars=500):
    """Each line is its own chunk — already one card entry per line."""
    lines = text.split("\n")
    chunks = [line.strip() for line in lines if line.strip()]
    return chunks

def create_embeddings_batch(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def index_file(file_name, month_label):
    text = load_text(file_name)
    chunks = chunk_text(text)
    if not chunks:
        return

    embeddings = create_embeddings_batch(chunks)

    add_batch_size = 500
    for i in range(0, len(chunks), add_batch_size):
        batch_chunks = chunks[i:i + add_batch_size]
        batch_embeddings = embeddings[i:i + add_batch_size]
        batch_ids = [f"{month_label}_{i + j}" for j in range(len(batch_chunks))]
        batch_metas = [{"month": month_label}] * len(batch_chunks)
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_chunks,
            metadatas=batch_metas
        )


def get_relevant_chunks(question, n_results=15):
    q_embedding = create_embeddings_batch([question])[0]
    results = collection.query(query_embeddings=[q_embedding], n_results=n_results)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    semantic_chunks = [f"[{m['month']}] {d}" for d, m in zip(docs, metas)]

    exact_chunks = []
    # crude heuristic: try matching capitalized word sequences from the question
    # against card names via Chroma's document contains filter
    words = question.split()
    candidates = [w.strip(",.?!\"'") for w in words if w[0:1].isupper()]
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        matches = collection.get(where_document={"$contains": candidate}, limit=50)
        for doc, meta in zip(matches["documents"], matches["metadatas"]):
            exact_chunks.append(f"[{meta['month']}] {doc}")

    combined = list(dict.fromkeys(exact_chunks + semantic_chunks))
    return combined

# --- Files to index: use the preprocessed *_notes.txt files, NOT the raw
#     Scryfall dumps. Run preprocess_scryfall.py once first to generate these. ---
FILES_TO_INDEX = [
    ("august_notes.txt", "august"),
    ("july_notes.txt", "july"),
    ("june_notes.txt", "june"),
    ("may_notes.txt", "may"),
    ("april_notes.txt", "april"),
    ("march_notes.txt", "march"),
    ("february_notes.txt", "february"),
    ("january_notes.txt", "january"),
]

st.title("MTG Card Price Helper")

# Only index once — check if the collection is already populated.
if collection.count() == 0:
    progress = st.progress(0, text="Indexing price files...")
    for idx, (file_name, month_label) in enumerate(FILES_TO_INDEX):
        index_file(file_name, month_label)
        progress.progress((idx + 1) / len(FILES_TO_INDEX), text=f"Indexed {month_label}")
    st.success(f"Indexed {collection.count()} chunks.")




question = st.chat_input("Ask a question about mtg or card prices")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )
if question:
    relevant_chunks = get_relevant_chunks(question)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
    use only these notes to answer the question, and don't guess — if you can't find any data just say so
    use the pricing data on cards throughout this year to try and predict trends
    if somebody asks about a current price, then refrence august, because it is currently august 2026
    if you are asked to predict the price trends for a card, use different price points across all 7 months and try to predict the next month based on the data give a rough estimate number (example: $1.40)
    RELEVANT NOTES:
    {context}

    QUESTION:
    {question}

    if you cannot find the answer in the notes, say that you cannot find the answer
    do not answer any questions about anything except magic: the gathering
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 8192}
    )
    answer = response["message"]["content"]
    with st.chat_message(
        "assistant"
    ):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )
    st.divider()
    st.subheader("Sources:")
    st.write(context)
