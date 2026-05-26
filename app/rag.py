from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import os
#
#
load_dotenv()
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)
idx = pc.Index(
    os.getenv(
        "PINECONE_INDEX"
    )
)
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

mdl = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

TOP_K = 5
MODEL = "llama-3.3-70b-versatile"


def create_query_embedding(question):

    embedding = mdl.encode(
        [question]
    )[0]

    return embedding.tolist()


def retrieve_documents(vector):

    result = idx.query(
        vector=vector,
        top_k=TOP_K,
        include_metadata=True
    )

    return result.get(
        "matches",
        []
    )


def build_context(matches):

    contexts = []

    for match in matches:

        metadata = match.get(
            "metadata",
            {}
        )

        text = metadata.get(
            "text",
            ""
        )

        if text:
            contexts.append(text)

    return "\n\n".join(
        contexts
    )


def build_prompt(question, context):

    return f"""
Answer only from the given context.

Format for chat:
- Short lines
- Blank lines
- Bullet points
- No huge paragraphs

If answer missing say exactly:
I could not find that in the retrieved documents.

Context:
{context}

Question:
{question}
"""


def ask_question(question):

    vector = create_query_embedding(
        question
    )

    matches = retrieve_documents(
        vector
    )

    if not matches:
        return "No relevant context found"

    context = build_context(
        matches
    )

    prompt = build_prompt(
        question,
        context
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content