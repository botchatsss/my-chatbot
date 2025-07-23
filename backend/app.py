from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings())

llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("question", "")
    if not query:
        return {"answer": "Please provide a question."}
    answer = qa.run(query)
    return {"answer": answer}
