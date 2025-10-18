from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from functools import lru_cache
from pymongo import MongoClient
import pandas as pd
import os
import asyncio

# --- CONFIG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://localhost/")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

app = FastAPI(title="ApnaBazzar Chatbot")

# --- CORS ---
origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA LOADING ---
def making_data():
    client = MongoClient(MONGODB_URI)
    db = client["ECommerce"]
    products = list(db["products"].find())
    users = list(db["users"].find())

    product_data = [
        {
            "productID": str(p["_id"]),
            "name": p["name"],
            "price": p["price"],
            "category": p["category"],
            "description": p.get("description", ""),
        }
        for p in products if p.get("isActive")
    ]

    user_data = []
    order_data = []

    for u in users:
        for h in u.get("history", []):
            user_data.append({
                "user_id": str(u["_id"]),
                "productID": str(h.get("productId", "")),
                "event": h.get("event", {}).get("type", "Not Found"),
                "Timestamp": h.get("time", ""),
                "duration": h.get("duration", 0)/1000
            })
        for o in u.get("orders", []):
            order_data.append({'user_id': str(u["_id"]), "orderID": o})

    return pd.DataFrame(product_data), pd.DataFrame(user_data), pd.DataFrame(order_data)

products, users, orders = making_data()
products = products[['name', 'category', 'price', 'description', 'productID']]

# --- VECTOR STORE & EMBEDDINGS ---
combined_text = products.to_string() + "\n" + users.to_string()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([combined_text])

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vector_store = FAISS.load_local("faiss_index", embeddings)
except:
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")

# --- LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# --- CACHE SEARCH ---
@lru_cache(maxsize=100)
def cached_search(query):
    return vector_store.similarity_search(query, k=5)

# --- CHAT LOGIC ---
async def chat_ai_async(user_id: str, question: str):
    if not question:
        return {"message": "No query found for user."}
    try:
        docs = cached_search(question)
        context = "\n".join([d.page_content for d in docs])

        prompt = PromptTemplate(
            template="""You are a helpful chatbot for an e-commerce website.
            Use ONLY the context below. If info not found, reply exactly: "No data found".
            Context:
            {context}
            Question: {question}
            Product Data : {products}
            Order Data : {orders}
            """,
            input_variables=["context", "question", "products", "orders"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = await chain.ainvoke({
            "context": context,
            "question": question,
            "products": products.to_string(),
            "orders": orders.to_string()
        })
        return {"message": result.get("text", str(result))}
    except Exception as e:
        return {"message": f"Internal error: {str(e)}"}

# --- API ENDPOINTS ---
@app.get("/chat")
async def chat(user_id: str, option: str):
    return JSONResponse({"message": f"Received option: {option}", "options": ["Back"]})

@app.get("/chat/ai")
async def chat_ai_endpoint(user_id: str, question: str):
    return JSONResponse(await chat_ai_async(user_id, question))

# --- ENTRYPOINT FOR RENDER ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
