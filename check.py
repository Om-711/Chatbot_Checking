from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware
import os
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from pydantic import BaseModel
import pandas as pd
from pymongo import MongoClient
import numpy as np
from functools import lru_cache


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # filled below from env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read API keys and external URIs from environment variables
# Expected env vars:
# - GOOGLE_API_KEY: Google Generative AI API key
# - MONGODB_URI: MongoDB connection string
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not GOOGLE_API_KEY:
    # Allow running with empty key only if LLM is not invoked yet; will raise at first use
    pass

# Configure CORS origins dynamically
try:
    if ALLOWED_ORIGINS.strip() == "*":
        # Allow all origins
        app.user_middleware = [m for m in app.user_middleware if m.cls is not CORSMiddleware]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173/"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
        app.user_middleware = [m for m in app.user_middleware if m.cls is not CORSMiddleware]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
except Exception:
    # If anything goes wrong, default to permissive CORS to avoid blocking
    app.user_middleware = [m for m in app.user_middleware if m.cls is not CORSMiddleware]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

from pymongo import MongoClient
import pandas as pd

def making_data():
        
    mongo_url = MONGODB_URI or "mongodb+srv://localhost/"  # fallback for local dev; should set MONGODB_URI in prod

    client = MongoClient(mongo_url)

    db = client["ECommerce"]

    product_collection = db["products"]
    user_data_collection = db["users"]

    products = list(product_collection.find())
    users  = list(user_data_collection.find())

    product_data = []
    for p in products:
        if(p.get("isActive")):
            product_data.append({
                "productID": str(p["_id"]),
                "name": p["name"],
                "price": p["price"],
                "category": p["category"],
                "description": p.get("description", ""),
                "images": p.get("images", "Not Found"),
                "stock" : p.get("stock", "0"),
                "rating" : p.get("rating", "0"),
                "reviews" : p.get("reviews", "0"),
                "createdAt": p.get("createdAt", ""),
                "updatedAt": p.get("updatedAt", ""),
                "isActive": p.get("isActive", True)
            })

    user_data = []
    order_data = []
    for u in users:
        for history in u.get("history", []):
            user_data.append({
                "user_id": str(u["_id"]),
                "productID": str(history.get("productId", "")),
                "event": history.get("event", {}).get("type","Not Found"),
                "Timestamp": history.get("time", ""),
                "duration":history.get("duration", 0)/1000 # Convert milli-second to second betwa
            })
        # print(u)
        for order in u.get("orders", []):
            order_data.append({
                'user_id': str(u["_id"]),
                "orderID": (order)
            })
        
    # print(order_data)
    df_products = pd.DataFrame(product_data)
    # print(df_products.head())

    df_user = pd.DataFrame(user_data)
    # print(df_user.head())

    df_orders = pd.DataFrame(order_data)
    # print(df_orders.head())

    return df_products, df_user, df_orders

# Load and preprocess data once at startup so that afterwards it can used directly and take less time
products, users, orders = making_data()
products = products[['name', 'category', 'price', 'description', 'productID']]

combined_text = products.to_string() + "\n" + users.to_string()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([combined_text])

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = None
if vector_store is None:
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings)
    except:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)




@lru_cache(maxsize=100)
def cached_search(query):
    return vector_store.similarity_search(query, k=5)


async def chat_ai_async(user_id: str, question: str):
    if not question:
        return {"message": "No query found for user.", "options": ["Back"]}

    try:
        # if user ask same question again then use cached result to answer it
        docs = cached_search(question)
        context = "\n".join([d.page_content for d in docs])

        prompt = PromptTemplate(
            template="""
            You are a helpful chatbot for an e-commerce website. 
            Use ONLY the information found in the provided context. Answer concisely in 1–2 lines.

            If the context does not contain enough information to answer, reply exactly: "No data found".

            Rules:
            1. If the user requests a product recommendation, recommend up to 3 products that best match the user's preferences and needs, using only product fields present in the context along with the product data and link to the product.
            Product link should be in this format:
            https://apnabzaar.netlify.app/productdetail/product_id
            Replace `product_id` with the product's `id` value from the Product Data.
            Example:
            Context:
            Product: Product Name: Product Description: Product Price: Product Link:
            Question: What are some good products for me?
            Answer: Based on your preferences, we recommend the following products:
            Product 1: Product Name: Product Description: Product Price: Product Link:
            Product 2: Product Name: Product Description: Product Price: Product Link:
            Product 3: Product Name: Product Description: Product Price: Product Link:
            2. If the user asks for a product that is not available in the context, reply exactly:
            "We are sorry, the product you requested is currently not available on our site. However, we value your interest and would be happy to assist you with similar products or alternatives that meet your needs. Please let us know what you're looking for, and we'll do our best to help you find a suitable option."
            3. If the user asks for order details, return only order information present in the order data (e.g., order id, items, status, delivery ETA). Do NOT invent or assume missing fields.
            4. If the user asks for a product's price, reply with the price, Name of product and the product link in this exact format:
            https://apnabzaar.netlify.app/productdetail/product_id
            Replace `product_id` with the product's `id` value from the Product Data.
            5. Do not provide any information that is not present in the context. Do not add technical notes, disclaimers, or extra sentences—keep it to 1–2 lines.
            "

            Context:
            {context}

            Question: {question}

            Product Data : {products}
            Order Data : {orders}
            """,
            input_variables=["context", "question", "products", "orders"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        result = await chain.ainvoke({"context": context, "question": question, "products": products.to_string(), "orders": orders.to_string()})

        return {"message": result["text"] if isinstance(result, dict) and "text" in result else str(result)}

    except Exception as e:
       
        print("Error in chat_ai_async:", repr(e))
        return {"message": f"Internal error: {str(e)}"}



@app.get("/chat")
async def chat(user_id: str, option: str):
    
    if option == "main":
        return JSONResponse({
            "message": f" Hello Betwa! Welcome to ApnaBazzar! How may I help you today?",
            "options": ["Order Related", "Product Related", "Others"]
        })
    elif option == "Order Related":
        return JSONResponse({"message": "Please choose an option related to your orders:",
                             "options": ["Recent Order", "All Orders", "Track Order", "Back"]})
    elif option == "Product Related":
        return JSONResponse({"message": "Need help with products? Select an option below:",
                             "options": ["Request Product", "Back"]})
    elif option == "Others":
        return JSONResponse({"message": "You can chat with our AI assistant for general help 💬",
                             "options": ["Chat with AI Assistant", "Back"]})

    elif option == "Recent Order":
        user_id_orders = orders[orders['user_id'] == user_id]
        return user_id_orders[-1:]

    elif option == "All Orders":
        user_id_orders = orders[orders['user_id'] == user_id]
        return user_id_orders[-5:]

    elif option == "Track Order":
        user_id_orders = orders[orders['user_id'] == user_id]
        return user_id_orders[-1:]

    elif option == "Request Product":
        return JSONResponse({"message": "Send us the product name you want to request (not available on site).",
                             "options": ["Back"]})

    elif option == "Chat with AI Assistant":
        return JSONResponse({"message": "You’re now connected to the AI Assistant. Please type your question below:",
                             "options": ["Back"]})

    elif option == "Back":
        return JSONResponse(await chat(user_id, "main"))

    return JSONResponse({"message": "Invalid option. Try again.", "options": ["Back"]})



@app.get("/chat/ai")
async def chat_ai_endpoint(user_id: str, question: str):
    resp = await chat_ai_async(user_id, question)
    return JSONResponse(resp)



