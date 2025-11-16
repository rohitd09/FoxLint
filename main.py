import asyncio

# Set Windows Proactor event loop for subprocess support (only on Windows)
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import sys
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from scraper import ScrapeData
from VectorStores.chroma_collection_manager import ChromaCollectionManager

app = FastAPI(title="FoxLint", version="0.1")


@app.get("/")
def read_root():
    return {"Description": "Welcome to FoxLint! Your all in one place to know which policy is safe for you to agree with and which can expose your data!"}


@app.get("/scrape/")
async def scrape_data(url: str, website_name: str):
    try:
        scraper = ScrapeData(url, website_name)
        await scraper.run()
        return {"Description": f"Scraped data stored in {website_name}.txt in ./scraped_policy."}
    except Exception as e:
        return {"Description": f"Oops! Something went wrong: {str(e)}."}


@app.get("/store_embeddings")
def store_embeddings(file_name: str):
    chroma = ChromaCollectionManager()
    chroma.add_chunks_to_chroma(file_name)
    return {"Description": f"{file_name}.txt broken down to chunks and stored in ChromaDB with collection name {file_name}"}

@app.get("/get_similar_chunks")
def get_similar_chunks(file_name: str, query: str):
    chroma = ChromaCollectionManager()
    result = chroma.get_similar_chunks(file_name, query)
    # If error returned, pass-through
    if isinstance(result, dict) and "error" in result:
        return result

    # ChromaDB returns a dict with keys like "documents", "ids", "embeddings"
    out = {}
    for i in range(len(result.get("documents", []))):
        out[i] = {
            "document": result["documents"][i],
            "id": result["ids"][i],
            "embedding": result["embeddings"][i]
        }
    return out