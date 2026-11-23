from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Basic FastAPI App")

# Data model for request/response validation
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

# In-memory storage
items_db: dict[int, Item] = {}

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to FastAPI"}

@app.post("/items/{item_id}", response_model=Item)
def create_item(item_id: int, item: Item):
    if item_id in items_db:
        raise HTTPException(status_code=400, detail="Item ID already exists")
    items_db[item_id] = item
    return item

@app.get("/items/{item_id}", response_model=Item)
def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]