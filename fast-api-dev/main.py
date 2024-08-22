from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/add_secret/{namespace}/{secret}")
def add_secret(namespace, secret):
    
    return {"message": "Hello World"}