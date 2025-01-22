from fastapi import FastAPI
from starlette.responses import FileResponse 
from pydantic import BaseModel
from aimodule import predict_sentiment

app = FastAPI()

@app.get("/")
async def read_root():
    return FileResponse('index.html')

class Text(BaseModel):
    text: str


@app.post("/api/result")
async def get_result(request: Text):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Analyze the sentiment of the input text
    print(request.text)
    result = predict_sentiment(request.text)

    # Return the result
    return {"result": result}

@app.get('/background.jpg')
async def returnbg():
    return FileResponse("background.jpg")
