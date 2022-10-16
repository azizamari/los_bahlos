from fastapi import FastAPI, UploadFile, File
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/quizzes")
async def get_questions_from_pdf(file: UploadFile= File(...)):
    dictData={}
    with open('result.json') as d:
        dictData = json.load(d)
    return dictData
    