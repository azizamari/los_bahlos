from fastapi import FastAPI, UploadFile, File
import json
from fastapi.middleware.cors import CORSMiddleware
from generate_json  import *

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
async def get_questions_from_pdf(file: bytes = File(...)):
    dictData={}
    with open('result.json') as d:
        dictData = json.load(d)
    out_file=open('result.pdf','wb')
    out_file.write(bytes([(f) for f in file]))
    return dictData
    
@app.get("/quizzes")
def get_questions_from_pdf():
    dictData={}
    with open('result.json') as d:
        dictData = json.load(d)
    return dictData
    