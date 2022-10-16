from fastapi import FastAPI, UploadFile, File
import json

app=FastAPI()

@app.post("/quizzes")
async def get_questions_from_pdf(file: UploadFile= File(...)):
    dictData={}
    with open('result.json') as d:
        dictData = json.load(d)
    return dictData
    