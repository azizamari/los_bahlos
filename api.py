from fastapi import FastAPI
app=FastAPI()


@app.get("/paragraph")
def get_questions():
    return {"You can easily edit a symbol and propagate changes in real what?":{"answer":"Time","wrong":["Break","Zone/region","uwu"]}}