# Los Bahlos - GoMyCode Hackathon
### generate quizzes from pdf course
## Steps to get the project working:
### create virtual enviornment:
    python -m venv env
### install requirments.txt:
    pip install -r requirements.txt
### install fast api:
    pip install fastapi
### install needed Sense2Vec:
    !wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
    !tar -xvf  s2v_reddit_2015_md.tar.gz
### install node packages in frontend folder:
    cd frontend
    npm install
### install angular CLI:
    npm install -g @angular/cli
### start api:
    uvicorn api:app
### launch angular project:
    cd frontend
    ng serve
* los_bahlos.py  --- contains the NLP pipeline
* api.py --- contains fastapi implementation
* handling_pdf/pdf_operations.py pdf service
* generate_json.py uses both los_bahlos.py and pdf_operations.py to generate json result
## System Design Diagram:
<img src="https://github.com/azizamari/los_bahlos/blob/ac26dc9dad09ce16a1b31f7c1efff59172dfb792/sys-design.png">
