from los_bahlos import *
from handling_pdf.pdf_operations import *
import json

title=get_title()
content=get_pages_content()
result=[]
for text in content:
    if len(text)>200:
        result.append(generate_question(title,text,''))

result={"Title":title, "Questions":result}

with open("result.json", "a") as outfile:
    json.dump(result, outfile)    