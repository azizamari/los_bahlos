import PyPDF2

file_name="course.pdf"
doc=PyPDF2.PdfFileReader(file_name)

# convert page to text
text=""
for i in range(0,3):
    text+=doc.pages[i].extract_text()+'\n'
text= text.replace('\n',' ')

# extract skill title
def get_title():
    pos=text.find('[Skill]:')
    pos2=text.find('Learning Style:')
    return text[pos+9:pos2].strip()

# extract pages content
def get_pages_content():
    content=[]
    pos=text.find("Page:  1 ")
    i=1
    while True:
        if text.find(f"Page:  {i+1} ")!=-1:
            pos2=text.find(f"Page:  {i+1} ")
            content.append(text[pos+9:pos2-1])
            pos=pos2
            i+=1
        else:
            pos2=text.find(f"Assessments:")
            content.append(text[pos+9:pos2-1])
            break
    return content
def get_pages_content_merged():
    return " ".join(get_pages_content())
    
l=get_pages_content()
print(l[4])