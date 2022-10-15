
from textwrap3 import wrap
import random
import numpy as np

def text_wrap_example(inputText):
    for wrp in wrap(inputText, 150):
        print (wrp)
    print ("\n")
txt="""You can easily edit a Symbol and propagate changes in real time across all the instances. In order to edit a Symbol, you must double click the Symbol. (Similar to editing a group on canvas) Any position, size or appearance changes to the elements in a Symbol are propagated to all copies. There is no master copy of the Symbol. You can edit from any copy of the Symbol and preview those changes in real time across your document. While position and appearance changes are linked, you can have unique text and bitmap content in a Symbol. What this means is that, you can override the text and image in a Symbol while keeping it appearance linked to all the other copies. Edit the Symbol and change the text or drop in a new image to override the content. Having the ability to override Symbols is very helpful. However, what happens if need to update all the copies of the Symbol to have the same text and bitmap content? The update all commands helps you specifically do that. Right click on the Symbol and select Update All from the context menu to push the text and bitmap content from that Symbol to all the copies. If you select a specific part of a nested Symbol and hit Update All, it only updates that part of the nested Symbol across all copies. Simplicity and power with the update all command."""


import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

def setup_t5_trans():

    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_model = summary_model.to(device)
    return summary_tokenizer, summary_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## main program
text_wrap_example(txt)
summary_tokenizer, summary_model = setup_t5_trans()
set_seed(42)
