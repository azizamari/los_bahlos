from textwrap3 import wrap
import random
import numpy as np

# def text_wrap_example(inputText):
#     for wrp in wrap(inputText, 150):
#         print (wrp)
#     print ("\n")
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## init program
# text_wrap_example(txt)
summary_tokenizer, summary_model = setup_t5_trans()
set_seed(42)


## summarizer service 

from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final


def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary


# summarized_text = summarizer(txt,summary_model,summary_tokenizer)


# print ("\noriginal Text >>")
# for wrp in wrap(txt, 150):
#   print (wrp)
# print ("\n")
# print ("Summarized Text >>")
# for wrp in wrap(summarized_text, 150):
#   print (wrp)
# print ("\n")


# Extract keywords and nouns

from nltk.corpus import stopwords
import string
import pke
import traceback

def get_nouns_multipartite(content):
    out=[]
    try:
        extract_creator = pke.unsupervised.MultipartiteRank()
        extract_creator.load_document(input=content,language='en')
        #  only leavy propn and nouns
        pos = {'PROPN','NOUN'}
        stopwords_list = list(string.punctuation)
        stopwords_list += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stopwords_list += stopwords.words('english')
        extract_creator.candidate_selection(pos=pos)
        extract_creator.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extract_creator.get_n_best(n=15)
        for val in keyphrases:out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


# process extracted keywords

from flashtext import KeywordProcessor

def return_keywords(initial_text,summarized_text, number_of_examples=3):
  keyword_list = get_nouns_multipartite(initial_text)
  print ("keywords unsummarized: ",keyword_list)
  keyword_processor = KeywordProcessor()
  for i in keyword_list:
    keyword_processor.add_keyword(i)

  found_keywords_in_text = keyword_processor.extract_keywords(summarized_text)
  found_keywords_in_text = list(set(found_keywords_in_text))
  print ("found_keywords_in_text in summarized: ",found_keywords_in_text)

  keywords_important =[]
  for keyword in keyword_list:
    if keyword in found_keywords_in_text:
      keywords_important.append(keyword)
  if number_of_examples >= len(keywords_important):
    number_of_examples=len(keywords_important)
  return keywords_important[:number_of_examples]

# examples=5
# keywords_important = return_keywords(txt,summarized_text,examples)
# print(keywords_important)


## q generator using T5
question_generator = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
token_2_question = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_generator = question_generator.to(device)

# sample question

def generate_questions(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=500, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, max_length=72)

  decoded_output=[]
  for ids in outs:
    decoded_output.append(tokenizer.decode(ids,skip_special_tokens=True) )

  question_4_keyword = decoded_output[0].replace("question:","")
  question_4_keyword= question_4_keyword.strip()
  return question_4_keyword


# for answer in keywords_important:
#   ques = generate_questions(summarized_text,answer,question_generator,token_2_question)
#   print (ques,'\n',answer.capitalize())

# we generated answer/ questions with a single term answer
# now we need to find similar yet not exact distraction words to make the question a multiple choice quizz

# setup sense2vec for word vectors
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('s2v_old')


# sentence transforming model

from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')


# define needed functions for finding similarity

from similarity.normalized_levenshtein import NormalizedLevenshtein
from sklearn.metrics.pairwise import cosine_similarity

normalized_levenshtein = NormalizedLevenshtein()

def filtering_of_same_sense_words(original,wordlist):
  filtered=[]
  base_sense =original.split('|')[1] 
  for i in wordlist:
    if i[0].split('|')[1] == base_sense:
      filtered.append(i[0].split('|')[0].replace("_", " ").title().strip())
  return filtered

def return_words_with_high_similarity(wordlist,wrd):
  similarity_score=[]
  for i in wordlist:
    similarity_score.append(normalized_levenshtein.similarity(i.lower(),wrd.lower()))
  return max(similarity_score)

def get_words_from_sense2vec(word,s2v,topn,question):
    result = []
    # print ("word ",word)
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      # print (most_similar)
      result = filtering_of_same_sense_words(sense,most_similar)
    except:
      result =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in result:
      if return_words_with_high_similarity(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)
    
    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):

    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    keywords_index = [np.argmax(word_doc_similarity)]
    candidates_index = [i for i in range(len(words)) if i != keywords_index[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_index, :]
        target_similarities = np.max(word_similarity[candidates_index][:, keywords_index], axis=1)

        # valeur MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_index[np.argmax(mmr)]

        keywords_index.append(mmr_idx)
        candidates_index.remove(mmr_idx)

    return [words[idx] for idx in keywords_index]


# distractor functions

from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

def get_distractors_wordnet(word):
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]
      
      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0: 
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  distractors = get_words_from_sense2vec(word,sense2vecmodel,top_n,origsentence)
  if len(distractors) ==0:
    return distractors
  distractors_new = [word.capitalize()]
  distractors_new.extend(distractors)

  embedding_sentence = origsentence+ " "+word.capitalize()
  keyword_embedding = sentencemodel.encode([embedding_sentence])
  distractor_embeddings = sentencemodel.encode(distractors_new)

  max_keywords = min(len(distractors_new),5)
  filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() !=word.lower():
      final.append(wrd.capitalize())
  final = final[1:]
  return final

# sent = "You can easily edit a symbol and propagate changes in real what"
# keyword = "Time"
import re
def clean_distractors(wordlist):
  good=[]
  for word in wordlist:
    if not re.compile("^[a-zA-Z ]*$") is None:
      good.append(word)
  return good[:2]
# words=get_distractors(keyword,sent,s2v,sentence_transformer_model,40,0.2)
# clean_distractors(words)
# print(words)

def generate_question(title,context,radiobutton):
  result={"skill":title,"questions":[]}
  summary_text = summarizer(context,summary_model,summary_tokenizer)
  keys =  np.unique(return_keywords(context,summary_text,5))
  for answer in keys:
    ques = generate_questions(summary_text,answer,question_generator,token_2_question)
    if radiobutton=="Wordnet":
      distractors = get_distractors_wordnet(answer)
    else:
      distractors = get_distractors(answer.capitalize(),ques,s2v,sentence_transformer_model,40,0.2)
      
    choices=clean_distractors(distractors)
    if(len(choices)==0):continue
    result["questions"].append({"text":ques,"answer":answer,"choices":choices})

  return result

# print(generate_question("titre exemple",txt,""))