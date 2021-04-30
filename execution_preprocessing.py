#Created by
#Henri-cedric Mputu - henri-cedric.mputu.boleilanga@umontreal.ca
#Eugenie Yockell - eugenie.yockell@umontreal.ca
#bassirou Ndao  - bassirou.ndao@umontreal.ca



# all the instalation need
'''
!python -m spacy download en_core_web_lg
!pip install sentence-transformers
!pip install gensim
!pip install networks
!pip install matplotlib
!pip install yake
!pip install editdistance==0.3.1
!pip install nltk
'''

import sys
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords


from keyword_extractor import keyword_extractor


# to lower case
# remove digit
# remove ponctuation 
# remove stop word
# lemma
# remove single characters and empty token
def preprocessing(text, lemm_or_stem = "stem"):
    text=text.lower()

    text = ''.join([i for i in text if not i.isdigit()])

    tokens = nltk.word_tokenize(text)

    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        tokens = np.char.replace(tokens, i, '')
        
    stop_words = stopwords.words('english')

    # stemming
    if lemm_or_stem == "stem":
        snow_stemmer = SnowballStemmer(language='english')
        new_tokens = [ snow_stemmer.stem(token) for token in tokens if token not in stop_words or token != "said"]
        
    # lemmatising
    else:
        lemmatizer = WordNetLemmatizer()
        new_tokens = [ lemmatizer.lemmatize(token) for token in tokens if token not in stop_words or token != "said"]

    lemma_tokens = [token for token in tokens]

    return_tokens=[token for token in new_tokens if len(token)>1]
    
    return return_tokens


def do_preprocessing_for_news_articles(news_dataframe, destination_file):

    # We merged the text and the title of each document ; we called the preprocessing function from utils;
    #then we extracted all the keywords

    del news_dataframe['id']
    del news_dataframe['url']

    news_dataframe['title'] = news_dataframe['title'].astype(str)
    news_dataframe['content'] = news_dataframe['content'].astype(str)

    news_dataframe['text_and_title'] = news_dataframe[['title','content']].apply(lambda x :" ".join(x), axis=1)
    news_dataframe['preprocessed_text'] = news_dataframe['text_and_title'].apply(preprocessing).apply(lambda x :" ".join(x))

    news_dataframe["extracted_keywords"] = news_dataframe["preprocessed_text"].apply(keyword_extractor)

    news_dataframe.to_pickle(destination_file+"/news_with_extracted_keywords_2.pkl") 


def do_preprocessing_for_dsk_articles(dsk_dataframe, destination_file):
    #We removed all duplicate
    # We merged the text and the title of each document ; we called the preprocessing function from utils;
    #then we extracted all the keywords

    dsk_dataframe[dsk_dataframe.duplicated(subset=['text'])]
    dsk_dataframe.drop_duplicates(subset=['text']) #.drop_duplicates(subset=['TITLE'])

    dsk_dataframe['title'] = dsk_dataframe['title'].astype(str)
    dsk_dataframe['text'] = dsk_dataframe['text'].astype(str)

    dsk_dataframe['text_and_title'] = dsk_dataframe[['title','text']].apply(lambda x :" ".join(x), axis=1)
    dsk_dataframe['preprocessed_text'] = dsk_dataframe['text_and_title'].apply(preprocessing).apply(lambda x :" ".join(x))
    dsk_dataframe["extracted_keywords"] = dsk_dataframe["preprocessed_text"].apply(keyword_extractor)

    dsk_dataframe.to_pickle(destination_file+"/news_dsk_with_extracted_keywords.pkl")


