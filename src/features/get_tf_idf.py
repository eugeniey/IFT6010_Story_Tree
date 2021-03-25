import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


# to lower case
# remove digit
# remove ponctuation 
# remove stop word
# remove single characters and empty token
def preprocessing(text):
    text=text.lower()

    text = ''.join([i for i in text if not i.isdigit()])

    tokens = nltk.word_tokenize(text)

    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        tokens = np.char.replace(tokens, i, '')


    stop_words = stopwords.words('english')

    new_tokens=[token for token in tokens if token not in stop_words]

    return_tokens=[token for token in new_tokens if len(token)>1]

    return_text =' '.join(map(str, return_tokens))
    
    return return_text


def document_vector(text,dict_tf_idf):
    tokens = nltk.word_tokenize(text)
    vector=[dict_tf_idf[token] for token in tokens if token in dict_tf_idf]
    
    return vector

def get_corpus_tf_idf(dataframe, colomn,ngram=1):
    dataframe['PREPROCESSING_TEXT']=dataframe[colomn].apply(preprocessing)
    corpus=dataframe['PREPROCESSING_TEXT']
    corpus=corpus.to_list()

    count_vec = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1,ngram))
    fit_corpus = count_vec.fit_transform(corpus)

    do_transformation = TfidfTransformer()
    transformed_weights = do_transformation.fit_transform(fit_corpus)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    dict_tf_idf={}
    for k,v in zip(count_vec.get_feature_names(),weights):
        dict_tf_idf[k]=v
    
    dataframe['VECTOR']=dataframe['PREPROCESSING_TEXT'].apply(document_vector,dict_tf_idf=dict_tf_idf)

    return dataframe,dict_tf_idf

'''
example of execution 

news_dataset = pd.read_pickle("/work/IFT6010_Story_Tree/data/short_news_dataset_2_with_extractedkeyword.pickle")

a,b = get_corpus_tf_idf(news_dataset,'TEXT',ngram=1)

'''