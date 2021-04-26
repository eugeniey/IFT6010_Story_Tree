import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from scipy import spatial

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords


'''
do_cosine_similarity(dataset_tf_idf.iloc[40]['VECTOR'],dataset_tf_idf.iloc[0]['VECTOR'])

output 
{'chat': 0.010449574311118917, 'us': 0.018608473383810775, 'facebook': 0.010402222166680766, 'messenger': 0.010789461602160046, 'find': 0.011035393649675106, 'happening': 0.009717578335407207, 'world': 0.01649194096565685, 'unfolds': 0.010844213680244323, 'lubricates': 0, 'cushions': 0, 'joints': 0, 'help': 0, 'keep': 0, 'exercising': 0}
{'chat': 0, 'us': 0, 'facebook': 0, 'messenger': 0, 'find': 0, 'happening': 0, 'world': 0, 'unfolds': 0, 'lubricates': 0.001043660041972249, 'cushions': 0.0010170320598814641, 'joints': 0.0010702493432690977, 'help': 0.008318070060455958, 'keep': 0.00410092040383734, 'exercising': 0.0010905227309952364}

do_cosine_similarity(dataset_tf_idf.iloc[0]['VECTOR'],dataset_tf_idf.iloc[2]['VECTOR'])
return --> 0.21925583584401553

'''

def do_cosine_similarity(vector_a,vector_b):
    #make sure that vector a is the longest vector
    if len(vector_a)<len(vector_b):
        temp = vector_a
        vector_a = vector_b
        vector_b = temp

    list_1={}
    list_2={}

    for elem in vector_a:
        if elem in vector_b:
            list_1[elem]= vector_a[elem]
            list_2[elem]= vector_b[elem]
        else:
            list_1[elem]= vector_a[elem]
            list_2[elem]= 0

    for elem in vector_b:
        if not elem in list_1:
            list_1[elem]= 0 
            list_2[elem]= vector_b[elem]

    # print(list_1)
    # print(list_2)

    # print(len(list_1))
    # print(len(list_2))

    # print(list(list_1.values()))
    # print(list(list_2.values()))

    return(1 - spatial.distance.cosine(list(list_1.values()), list(list_2.values())))



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
    # vector=[dict_tf_idf[token] for token in tokens if token in dict_tf_idf]
    dict_vector={}   
    for token in tokens:
        if token in dict_tf_idf:
            dict_vector[token]=dict_tf_idf[token]
    
    return dict_vector

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


# to lower case
# remove digit
# remove ponctuation 
# remove stop word
# lemma
# remove single characters and empty token
def preprocessing(text):
    text=text.lower()

    text = ''.join([i for i in text if not i.isdigit()])

    tokens = nltk.word_tokenize(text)

    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        tokens = np.char.replace(tokens, i, '')

    stop_words = stopwords.words('english')

    lemmatizer = WordNetLemmatizer()

    new_tokens = [ lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    lemma_tokens = [token for token in tokens]

    return_tokens=[token for token in new_tokens if len(token)>1]
    
    return return_tokens



def latest_tfidf(doc, allDocs):
    """
    doc: list of string. Represents text of an article
    allDocs: list of list of string
    """
    
    dict_tf ={}

    for term in doc:
        term_in_document = doc.count(term) 
        len_of_document = float(len(doc)) 
        normalized_tf = term_in_document / len_of_document 

        num_docs_with_given_term = 0
        # print(num_docs_with_given_term)
        for docs in allDocs:
            if term in docs:
                num_docs_with_given_term += 1

        if num_docs_with_given_term > 10:
        # Total number of documents
            total_num_docs = len(allDocs) 

            idf_val = np.log(float(total_num_docs) / num_docs_with_given_term)
            dict_tf[term]=idf_val*normalized_tf
        else:
            dict_tf[term]= 0 

    return dict_tf



'''
example of execution 

news_dataset = pd.read_pickle("/work/IFT6010_Story_Tree/data/short_news_dataset_2_with_extractedkeyword.pickle")

a,b = get_corpus_tf_idf(news_dataset,'TEXT',ngram=1)

'''