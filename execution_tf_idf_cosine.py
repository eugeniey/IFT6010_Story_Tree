#Created by
#Henri-cedric Mputu - henri-cedric.mputu.boleilanga@umontreal.ca
#Eugenie Yockell - eugenie.yockell@umontreal.ca
#bassirou Ndao  - bassirou.ndao@umontreal.ca


import numpy as np


from nltk.stem import WordNetLemmatizer


import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords


def tf_inverseDocumentFrequency(doc, allDocs):
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
    total= 0
    for a,b in zip(list_1,list_2):
        c= list_1[a]*list_2[b]
        total+=c


    cosine = total / float( (sum(list(list_1.values()))*sum(list(list_2.values())))**0.5 )

    return cosine



# for the cluster, we assume they are in a dataframe
def prepare_cluster_vector(dataframe, dict_weights):
    all_cluster = dataframe['cluster'].unique()

    all_cluster_dict = {}

    for cluster in all_cluster:
        cluster_dict ={}
        all_keyword=list(dataframe[dataframe["cluster"] == cluster]['keyword'])
        all_keyword=' '.join(all_keyword)

        # We will have more keywords than 
        tokens = nltk.word_tokenize(all_keyword)
        
        #!!!!DEAL with keyword with multiple word ... some keyword are not int the dictionary of all weights
        # for token in tokens:
        #     if token in dict_weights:
        #         cluster_dict[token]=dict_weights[token]
        #     else:
        #         cluster_dict[token]=0

        for token in tokens:
            cluster_dict[token]=1
           

        # print(cluster_dict)
        all_cluster_dict[cluster]=cluster_dict
        
    return all_cluster_dict


def set_document_topic(document_vector, all_cluster_vector):
    max_cluster_name=''
    max_cluster_value=0
    for cluster in all_cluster_vector:
        cluster_value= do_cosine_similarity(document_vector,all_cluster_vector[cluster])
        # print(cluster_value)

        if (max_cluster_value<cluster_value):
            max_cluster_value=cluster_value
            max_cluster_name=cluster
    
    # print(max_cluster_name)
    # print(max_cluster_value)
    return max_cluster_name