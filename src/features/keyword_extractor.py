import numpy as np
import re
import nltk
import itertools
import yake
import gensim

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from gensim.summarization import keywords as keywords_gensim
import editdistance as Levenshtein
import spacy

#nlp = spacy.load("en_core_web_lg")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def add_ner(text):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    ner=[]
    for ent in doc.ents:
        if(ent.label_ == "PER" or  ent.label_=="ORG" or ent.label_=="GPE"):
            ner.append(ent.text)
    return ner
def bert_max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates=10):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def bert_mmr(doc_embedding, word_embeddings, words, top_n, diversity=0.7):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def add_bert(text, top_n, max_sum_sim=False,mmr=False,diversity=0.7,nr_candidates=10):
    n_gram_range = (1, 1)
    stop_words = "english"

    # Extract candidate words/phrasestrain_set["question"][0]
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    candidates = count.get_feature_names()

    model = SentenceTransformer("/work/bert-model")
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)

    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    keywords_max_sum_sim=[]
    keywords_mmr=[]

    if (max_sum_sim):
        keywords_max_sum_sim=bert_max_sum_sim(doc_embedding,candidate_embeddings,candidates,top_n,nr_candidates)

    if (mmr):
        keywords_mmr=bert_mmr(doc_embedding,candidate_embeddings,candidates,top_n,diversity)

    return keywords,keywords_max_sum_sim,keywords_mmr


def add_gensim(text, number_keyword):
    return keywords_gensim(text, words=number_keyword, lemmatize = True).split('\n')


def add_yake(text, number_keyword,language = "en",max_ngram_size = 1,deduplication_thresold = 0.9,deduplication_algo = 'seqm',windowSize = 1):
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=number_keyword, features=None)
    keywords = kw_extractor.extract_keywords(text)
    return keywords

def keyword_extractor(text, number_keyword=10):
    # for each we have a list of top keyword 
    yake_extract   = add_yake(text, number_keyword)
    yake_keyword   = [i[0] for i in yake_extract] 
    gensim_keyword = add_gensim(text, number_keyword)
    bert_keyword   = add_bert(text, number_keyword)[0]
    all_ner        = add_ner(text)

    # TODO Prioritise the keyword that are in the different models
    keywords  = yake_keyword + gensim_keyword + bert_keyword + all_ner

    # put all element in lower case
    keywords = [element.lower() for element in keywords]

    #remove all NER duplicate
    keywords = list(dict.fromkeys(keywords))
    keyword_list = keywords

    # TRYING to remove similar words
    for i,word1 in enumerate(keywords):
        for word2 in keywords[i+1:len(keywords)+1]:
            distance = Levenshtein.eval(word1, word2)
            if distance <= 2:
                if len(word1)<len(word2) and word1 in keyword_list:
                    keyword_list.remove(word1)
                elif len(word2)<len(word1) and word2 in keyword_list:
                    keyword_list.remove(word2)

    keyword_list_copy = keyword_list

    # I know all these others loops are stupid, but it works, TODO: MAKE BETTER
    # Remove word that are contain in other words 
    for i,word1 in enumerate(keywords):
        for word2 in keywords[i+1:len(keywords)+1]:
                # if word1 is subset
                if word1 in word2 and word1 in keyword_list_copy:
                    keyword_list_copy.remove(word1)
                # if word2 is subset
                elif word2 in word1 and word2 in keyword_list_copy:
                    keyword_list_copy.remove(word2)
                
    for word in keyword_list_copy:
        if any(i.isdigit() for i in word):
            keyword_list_copy.remove(word)

    return  keyword_list_copy


def keyword_extraction_baseline(text, number_keyword=40, language = "en", max_ngram_size = 1, deduplication_thresold = 0.9, deduplication_algo = 'seqm', windowSize = 1):
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=number_keyword, features=None)
    keywords = kw_extractor.extract_keywords(text)
    keywords_extractor = [i[0] for i in keywords] 

    return keywords_extractor


def keyword_extraction_baseline_gensim(text, number_keyword=40):
    return keywords_gensim(text, words=number_keyword, lemmatize = True).split('\n')


def precision(keyword,target):
    count = 0 
    for element in keyword:
        if element in target:
            count+=1
        else:
            for inside in target:
                if element in inside:
                    count+=1 
                elif inside in element:
                    count+=1

    return count/len(keyword)


def recall(keyword,target):
    count = 0 
    for element in keyword:
        if element in target:
            count+=1
        else:
            for inside in target:
                if element in inside:
                    count+=1 
                elif inside in element:
                    count+=1

    return count/len(target)


def f_measure(keyword,target):
    pre = precision(keyword,target)
    re =recall(keyword,target)
    return (2*pre*re)/(pre+re),pre,re

"""
def displayGraph(textGraph):

    graph = nx.Graph()
    for edge in textGraph.edges():
        graph.add_node(edge[0])
        graph.add_node(edge[1])
        graph.add_weighted_edges_from([(edge[0], edge[1], textGraph.edge_weight(edge))])

        textGraph.edge_weight(edge)
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='seagreen', alpha=0.9,
            labels={node: node for node in graph.nodes()})
    plt.axis('off')
    plt.show()
"""
