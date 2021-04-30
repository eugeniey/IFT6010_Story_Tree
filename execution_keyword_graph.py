
#Created by
#Henri-cedric Mputu - henri-cedric.mputu.boleilanga@umontreal.ca
#Eugenie Yockell - eugenie.yockell@umontreal.ca
#bassirou Ndao  - bassirou.ndao@umontreal.ca


import networkx as nx
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from icecream import ic
from tqdm import tqdm
import pickle

from cdlib import algorithms, viz


wordnet = WordNetLemmatizer()
porter = PorterStemmer()

lemmatizer_lem = wordnet.lemmatize
lemmatizer = porter.stem

# all the instalation need
'''
!pip install textgraphics
!pip install networkx
!pip install python-louvain
!pip install cdlib
!pip install communities
!pip install icecream
'''


def increment_edge (graph, node0, node1):
    # ic(node0, node1)
    
    if graph.has_edge(node0, node1):
        graph[node0][node1]["weight"] += 1.0
        # ic(node0, node1, graph[node0][node1]["weight"])
    else:
        graph.add_edge(node0, node1, weight=1.0)

def link_sentence (doc, lemma_graph, seen_lemma, count_lemma):
    visited_tokens = []
    visited_nodes = []
    visited_lemma = []

    for idx, token in enumerate(doc):

        lemma_key = lemmatizer(token)
        create_node = False
        if lemma_key not in seen_lemma:
            seen_lemma[lemma_key] = set([idx])
            create_node = True
        else:
            seen_lemma[lemma_key].add(idx)

        node_id = list(seen_lemma.keys()).index(lemma_key)

        if create_node:
            count_lemma[lemma_key] = 1
        else:
            count_lemma[lemma_key]+=1

        if not node_id in lemma_graph:
            lemma_graph.add_node(node_id)

        # ic(visited_tokens, visited_nodes)
        # ic(list(range(len(visited_tokens) - 1, -1, -1)))
        
        for prev_token in range(len(visited_tokens) - 1, -1, -1):
            # ic(prev_token, (idx - visited_tokens[prev_token]))
            
            if (idx - visited_tokens[prev_token]) <= 3:
                # ic(node_id)
                increment_edge(lemma_graph, lemma_key, visited_lemma[prev_token])
                # increment_edge(lemma_graph, node_id, visited_nodes[prev_token])
            else:
                break

        # ic(idx, token, lemma_key, len(visited_tokens), len(visited_nodes))

        visited_tokens.append(idx)
        visited_nodes.append(node_id)
        visited_lemma.append(lemma_key)



def create_keyword_graph(dataframe,destination_file, minimum_count= 4):

    keyword_nxGraph_new = nx.Graph()
    seen_lemma = {}
    count_lemma = {}
    keyword_list = list(dataframe["extracted_keywords"])

    print(keyword_list[2])
    for doc in tqdm(keyword_list):
        link_sentence(doc, keyword_nxGraph_new, seen_lemma, count_lemma)

    filtered_node = [k for k, v in count_lemma.items() if v >= minimum_count]
    print(filtered_node)

    keyword_nxGraph_filtered = keyword_nxGraph_new.subgraph(filtered_node)
    print(len(keyword_nxGraph_filtered))

    keyword_nxGraph = keyword_nxGraph_filtered
    print(len(keyword_nxGraph))

    keyword_communities = algorithms.louvain(keyword_nxGraph, weight='weight', resolution=1., randomize=False)
    keyword_pos = nx.spring_layout(keyword_nxGraph)

    # print('\n\n VIZ Community graph \n\n')

    viz.plot_community_graph(keyword_nxGraph, keyword_communities, plot_labels= True)

    # print('\n\n VIZ network clusters \n\n')

    viz.plot_network_clusters(keyword_nxGraph, keyword_communities, keyword_pos, figsize=(10, 10), cmap="Paired", plot_labels= True)

    pickle.dump(keyword_communities.to_node_community_map(), open(destination_file+"/extracted_communities.pickle",'wb'))

    
