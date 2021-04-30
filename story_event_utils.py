import pandas as pd
import pickle
import numpy as np
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')


def jaccard_similarity(keyword_list_1, keyword_list_2):
    list1_as_set = set(keyword_list_1)

    intersection = list1_as_set.intersection(keyword_list_2)

    set_union = set(keyword_list_1 + keyword_list_2)

    return len(intersection) / len(set_union)


def count_similar_word_in_title(title1, title2):

    title1.replace(' - The New York Times','')
    title2.replace(' - The New York Times','')

    title1_low = title1.lower()
    title2_low = title2.lower()

    title1_list = title1_low.split(" ")
    title2_list = title2_low.split(" ")

    stop_words = stopwords.words('english')

    title1_tokens = [ token for token in title1_list if token not in stop_words]
    title2_tokens = [ token for token in title2_list if token not in stop_words]

    return len(list(set(title1_tokens)&set(title2_tokens)))


# Needs to be true with at least 1 event in the story
# event_keyword: keywords of the event
# event_title: title of the event
# story: Story object
def is_event_in_story(event_keyword, event_title, story, threshold = 0.20):
    similarity = jaccard_similarity(event_keyword, story.get_list_of_keywords())

    one_event_common_title = False

    for event_of_story in story.get_list_of_events():

        common_words_title = count_similar_word_in_title(event_title, event_of_story.get_title())

        if common_words_title >= 1 and not common_words_title>4:
            one_event_common_title = True
            break

    if  similarity > threshold and one_event_common_title:
        return similarity, True
    
    return similarity, False


def compatiblity(tf_new_event, tf_event_story):

    #tf_new_event = np.array(tf_new_event)
    #tf_event_story = np.array(tf_event_story)

    if len(tf_new_event) < len(tf_event_story):
        temp = tf_new_event
        vector_a = tf_event_story
        vector_b = temp
    else:
        vector_a = tf_new_event
        vector_b = tf_event_story

    list_1={}
    list_2={}

    for elem in vector_a:
        if elem in vector_b:
            list_1[elem] = vector_a[elem]
            list_2[elem] = vector_b[elem]
        else:
            list_1[elem] = vector_a[elem]
            list_2[elem] = 0

    for elem in vector_b:
        if not elem in list_1:
            list_1[elem] = 0
            list_2[elem] = vector_b[elem]

    # turn dictionary to numpy array
    list_1_vector = np.fromiter(list_1.values(), dtype=float)
    list_2_vector = np.fromiter(list_2.values(), dtype=float)

    prod = np.dot(list_1_vector, list_2_vector)


    return prod / np.linalg.norm(list_1_vector) * np.linalg.norm(list_2_vector)


def conherence():
    event1 = news_dataset['VECTOR'].iloc[2]

    sum_ = 0

    for event in stories:
        event2 = news_dataset['VECTOR'].iloc[3]

        sum_ += compatiblity(event1, event2)

    return sum_/len(stories)


def time_penalty(delta, time1, time2):
    if time1 < time2:
        return math.exp(delta)

    return 0
