Story_Forest
==============================

The purpose of this project is to attempt to re-implement the system discussed in the article: "Growing Story Forest Online from Massive Breaking News". The system that was introduced in the paper is designed to find and organize events in story trees from a large amount of trending news and breaking news. This system was created to quickly process a large amount of breaking news data, and its news structure is likely to be captured by a tree, a timeline or a flat structure. In order to re-implement the project, we are following the main steps of the system presented in the paper while trying new approaches to see if we can obtain better performance. The authors of that paper tested 4 types of structures and concluded that the tree structure outperforms all other structures. Thus, we will implement the tree structure. Contrary to what has been implemented by the authors, our system is being implemented entirely in python rather than java like the original paper. 



Project Organization
------------

```main_run.ipynb``` is the notebook that contains all the steps: Text preprocessing; Keyword extraction; Topic detection; Story Tree Organization

```execution_preprocessing.py``` contains functions for preprocessing of texts.

```execution_tf_idf_cosine.py``` contains functions to create tfdif and cosine similarity.

```keyword_extractor.py``` contains functions for keyword extraction.

```class_event_story.py``` contains classes of Event and Story for story clustering.

```story_event_utils.py``` contains functions story clustering.

```/archive/``` contains some old notebook for development.

```/data/```is empty. The data is located in a google drive at https://drive.google.com/drive/folders/1Yk_MolBefkJ8iQF58x5S9Cbtb93NursN?usp=sharing.
