Story_Forest
==============================

The purpose of this project is to attempt to re-implement the system discussed in the article: "Growing Story Forest Online from Massive Breaking News". The system that was introduced in the paper is designed to find and organize events in story trees from a large amount of trending news and breaking news. This system was created to quickly process a large amount of breaking news data, and its news structure is likely to be captured by a tree, a timeline or a flat structure. In order to re-implement the project, we are following the main steps of the system presented in the paper while trying new approaches to see if we can obtain better performance. The authors of that paper tested 4 types of structures and concluded that the tree structure outperforms all other structures. Thus, we will implement the tree structure. Contrary to what has been implemented by the authors, our system is being implemented entirely in python rather than java like the original paper. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
