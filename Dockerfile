FROM jupyter/minimal-notebook

RUN pip install pandas lightfm matplotlib praw pymongo python-dotenv google-api-python-client
RUN pip install scikit-learn scikit-optimize
