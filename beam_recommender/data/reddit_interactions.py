import pandas as pd


class RedditInteractions:

    @staticmethod
    def get_interactions():
        return pd.read_csv('./data/reddit_2016_10_2016_11_with_id.csv')
