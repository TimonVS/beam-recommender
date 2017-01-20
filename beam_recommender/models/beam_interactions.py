import pandas as pd


class BeamInteractions:

    def get_raw_interactions_df(self):
        # TODO: get data from BigQuery
        return pd.read_pickle('./data/beam_interactions_df.pk')

    def get_interactions_sum_df(self):
        # TODO: get data from BigQuery
        # return pd.read_pickle('./data/beam_interactions_sum.pk')

        raw_interactions_df = self.get_raw_interactions_df()

        # Group interactions per user per subreddit and calculate the count of interactions
        interactions_sum_df = raw_interactions_df.groupby(['user', 'subreddit_id'])\
            .size().to_frame(name='count').reset_index()

        # Get timestamp from last interaction the user has had with a subreddit
        last_interaction = raw_interactions_df.groupby(['user', 'subreddit_id'])['timestamp']\
            .max().to_frame(name='last_interaction').reset_index()['last_interaction']

        interactions_sum_df['last_interaction'] = last_interaction

        return interactions_sum_df
