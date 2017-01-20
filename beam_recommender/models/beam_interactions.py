from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


class BeamInteractions:

    def get_raw_interactions(self):
        # TODO: get data from BigQuery
        return pd.read_pickle('./data/beam_interactions_df.pk')

    def get_interactions_sum(self):
        # TODO: get data from BigQuery
        # return pd.read_pickle('./data/beam_interactions_sum.pk')

        raw_interactions_df = self.get_raw_interactions()

        # Group interactions per user per subreddit and calculate the count of
        # interactions
        interactions_sum_df = raw_interactions_df.groupby(['user', 'subreddit_id'])\
            .size().to_frame(name='count').reset_index()

        # Get timestamp from last interaction the user has had with a subreddit
        last_interaction = raw_interactions_df.groupby(['user', 'subreddit_id'])['timestamp']\
            .max().to_frame(name='last_interaction').reset_index()['last_interaction']

        interactions_sum_df['last_interaction'] = last_interaction

        return interactions_sum_df

    def get_interactions_with_weights(self):
        interactions_df = self.get_interactions_sum()

        now = datetime.now()
        since_last_month = datetime.timestamp(
            now + relativedelta(months=-1)) * 1000000

        interactions_df['weight'] = (
            interactions_df['last_interaction']
            .apply(lambda x: 2 if x > since_last_month else 1))

        return interactions_df
