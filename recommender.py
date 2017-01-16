from lightfm import LightFM
import helpers as h
import numpy as np
import pandas as pd
import praw

# TODO: move to .env
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""


class Recommender:

    def __init__(self):
        self.__r = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                               client_secret=REDDIT_CLIENT_SECRET,
                               user_agent='beam-discover')
        self.__load_data()

    def __load_data(self, from_cache=True):
        self._beam_interactions_df = pd.read_pickle('./data/beam_interactions.pk')
        self._beam_interactions_sum_df = self._beam_interactions_df.groupby(['user', 'subreddit_id']).size()\
            .to_frame(name = 'count').reset_index()

        self._ratings_df = self._beam_interactions_df
        self._ratings_sum_df = self._beam_interactions_sum_df

        self._ratings, self._uid_to_idx, self._idx_to_uid,\
            self._sid_to_idx, self._idx_to_sid = h.df_to_matrix(self._ratings_df, 'user', 'subreddit_id')

    def train(self, model_args={}, fit_args={}, train_test_split=False):
        default_model_args = dict(loss='warp', learning_schedule='adagrad', user_alpha=1e-06, item_alpha=1e-06)
        default_fit_args = dict(epochs=50, num_threads=4)

        if train_test_split:
            train, test, user_index = h.train_test_split(
                self._ratings, 5, fraction=0.2)
        else:
            train = self._ratings

        self._model = LightFM(**{**default_model_args, **model_args})
        self._model.fit(
            **{**default_fit_args, **dict(interactions=train), **fit_args})
            self.recommend

    def recommend(self, user_ids=[], n=20):
        """Generate recommendations.

        Parameters
        ----------
        `user_ids`: `List[str]`
            List with usernames of users to generate recommendations for.
            If left empty, all users in provided interactions DataFrame are assumed.

        `n`: `int`
            Number of recommendations to generate per user.

        Returns
        -------
        recommendations: `List[List[str]]`
            List containing lists with recommendations for users.
        """

        # Default to all users
        if len(user_ids) == 0:
            user_ids = np.arange(self._ratings.shape[0])
        else:
            if isinstance(user_ids, str):
                user_ids = [user_ids]

            user_ids = [self._uid_to_idx[u] for u in user_ids]

        # Default to all items
        item_ids = np.arange(self._ratings.shape[1])

        recommendations = []

        for user_id in user_ids:
            username = self._idx_to_uid[user_id]
            scores = self._model.predict(user_id, item_ids)
            raw_recs = np.array(list(self._idx_to_sid.values()))[
                np.argsort(-scores)]
            raw_recs = list(self.filter_known_subreddits(username, raw_recs))

            # Filter recommendations from nsfw subreddits, until `n` is reached
            recs = []
            i = 0
            while (len(recs) < n) or i == 3:
                recs.extend(self.filter_nsfw(raw_recs[:n]))

            recommendations.append(recs)

        return recommendations

    def filter_known_subreddits(self, username, items, threshold=1):
        subs_interacted_with = self._ratings_sum_df[self._ratings_sum_df['user'] == username]\
            .query('count > {}'.format(threshold))['subreddit_id'].tolist()

        return (x for x in items if x not in (subs_interacted_with + h.DEFAULT_SUBS_IDS))

    def filter_nsfw(self, subreddit_ids):
        subs_info = self.__r.info(subreddit_ids)
        return [id for (id, info) in zip(subreddit_ids, subs_info) if info.over18 == False]

    def evaluate(self, measures=[]):
        print('TODO')
