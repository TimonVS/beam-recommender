from datetime import datetime

import numpy as np
import praw
import scipy.sparse as sp
from dateutil.relativedelta import relativedelta

from lightfm import LightFM
from lightfm.evaluation import (auc_score, precision_at_k)

from . import helpers as h
from . import config
from .default_subreddits import DEFAULT_SUBREDDITS_IDS
from .models.beam_interactions import BeamInteractions


def get_weights(df, row_name, col_name):
    rid_to_idx, idx_to_rid,\
        cid_to_idx, idx_to_cid = h.get_df_matrix_mappings(df,
                                                          row_name,
                                                          col_name)

    def map_ids(row, mapper):
        return mapper[row]

    i = df[row_name].apply(map_ids, args=[rid_to_idx]).as_matrix()
    j = df[col_name].apply(map_ids, args=[cid_to_idx]).as_matrix()
    v = df['weight'].as_matrix()
    weights = sp.coo_matrix((v, (i, j)), dtype=np.float64)

    return weights


class Recommender:

    DEFAULT_MODEL_ARGS = dict(loss='warp', learning_schedule='adagrad',
                              user_alpha=1e-06, item_alpha=1e-06, no_components=50)
    DEFAULT_FIT_ARGS = dict(epochs=100, num_threads=4)

    def __init__(self):
        self.__r = praw.Reddit(client_id=config.REDDIT_CLIENT_ID,
                               client_secret=config.REDDIT_CLIENT_SECRET,
                               user_agent=config.REDDIT_USER_AGENT)
        self.__load_data()

    def __load_data(self, from_cache=True):
        beam_interactions = BeamInteractions()
        self._beam_interactions_sum_df = beam_interactions.get_interactions_sum_df()

        now = datetime.now()
        since_last_week = datetime.timestamp(
            now + relativedelta(weeks=-1)) * 1000000

        self._beam_interactions_sum_df['weight'] = self._beam_interactions_sum_df['last_interaction']\
            .apply(lambda x: 2 if x > since_last_week else 1)

        self._ratings_df = self._beam_interactions_sum_df
        self._weights = get_weights(
            self._beam_interactions_sum_df, 'user', 'subreddit_id')

        self._ratings, self._uid_to_idx, self._idx_to_uid,\
            self._sid_to_idx, self._idx_to_sid = h.df_to_matrix(
                self._ratings_df, 'user', 'subreddit_id', 'count')

    def train(self, interactions=None, weights=None, model_args={}, fit_args={}):
        """Instantiate and train LightFM model.

        Parameters
        ----------
        `interactions`: `scipy csr matrix`
            Interactions used to train the model. Defaults to all interactions.

        `model_args`: `dict`
            Arguments used to instantiate `LightFM`.

        `fit_args`: `dict`
            Arguments used to fit the LightFM model with `LightFM.fit`.
        """

        if interactions is None:
            interactions = self._ratings

        if weights is None:
            weights = self._weights

        self._model = LightFM(**{**self.DEFAULT_MODEL_ARGS, **model_args})
        self._model.fit(
            **{**self.DEFAULT_FIT_ARGS, **dict(interactions=interactions), **fit_args},
            sample_weight=weights)

    def recommend(self, user_ids=[], n=20):
        """Generate recommendations.

        Parameters
        ----------
        `user_ids`: `list[str]`
            List with usernames of users to generate recommendations for.
            If left empty, all users in provided interactions DataFrame are assumed.

        `n`: `int`
            Number of recommendations to generate per user.

        Returns
        -------
        recommendations: `list[list[str]]`
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
            raw_recs = list(self._filter_known_subreddits(username, raw_recs))

            # Filter recommendations from nsfw subreddits, until `n` is reached
            recs = []
            i = 0
            while (len(recs) < n) or i == 3:
                recs.extend(self._filter_nsfw(raw_recs[n * i:n * (i + 1)]))
                i += 1

            recommendations.append(recs[:n])

        return recommendations

    def _filter_known_subreddits(self, username, items, threshold=0):
        subs_interacted_with = self._ratings_df[self._ratings_df['user'] == username]\
            .query('count > {}'.format(threshold))['subreddit_id'].tolist()

        return (x for x in items if x not in (subs_interacted_with + DEFAULT_SUBREDDITS_IDS))

    def _filter_nsfw(self, subreddit_ids):
        subs_info = self.__r.info(subreddit_ids)
        return [info.display_name for (id, info) in zip(subreddit_ids, subs_info) if info.over18 is False]

    def evaluate(self, measures=['precision']):
        """Evaluate the performance of the recommender system.

        Parameters
        ----------
        `measures`: `list[str]`
            Possible values: precision, recall, auc.
        """

        train, test, user_index, test_interactions = h.train_test_split(
            self._ratings, 5, fraction=0.2)

        weights_copy = self._weights.copy().tolil()

        for user, interaction_index in test_interactions.items():
            weights_copy[user, interaction_index] = 0.

        weights_copy = weights_copy.tocoo()

        self.train(train, weights_copy)
        patk = precision_at_k(
            self._model, test_interactions=test, train_interactions=train, num_threads=4)
        auc = auc_score(self._model, test_interactions=test,
                        train_interactions=train, num_threads=4)
        print(patk.mean())
        print(auc.mean())
