import numpy as np
import praw
import scipy.sparse as sp

from pandas import DataFrame

from lightfm import LightFM

from . import helpers as h
from . import config
from .default_subreddits import DEFAULT_SUBREDDITS_IDS


class Recommender:

    DEFAULT_MODEL_ARGS = dict(loss='warp', learning_schedule='adagrad',
                              user_alpha=1e-06, item_alpha=1e-06, no_components=50)
    DEFAULT_FIT_ARGS = dict(epochs=100, num_threads=4)

    def __init__(self, interactions_df: DataFrame):
        """
        Parameters
        ----------
        `interactions` : `pandas.DataFrame`
            Interaction matrix containing users and items (columns: `user`, `subreddit_id`)
            It's possible to provide a count and a weight per interaction with a `count` and
            `weight` column.
        """

        self._interactions_df = interactions_df
        self.__r = praw.Reddit(client_id=config.REDDIT_CLIENT_ID,
                               client_secret=config.REDDIT_CLIENT_SECRET,
                               user_agent=config.REDDIT_USER_AGENT)
        self.__setup()

    def __setup(self):
        self._interactions, self._weights, self._uid_to_idx, self._idx_to_uid,\
            self._sid_to_idx, self._idx_to_sid = self._extract_compact_representation(
                self._interactions_df)

    def _extract_compact_representation(self, interactions_df: DataFrame, column_labels={}):
        """
        Create compact representations of interaction matrix and weights associated with ratings.
        """

        default_column_labels = dict(user='user', item='subreddit_id', rating='count',
                                     weight='weight')
        column_labels = {**default_column_labels, **column_labels}

        rid_to_idx, idx_to_rid,\
            cid_to_idx, idx_to_cid = h.get_df_matrix_mappings(
                interactions_df,
                column_labels['user'],
                column_labels['item'])

        def map_ids(row, mapper):
            return mapper[row]

        i = interactions_df[column_labels['user']].apply(
            map_ids, args=[rid_to_idx]).as_matrix()
        j = interactions_df[column_labels['item']].apply(
            map_ids, args=[cid_to_idx]).as_matrix()
        # Assume all ratings are 1.0 if ratings column doesn't exist
        v = interactions_df[column_labels['rating']].as_matrix(
        ) if column_labels['rating'] in interactions_df else np.ones(i.shape[0])
        w = interactions_df[column_labels['weight']].as_matrix(
        ) if column_labels['weight'] in interactions_df else np.ones(i.shape[0])

        interactions = sp.coo_matrix((v, (i, j)), dtype=np.float64)
        weights = sp.coo_matrix((w, (i, j)), dtype=np.float64)

        return interactions, weights, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

    def train(self, interactions=None, weights=None, model_args={}, fit_args={}):
        """Instantiate and train LightFM model.

        Parameters
        ----------
        `interactions` : `scipy csr matrix`
            Interactions used to train the model. Defaults to all interactions.

        `model_args` : `dict`
            Arguments used to instantiate `LightFM`.

        `fit_args` : `dict`
            Arguments used to fit the LightFM model with `LightFM.fit`.
        """

        if interactions is None:
            interactions = self._interactions

        if weights is None:
            weights = self._weights

        self._model = LightFM(**{**self.DEFAULT_MODEL_ARGS, **model_args})
        self._model.fit(**{**self.DEFAULT_FIT_ARGS,
                           **dict(interactions=interactions, sample_weight=weights),
                           **fit_args})

    def recommend(self, user_ids=[], n=20):
        """Generate recommendations.

        Parameters
        ----------
        `user_ids` : `list[str]`
            List with usernames of users to generate recommendations for.
            If left empty, all users in provided interactions DataFrame are assumed.

        `n` : `int`
            Number of recommendations to generate per user.

        Returns
        -------
        recommendations : `list[list[str]]`
            List containing lists with recommendations for users.
        """

        # Default to all users
        if len(user_ids) == 0:
            user_ids = np.arange(self._interactions.shape[0])
        else:
            if isinstance(user_ids, str):
                user_ids = [user_ids]

            user_ids = [self._uid_to_idx[u] for u in user_ids]

        # Default to all items
        item_ids = np.arange(self._interactions.shape[1])

        recommendations = []

        for user_id in user_ids:
            username = self._idx_to_uid[user_id]
            scores = self._model.predict(user_id, item_ids)
            raw_recs = np.array(list(self._idx_to_sid.values()))[
                np.argsort(-scores)]
            raw_recs = list(self._filter_known_items(username, raw_recs))

            # Filter recommendations from nsfw subreddits, until `n` is reached
            recs = []
            i = 0
            while (len(recs) < n) or i == 3:
                recs.extend(self._filter_nsfw(raw_recs[n * i:n * (i + 1)]))
                i += 1

            recommendations.append(recs[:n])

        return recommendations

    def _filter_known_items(self, username, items, threshold=0):
        """
        Filter default subreddits and subreddits the user has already interacted with.

        Parameters
        ----------
        `threshold` : `int`
            How many interactions must a user have had with an item before it's filtered
        """
        subs_interacted_with = self._interactions_df[self._interactions_df['user'] == username]\
            .query('count > {}'.format(threshold))['subreddit_id'].tolist()

        return (sub for sub in items if sub not in (subs_interacted_with + DEFAULT_SUBREDDITS_IDS))

    def _filter_nsfw(self, subreddit_ids):
        """
        Filter subreddits that are marked as nsfw.
        """

        subs_info = self.__r.info(subreddit_ids)
        return [info.display_name for (id, info) in zip(subreddit_ids, subs_info)
                if info.over18 is False]
