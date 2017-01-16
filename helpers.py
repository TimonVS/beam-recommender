import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import praw

DEFAULT_SUBS = [
    'art',
    'askscience',
    'creepy',
    'dataisbeautiful',
    'DIY',
    'Documentaries',
    'gadgets',
    'GetMotivated',
    'OldSchoolCool',
    'nosleep',
    'announcements',
    'mildlyinteresting',
    'trees',
    'Games',
    'LifeProTips',
    'todayilearned',
    'funny',
    'politics',
    'sports',
    'Music',
    'EarthPorn',
    'gaming',
    'aww',
    'IAmA',
    'AdviceAnimals',
    'science',
    'worldnews',
    'explainlikeimfive',
    'television',
    'bestof',
    'askscience',
    'Futurology',
    'pics',
    'movies',
    'news',
    'gifs',
    'videos',
    'AskReddit',
    'books',
    'technology',
    'funny',
    'fffffffuuuuuuuuuuuu',
    'Showerthoughts',
    'tifu',
    'TwoXChromosomes',
    'photoshopbattles',
    'WritingPrompts'
]

DEFAULT_SUBS_IDS = [
    't5_2qh7a',
    't5_2qm4e',
    't5_2raed',
    't5_2tk95',
    't5_2qh7d',
    't5_2qhlh',
    't5_2qgzt',
    't5_2rmfx',
    't5_2tycb',
    't5_2rm4d',
    't5_2r0ij',
    't5_2ti4h',
    't5_2r9vp',
    't5_2qhwp',
    't5_2s5oq',
    't5_2qqjc',
    't5_2qh33',
    't5_2cneq',
    't5_2qgzy',
    't5_2qh1u',
    't5_2sbq3',
    't5_2qh03',
    't5_2qh1o',
    't5_2qzb6',
    't5_2s7tt',
    't5_mouw',
    't5_2qh13',
    't5_2sokd',
    't5_2qh6e',
    't5_2qh3v',
    't5_2qm4e',
    't5_2t7no',
    't5_2qh0u',
    't5_2qh3s',
    't5_2qh3l',
    't5_2qt55',
    't5_2qh1e',
    't5_2qh1i',
    't5_2qh4i',
    't5_2qh16',
    't5_2qh33',
    't5_2qqlo',
    't5_2szyo',
    't5_2to41',
    't5_2r2jt',
    't5_2tecy',
    't5_2s3nb'
]

def calc_sparsity_df(df, row_name, col_name):
    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Starting interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of cols: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

def threshold_interactions_df(df, row_name, col_name, row_min, col_min):
    """Limit interactions df to minimum row and column interactions.

    Parameters
    ----------
    df : DataFrame
        DataFrame which contains a single row for each interaction between
        two entities. Typically, the two entities are a user and an item.
    row_name : str
        Name of column in df which corresponds to the eventual row in the
        interactions matrix.
    col_name : str
        Name of column in df which corresponds to the eventual column in the
        interactions matrix.
    row_min : int
        Minimum number of interactions that the row entity has had with
        distinct column entities.
    col_min : int
        Minimum number of interactions that the column entity has had with
        distinct row entities.
    Returns
    -------
    df : DataFrame
        Thresholded version of the input df. Order of rows is not preserved.

    Examples
    --------

    df looks like:

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004
      1002  |  2002

    thus, row_name = 'user_id', and col_name = 'item_id'

    If we were to set row_min = 2 and col_min = 1, then the returned df would
    look like

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004

    """

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Starting interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of cols: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    done = False
    while not done:
        starting_shape = df.shape[0]
        col_counts = df.groupby(row_name)[col_name].count()
        df = df[~df[row_name].isin(col_counts[col_counts < col_min].index.tolist())]
        row_counts = df.groupby(col_name)[row_name].count()
        df = df[~df[col_name].isin(row_counts[row_counts < row_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Ending interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of columns: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df


def get_df_matrix_mappings(df, row_name, col_name):
    """Map entities in interactions df to row and column indices

    Parameters
    ----------
    df : DataFrame
        Interactions DataFrame.
    row_name : str
        Name of column in df which contains row entities.
    col_name : str
        Name of column in df which contains column entities.

    Returns
    -------
    rid_to_idx : dict
        Maps row ID's to the row index in the eventual interactions matrix.
    idx_to_rid : dict
        Reverse of rid_to_idx. Maps row index to row ID.
    cid_to_idx : dict
        Same as rid_to_idx but for column ID's
    idx_to_cid : dict
    """


    # Create mappings
    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def df_to_matrix(df, row_name, col_name, value_name=None):
    """Take interactions dataframe and convert to a sparse matrix

    Parameters
    ----------
    df : DataFrame
    row_name : str
    col_name : str

    Returns
    -------
    interactions : sparse csr matrix
    rid_to_idx : dict
    idx_to_rid : dict
    cid_to_idx : dict
    idx_to_cid : dict

    """

    rid_to_idx, idx_to_rid,\
        cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,
                                                        row_name,
                                                        col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).as_matrix()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).as_matrix()
    V = df[value_name].as_matrix() if value_name else np.ones(I.shape[0])
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def train_test_split(interactions, split_count, fraction=None):
    """
    Split recommendation data into train and test sets

    Params
    ------
    interactions : scipy.sparse matrix
        Interactions between users and items.
    split_count : int
        Number of user-item-interactions per user to move
        from training to test set.
    fractions : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all
        users are considered.
    """
    # Note: likely not the fastest way to do things below.
    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        test_interactions = np.random.choice(interactions.getrow(user).indices,
                                        size=split_count,
                                        replace=False)
        train[user, test_interactions] = 0.
        # These are just 1.0 right now
        test[user, test_interactions] = interactions[user, test_interactions]


    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index
