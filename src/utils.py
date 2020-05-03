import pandas as pd
import numpy as np


def load_dataframe(data_path):
    df = pd.read_csv(data_path)
    return df


def transform_df_to_matrix(df, books_count=10000):

    users_count = df.user_id.nunique()
    X = np.zeros([users_count, books_count])

    df.sort_values(["user_id", "book_id"], inplace=True)
    index_values = df.set_index(["user_id", "book_id"]).to_dict()["rating"]

    for k, v in index_values.items():
        X[k[0]-1, k[1]-1] = v

    X = X.astype('float32')

    return X


def custom_train_test_split(df, X, share_hidden=0.2):
    """
    For each sample, hide a share of the ratings
    :param X: (user, books) array
    :return: Matrix with hidden ratings
    """
    df = df.sample(frac=1.0)   # shuffle
    df = df.groupby('user_id')['book_id'].apply(list).reset_index(name='book_id')
    values_to_reset = df[['book_id']].to_dict()['book_id']
    values_to_reset = [(k + 1, v[:max(int(share_hidden*len(v)), 1)]) for k, v in values_to_reset.items()]
    X_train = np.copy(X)
    X_test = np.zeros(X.shape)

    for (user_id, book_list) in values_to_reset:
        for book_id in book_list:
            X_test[user_id-1, book_id-1] = X_train[user_id-1, book_id-1]
            X_train[user_id-1, book_id-1] = 0

    return X_train, X_test


def compute_baseline_output(X):
    """
    Naive baseline : I predict the avg of > 0 ratings for all empty values
    """
    Z = np.ones(X.shape)
    mean_val = np.nanmean(np.where(X > 0, X, np.nan), axis=1)
    Z = (Z.T * mean_val).T
    return Z


def get_my_recommendations(user_id, X_predict, books_path):
    df_books = pd.read_csv(books_path)
    df_books = df_books[['book_id', 'title', 'authors']]
    df_books['my_rating'] = X_predict[user_id - 1]
    df_books.sort_values("my_rating", ascending=False, inplace=True)
    return df_books
