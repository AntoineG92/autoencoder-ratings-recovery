import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def load_dataframe():

    df = pd.read_csv("data/ratings.csv")
    return df


def transform_df_to_matrix(df):

    N_USERS, N_BOOKS = df.user_id.nunique(), df.book_id.nunique()

    # make sure id's start from zero
    df["user_id"] += -1
    df['book_id'] += -1

    df.sort_values(["user_id", "book_id"], inplace=True)
    index_values = df.set_index(["user_id", "book_id"]).to_dict()["rating"]

    X = np.zeros([N_USERS, N_BOOKS])
    for k, v in index_values.items():
        X[k] = v

    X = csr_matrix(X, dtype=np.int16)
    return X

if __name__=="__main__":

    df = load_dataframe()

