from src.model import AutoEncoder
from src.train import load_dataframe
from src.train import transform_df_to_matrix
from src.args import parse_predict_arguments

import logging
import numpy as np
import pandas as pd


if __name__ == "__main__":

    parameters = parse_predict_arguments()
    logging.warning(parameters)

    df_books = pd.read_csv(parameters["data_books"])

    logging.warning("load the data to predict")
    df = load_dataframe(parameters['data_path'])

    df = df.merge(df_books[['book_id', 'title']], on='book_id')
    df.sort_values('rating', ascending=False, inplace=True)

    X = transform_df_to_matrix(df)

    if parameters['item_based']:
        X = X.T

    model_dir = parameters['model_path']

    logging.warning("create the model")
    autoenc = AutoEncoder(batch_size=X.shape[0], original_dim=X.shape[1],
                          latent_dim=parameters['latent_dim'], train=False)

    logging.warning("load the model weights")
    autoenc.load_weights(model_dir)

    logging.warning("predict")
    output = autoenc(X).numpy()

    logging.warning("store the output user/book matrix")
    if parameters['item_based']:
        output = output.T
    np.save('data/output_matrix', output)
