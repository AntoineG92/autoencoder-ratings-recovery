from src.utils import get_my_recommendations
from src.args import parse_recommend_arguments

import numpy as np


if __name__ == "__main__":

    parameters = parse_recommend_arguments()
    print(parameters)

    X_predict = np.load(parameters['matrix_path'])
    user_id = parameters['user_id']

    df = get_my_recommendations(user_id, X_predict, parameters['books_path'])

    print(f"Top 20 books recommended {df.head(20)}")
