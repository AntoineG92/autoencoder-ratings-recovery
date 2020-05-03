import argparse


def parse_train_arguments():

    parser = argparse.ArgumentParser(
        description="Autorec training arguments")

    parser.add_argument("--latent-dim", type=int, required=False, default=500)

    parser.add_argument("--count-epochs", type=int, required=False, default=12)

    parser.add_argument("--batch-size", type=int, required=False, default=500)

    parser.add_argument("--learning-rate", type=float, required=False, default=1e-3)

    parser.add_argument("--data-path", type=str, required=False, default="data/ratings.csv")

    args = parser.parse_args()

    return vars(args)


def parse_predict_arguments():

    parser = argparse.ArgumentParser(
        description="Autorec predict arguments")

    parser.add_argument("--model-path", type=str, required=False, default='models/autoenc20200502')

    parser.add_argument("--data-books", type=str, required=False, default="data/books.csv")

    parser.add_argument("--data-path", type=str, required=False, default="data/ratings.csv")

    parser.add_argument("--latent-dim", type=int, required=False, default=500)

    args = parser.parse_args()

    return vars(args)


def parse_recommend_arguments():

    parser = argparse.ArgumentParser(
        description="Autorec recommend arguments")

    parser.add_argument("--matrix-path", type=str, required=False, default='data/output_matrix.npy')

    parser.add_argument("--books-path", type=str, required=False, default='data/books.csv')

    parser.add_argument("--user_id", type=int, required=False, default=1)

    args = parser.parse_args()

    return vars(args)
