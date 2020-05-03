# goodreads-recommender

## Objectives
This project contains a deep learning autoencoder applied to reconstruction of unobserved ratings. The data comes from the public release of 53k+ users ratings of 10k books from Goodeads.com. The autoencoder is based on the paper *"Hybrid Recommender System Based on Autoencoders"* available [here](https://hal.inria.fr/hal-01336912v2/document).

### Background
Given a sparse user/item matrix of ratings with unobserved data, a deep learning model learns a compressed representation of the input data. Then it learns to reconstruct a dense matrix of ratings of the same size as the input matrix. This new matrix includes a recovery of all user/item ratings (observed and unobserved).


## Getting started
