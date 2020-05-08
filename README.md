# goodreads-recommender

## Objectives
This project contains a deep learning autoencoder applied to the inference of unobserved rating data, implemented with Tensorflow 2.0. The data comes from the public release of 53k+ users ratings of 10k books from Goodeads.com. This model is inspired frm the reference paper *"Hybrid Recommender System Based on Autoencoders"* which is available [here](https://hal.inria.fr/hal-01336912v2/document).

### Background
#### Introduction to autoencoder-based recommenders
Given a sparse user/item matrix of ratings, an Encoder layer learns a compressed representation of this input matrix. Then, given this latent representation, a Decoder layer learns to reconstruct a dense matrix of ratings of the same size as the input matrix. The combination of Encoder-Decoder aims at recovering the unboserved user/item ratings.

This network has two specificities :
* The batch loss is computed on observed ratings only. Therefore we apply a mask to hide the other values when computing the loss. The backpropagation only accounts for observed errors.
* We apply L2 regularization on both Encoder and Decoder weights, so that the weights do not degenerate to identity matrices (that is, reconstructing the exact input matrix)

#### Denoising autoencoder (DAE)
One method to fight degeneracy of weights is to add Gaussian noise to a fraction of the input data. Now the objective of the network is two-fold :
* Reconstructing the input matrix where the input data has not changed
* Denoising the input data where the input data has been corrupted

This is essentially what the denoising autoencoder (DAE) does. In essence, the loss function of the DAE breaks down into two components : the reconstruction loss and the denoising loss.


## Getting started

Install the packages
```
pip install requirements.txt
```

Train the model
```
python src/train.py
```

Reconstruct the full user/book matrix
```
python src/predict.py
```
