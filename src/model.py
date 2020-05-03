from tensorflow import keras
import tensorflow as tf
import numpy as np
import os


class GaussianNoise(keras.layers.Layer):
    def __init__(self):
        super(GaussianNoise, self).__init__()
        self.gaussian_noise = keras.layers.GaussianNoise(stddev=2e-1)

    def generate_mask(self, inputs, share_mask=0.3):
        """
        Randomly sample ratings that will be subject to gaussian noise
        :param inputs: ratings input matrix
        :param share_mask: share of ratings that will get noise
        :return: binary matrix of (observed) ratings that will get noise
        """
        mask_positive_values = tf.cast(inputs > 0, dtype=tf.float32)
        batch_size, original_dim = tf.shape(inputs).numpy()
        mask_noise = np.random.uniform(0, 1, [batch_size, original_dim])
        mask_noise = tf.multiply(mask_noise, mask_positive_values)
        mask_noise = tf.where(mask_noise > 1 - share_mask, 1, 0)
        return tf.cast(mask_noise, dtype=tf.float32)

    def call(self, inputs):
        mask = self.generate_mask(inputs)
        inputs_with_noise = self.gaussian_noise(tf.multiply(inputs, mask))
        return inputs_with_noise, mask


class Encoder(keras.layers.Layer):
    def __init__(self,
                 original_dim,
                 latent_dim=100,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__()
        self.input_layer = keras.layers.InputLayer(input_shape=original_dim, dtype='float32', name='input')
        self.hidden_layer_1 = keras.layers.Dense(latent_dim,
                                               activation='sigmoid',
                                               kernel_regularizer=keras.regularizers.l2(1e-3),
                                               kernel_initializer=keras.initializers.TruncatedNormal(mean=0,stddev=0.05),
                                               dtype='float32')

    def call(self, inputs):
        x = self.input_layer(inputs)
        return self.hidden_layer_1(x)


class Decoder(keras.layers.Layer):
    def __init__(self,
                 original_dim,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__()
        self.hidden_layer = keras.layers.Dense(original_dim,
                                               activation='linear',
                                               kernel_regularizer=keras.regularizers.l2(1e-3),
                                               kernel_initializer=keras.initializers.TruncatedNormal(mean=0,stddev=0.05),
                                               dtype='float32')

    def call(self, inputs):
        return self.hidden_layer(inputs)


class AutoEncoder(keras.Model):
    def __init__(self,
                 batch_size,
                 original_dim,
                 latent_dim,
                 train=True,
                 name='autoencoder',
                 **kwargs):
        super(AutoEncoder, self).__init__()
        self.gaussian_noise = GaussianNoise()
        self.encoder = Encoder(latent_dim=latent_dim, original_dim=original_dim)
        self.decoder = Decoder(original_dim=original_dim)
        self.train = train

    def call(self, inputs):
        if self.train:
            inputs_with_noise, mask = self.gaussian_noise(inputs)
            x = self.encoder(inputs_with_noise)
            reconstructed = self.decoder(x)
            return reconstructed, mask

        x = self.encoder(inputs)
        reconstructed = self.decoder(x)
        return reconstructed


def save_model(model, current_day):

    model_dir = os.path.join("models", "autoenc" + current_day)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_weights(model_dir)


def compute_reconstruction_loss(output, input, mask_positive_values, mask_gaussian_noise):
    """
    :param output:
    :param input:
    :param mask_positive_values:
    :param mask_gaussian_noise:
    :return: loss for positive ratings that are NOT subject to gaussian noise
    """
    input = tf.multiply(input, 1 - mask_gaussian_noise)
    output = tf.multiply(output, 1 - mask_gaussian_noise)
    return tf.reduce_sum(tf.multiply(output - input, mask_positive_values) ** 2)


def compute_denoise_loss(output, input, mask_positive_values, mask_gaussian_noise):
    """
    :param output:
    :param input:
    :param mask_positive_values:
    :param mask_gaussian_noise:
    :return: loss for positive ratings that are subject to gaussian noise
    """
    input = tf.multiply(input, mask_gaussian_noise)
    output = tf.multiply(output, mask_gaussian_noise)
    return tf.reduce_sum(tf.multiply(output - input, mask_positive_values) ** 2)


def compute_loss(output, input, mask_gaussian_noise, alpha=0.3, beta=0.7):
    """
    :param output:
    :param input:
    :param mask_gaussian_noise: binary matrix of values sampled in gaussian_noise layer
    :param alpha: weight of denoising loss
    :param beta: weight of reconstruction loss
    :return: total loss excluding regularization
    """
    input = tf.cast(input, dtype=tf.float32)
    mask_positive_values = input > 0
    mask_positive_values = tf.cast(mask_positive_values, dtype=tf.float32)
    reconstruction_loss = compute_reconstruction_loss(output, input, mask_positive_values, mask_gaussian_noise)
    denoise_loss = compute_denoise_loss(output, input, mask_positive_values, mask_gaussian_noise)
    total_loss = alpha * denoise_loss + beta * reconstruction_loss
    return {'total_loss': total_loss, 'reconstruction_loss': reconstruction_loss, 'denoise_loss': denoise_loss}


def compute_accuracy(output, input):
    """
    Custom accuracy : compute the absolute difference input/output restricted to observed ratings
    :param output: reconstructed batch of user ratings
    :param input: tensor batch of user ratings
    :return: accuracy float value
    """
    input = tf.cast(input, dtype=tf.float32)
    mask = tf.cast(input > 0, dtype=tf.float32)
    share_positive_values = tf.math.divide(tf.constant(np.ones(tf.shape(input).numpy()[0]), dtype=tf.float32),
                                           tf.reduce_sum(mask, 1))  # 1 / (count  of positive values)

    return (5 - tf.reduce_mean(
        tf.multiply(
            tf.reduce_sum(tf.multiply(tf.math.abs(output - input), mask), 1),  # delta on > 0 values
            share_positive_values)  # divide by count > 0 values
    )) / 5  # sum and center on the range [0-5]
