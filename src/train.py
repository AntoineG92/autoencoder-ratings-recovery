import tensorflow as tf
from datetime import datetime

from src.model import AutoEncoder, save_model, compute_loss, compute_accuracy
from src.utils import load_dataframe, transform_df_to_matrix, custom_train_test_split, compute_baseline_output
from src.args import parse_train_arguments

import logging


def write_and_print_results(epoch, loss_dict, accuracy_value, loss_dict_test, accuracy_value_test,
                            current_time, current_day):

    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    with train_summary_writer.as_default():
        tf.summary.scalar('total_loss', loss_dict['total_loss'], step=epoch)
        tf.summary.scalar('reconstruction_loss', loss_dict['reconstruction_loss'], step=epoch)
        tf.summary.scalar('denoise_loss', loss_dict['denoise_loss'], step=epoch)
        tf.summary.scalar('accuracy', accuracy_value, step=epoch)

    with test_summary_writer.as_default():
        tf.summary.scalar('total_loss', loss_dict_test['total_loss'], step=epoch)
        tf.summary.scalar('reconstruction_loss', loss_dict_test['reconstruction_loss'], step=epoch)
        tf.summary.scalar('denoise_loss', loss_dict_test['denoise_loss'], step=epoch)
        tf.summary.scalar('accuracy test', accuracy_value_test, step=epoch)

    template = 'Epoch {}, Total Loss: {}, Reconstruction Loss: {}, Denoise Loss: {},' \
               'Accuracy: {}, Loss test:{}, Accuracy test:{}'
    logging.warning(template.format(
        epoch + 1,
        loss_dict['total_loss'],
        loss_dict['reconstruction_loss'],
        loss_dict['denoise_loss'],
        accuracy_value * 100,
        loss_dict_test['total_loss'],
        accuracy_value_test * 100
    ))


if __name__ == "__main__":

    logging.warning("initial state")

    parameters = parse_train_arguments()

    logging.warning(f"PARAMETERS {parameters}")

    current_time = datetime.now().strftime("%Y%m%d - %H%M")
    current_day = datetime.now().strftime("%Y%m%d")

    # Load the data and split train/test
    df = load_dataframe(parameters['data_path'])
    X = transform_df_to_matrix(df)
    X_train, X_test = custom_train_test_split(df, X)
    logging.warning(f"Train size {X_train.shape} - Test size {X_test.shape}")

    output_baseline = compute_baseline_output(X)
    logging.warning(f"Baseline accuracy {compute_accuracy(X,output_baseline) * 100}")

    # transform the data to tensors
    data_set = tf.data.Dataset.from_tensor_slices((X_train, X_test)).batch(parameters["batch_size"])

    # create the model
    autoenc = AutoEncoder(batch_size=parameters['batch_size'],
                          original_dim=X_train.shape[1],
                          latent_dim=parameters['latent_dim'])

    # adding optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate'])

    # Training
    count_epochs = parameters['count_epochs']
    for epoch in range(count_epochs):
        logging.warning(f"start of epoch {epoch}")

        for step, (x_batch_train, x_batch_test) in enumerate(data_set):
            with tf.GradientTape() as tape:

                reconstructed, mask_gaussian_noise = autoenc(x_batch_train)
                loss_dict = compute_loss(reconstructed, x_batch_train, mask_gaussian_noise, alpha=0.5)
                loss_dict['total_loss'] += sum(autoenc.losses)   # add regularization losses
                accuracy_value = compute_accuracy(reconstructed, x_batch_train)

            grads = tape.gradient(loss_dict['total_loss'], autoenc.trainable_weights)
            optimizer.apply_gradients(zip(grads, autoenc.trainable_weights))

        # test at the end of each epoch
        loss_dict_test = compute_loss(reconstructed, x_batch_test, mask_gaussian_noise, alpha=0.5)
        accuracy_value_test = compute_accuracy(reconstructed, x_batch_test)

        write_and_print_results(epoch, loss_dict, accuracy_value, loss_dict_test, accuracy_value_test,
                                current_time, current_day)

    logging.warning(f"model summary {autoenc.summary()}")

    save_model(autoenc, current_day)
