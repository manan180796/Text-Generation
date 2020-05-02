import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras import Model
from tensorflow import keras
import utils
import datetime
import os


def split_sequences(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


class LanguageModel(object):
    def __init__(
            self,
            vocablury: utils.Vocablury,
            embedding_dim: int,
            rnn_units: int,
            batch_size: int):

        self.vocablury = vocablury
        self.text_vectorizer = utils.Text_Vectorizer(vocablury=self.vocablury)
        self.batch_size = batch_size
        vocab_size = len(self.vocablury)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer="glorot_uniform"),
            tf.keras.layers.Dense(vocab_size)
        ])

    def set_train_text(self, text: str,
                       sample_length: int,
                       buffer_size: int):
        text_vector = self.text_vectorizer.vectorize(text=text)

        dataset = tf.data.Dataset.from_tensor_slices(text_vector)

        slices = dataset.batch(batch_size=sample_length+1, drop_remainder=True)

        slices = slices.map(map_func=split_sequences)

        self.dataset = slices.shuffle(buffer_size=buffer_size).batch(
            batch_size=self.batch_size,
            drop_remainder=True)

    def train(self, loss=loss, epochs: int = 10):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        # for input_example_batch, target_example_batch in self.dataset.take(1):
        #     example_batch_predictions = self.model(input_example_batch)
        #     print(example_batch_predictions.shape,
        #           "# (batch_size, sequence_length, vocab_size)")
        # example_batch_loss = loss(
        #     target_example_batch, example_batch_predictions)
        # print("Prediction shape: ", example_batch_predictions.shape,
        #       " # (batch_size, sequence_length, vocab_size)")
        # print("scalar_loss:      ", example_batch_loss.numpy().mean())
        self.model.compile(optimizer="adam", loss=loss)

        # Directory where the checkpoints will be saved
        checkpoint_dir = "./model/training_checkpoints"
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.model.fit(self.dataset, epochs=epochs, callbacks=[
                                 checkpoint_callback, tensorboard_callback])

    def load_model(self, path_to_model: str = "./model/training_checkpoints"):
        self.model.load_weights(tf.train.latest_checkpoint(path_to_model))

    def sample(self, length: int, init_phrase: str, temperature: float = 1.0) -> str:
        text = init_phrase
        vector = self.text_vectorizer.vectorize(text=init_phrase)
        input_vector = tf.expand_dims(input=vector, axis=0)
        print(input_vector)
        for _ in range(len(init_phrase), length):
            prediction = self.model(input_vector)
            prediction = tf.squeeze(input=prediction, axis=0)
            prediction /= temperature
            prediction_char = tf.random.categorical(
                logits=prediction, num_samples=1)[-1, 0].numpy()

            text += self.vocablury.index_to_char[prediction_char]
            input_vector = tf.expand_dims(input=[prediction_char], axis=0)
        return text
