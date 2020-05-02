import tensorflow as tf
import utils
from Language_Model import LanguageModel

source_path = tf.keras.utils.get_file(
    fname="shakespeare.txt",
    origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(file=source_path, mode="rb").read().decode(encoding="utf-8")

vocablury = utils.Vocablury(text=text)


language_model = LanguageModel(
    vocablury=vocablury,
    embedding_dim=256,
    rnn_units=1024,
    batch_size=1
)

# language_model.set_train_text(text=text, sample_length=100, buffer_size=10000)
# language_model.train(epochs=40)

language_model.load_model()

print(language_model.sample(init_phrase="Hey", length=1000))

# def loss(labels, logits):
#     return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
