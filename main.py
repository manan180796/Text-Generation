import tensorflow as tf
import utils

source_path = tf.keras.utils.get_file(
    fname="shakespeare.txt",
    origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(file=source_path, mode="rb").read().decode(encoding="utf-8")
print(len(text))

text_vectorizer = utils.Text_Vectorizer(text=text)

text_vector = text_vectorizer.vectorize(text=text)

sample_length = 100

dataset = tf.data.Dataset.from_tensor_slices(text_vector)


slices = dataset.batch(batch_size=sample_length+1, drop_remainder=True)


def split_sequences(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


slices = slices.map(map_func=split_sequences)


batch_size = 64
buffer_size = 10000
dataset = slices.shuffle(buffer_size=buffer_size).batch(
    batch_size=batch_size, drop_remainder=True)

print(dataset)
