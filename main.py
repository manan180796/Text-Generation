import tensorflow as tf
import utils
from Language_Model import LanguageModel
import argparse

rnn_units = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--test", action="store_true", required=False)
    parser.add_argument("--start", type=str, required=False)
    parser.add_argument("--length", type=int, required=False)
    args = parser.parse_args()
    source_path = tf.keras.utils.get_file(
        fname="shakespeare.txt",
        origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    )
    text = open(file=source_path, mode="rb").read().decode(
        encoding="utf-8")

    vocablury = utils.Vocablury(text=text)

    if args.train:
        print("------------------------hello-----------------------")

        language_model = LanguageModel(
            vocablury=vocablury,
            embedding_dim=256,
            rnn_units=rnn_units,
            batch_size=64
        )

        language_model.set_train_text(
            text=text,
            sample_length=100,
            buffer_size=10000)
        language_model.train(epochs=args.epochs)
    else:
        source_path = tf.keras.utils.get_file(
            fname="shakespeare.txt",
            origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
        )
        text = open(file=source_path, mode="rb").read().decode(
            encoding="utf-8")

        vocablury = utils.Vocablury(text=text)

        language_model = LanguageModel(
            vocablury=vocablury,
            embedding_dim=256,
            rnn_units=rnn_units,
            batch_size=1
        )

        language_model.load_model()

        print(language_model.sample(init_phrase=args.start, length=args.length))


if __name__ == "__main__":
    main()
