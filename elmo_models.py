import argparse
from labeler import BinaryLabels
from elmo import *
import numpy as np
np.random.seed(123)
import tensorflow as tf
import keras.layers as layers
from keras.models import Model
from utils import *

# Function to build model
def build_model():
  input_text = layers.Input(shape=(1,), dtype="string")
  embedding = ElmoEmbeddingLayer()(input_text)
  dense = layers.Dense(256, activation='relu')(embedding)
  pred = layers.Dense(2, activation='sigmoid')(dense)

  model = Model(inputs=[input_text], outputs=pred)

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()

  return model


def do_experiment(args):
    # Read in training and testing data
    labeler = BinaryLabels()
    train_text = ElmoParser(args.training, args.train_size, args.article_length).all_text
    train_label = labeler.process(args.labels, args.train_size)
    test_text = ElmoParser(args.testing, args.test_size, args.article_length).all_text
    test_label = labeler.process(args.test_labels, args.test_size)

    # print("train_text", train_text)
    # print("train_labels", train_labels)

    sess = tf.Session()
    K.set_session(sess)

    # Build and fit
    model = build_model()
    model.fit(train_text,
              train_label,
              validation_data=(test_text, test_label),
              epochs=1,
              batch_size=32)


if __name__ == '__main__':
    # Initialize session

    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=argparse.FileType('rb'), help="Training articles")
    parser.add_argument("labels", type=argparse.FileType('rb'), help="Training article labels")
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances.", default=None)
    parser.add_argument("-l", "--article_length", type=int, metavar="N", help="Article max length", default=200)
    parser.add_argument("-t", "--testing", type=argparse.FileType('rb'), metavar="FILE", default=None)
    parser.add_argument("--test_labels", type=argparse.FileType('rb'), help="Test article labels", default=None)
    parser.add_argument("--test_size", type=int, metavar="N", help="Only test on the first N instances.", default=None)


    args = parser.parse_args()
    do_experiment(args)

    for fp in (args.training, args.labels, args.testing, args.test_labels): fp.close()
