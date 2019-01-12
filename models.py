# import tensorflow as tf
import argparse
import sys
from labeler import BinaryLabels
import numpy as np
np.random.seed(123)
from keras.layers import Dropout, Dense, Bidirectional, LSTM, \
    Embedding, GaussianNoise, Activation, SpatialDropout1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from kutilities.layers import AttentionWithContext, Attention
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from parser import Parser
import dill as pickle
from utils import *
# HOME_DIR = '/Users/lngoquy'


def evaluate_result(preds, truth_labels):
    preds_output = np.where(preds > 0.5, 1, 0)[:,1]
    if len(truth_labels.shape) == 2:
        truth_labels = truth_labels[:,1]
    precision, recall, fscore, _ = precision_recall_fscore_support(truth_labels, preds_output, average="binary")
    return precision, recall, fscore


def get_lstm(cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    rnn = LSTM(cells, return_sequences=return_sequences,
               consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

def get_embedding(embedding_dim, word_index):
    # Get glove embedding
    embeddings_index = dict()
    f = open('/scratch/lngoquy1/glove.6B/glove.6B.'+str(embedding_dim)+'d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # Create a weight matrix for words in training doc
    vocabulary_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, index in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

def train_lstm(vocab, n_class, num_features, embed_output_dim=300, lstm_output_dim=150,
               noise=0.3, layers=2, bi=True, attention="simple", dropout_attention=0.5,
               dropout_words=0.3, dropout_rnn=0.3, dropout_rnn_U=0.3, clipnorm=1, lr=0.001, loss_l2=0.0001):
    model = Sequential()
    # Embedding layer
    embed_input_dim = len(vocab) + 1
    embedding_matrix = get_embedding(embed_output_dim, vocab)
    model.add(Embedding(embed_input_dim, embed_output_dim, input_length = num_features))
    # model.add(Embedding(embed_input_dim, embed_output_dim, input_length = num_features,
    #                    weights = [embedding_matrix], trainable = False))
    model.add(SpatialDropout1D(0.2))
    # GaussianNoise layer
    if noise > 0:
        model.add(GaussianNoise(noise))

    # Dropout layer
    if dropout_words > 0:
        model.add(Dropout(dropout_words))

    for i in range(layers):
        rs = (layers > 1 and i < layers - 1) or attention
        model.add(get_lstm(lstm_output_dim, bi, return_sequences=rs,
                          dropout_U=dropout_rnn_U))
        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))
    if attention == "memory":
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    elif attention == "simple":
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))

    model.add(Dense(n_class, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy')
    return model

def train_dev_split(features, labels, split_ratio):
    split_index = int(split_ratio * len(features))
    X_train, X_dev = features[:split_index], features[split_index:]
    Y_train, Y_dev = labels[:split_index], labels[split_index:]
    return X_train, Y_train, X_dev, Y_dev

def run_lstm_exp(train_features, train_labels, train_vocab, batch_size, num_epochs,
                layers, bi, attention, dropout_attention, dropout_rate, test_features=[], test_labels=[]):

    if len(test_features) == 0:
        X_train, Y_train, X_test, Y_test = train_dev_split(train_features, train_labels, split_ratio = 0.8)
    else:
        X_train, Y_train, X_test, Y_test = train_features, train_labels, test_features, test_labels
    lstm_model = train_lstm(train_vocab, n_class=Y_train.shape[1], num_features=X_train.shape[1],
                            layers=layers, bi=bool(bi), attention=attention, dropout_attention=dropout_attention,
                            dropout_words=dropout_rate, dropout_rnn=dropout_rate, dropout_rnn_U=dropout_rate)
    print(lstm_model.summary())

    lstm_model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=2)
    preds = lstm_model.predict(X_test, verbose = 2)
    precision, recall, fscore = evaluate_result(preds, Y_test)
    pickle.dump(lstm_model, open("lstm_model.pkl", "wb"))
    print(precision, recall, fscore)


def run_rf_exp(train_features, train_labels, train_feature_dict, test_features=[], test_labels=[]):
    if len(test_features) == 0:
        X_train, Y_train, X_test, Y_test = train_dev_split(train_features, train_labels, split_ratio = 0.8)
    else:
        X_train, Y_train, X_test, Y_test = train_features, train_labels, test_features, test_labels

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    clf.fit(X_train, Y_train)
    preds = clf.predict_proba(X_test)
    precision, recall, fscore = evaluate_result(preds[1], Y_test)
    print(precision, recall, fscore)

    # Plot the feature importance
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(25):
        print("%d. feature %s (%f)" % (f + 1, train_feature_dict[indices[f]], importances[indices[f]]))
    # Persist the model
    pickle.dump(clf, open("rf_model.pkl", "wb"))
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X_train.shape[1]), importances[indices],
    #        color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X_train.shape[1]), indices)
    # plt.xlim([-1, X_train.shape[1]])
    # plt.show()

def run_rf_exp_bagging(train_features, train_labels, train_feature_dict, test_features=[], test_labels=[]):
    train_labels = np.nonzero(train_labels)[1]
    if len(test_features) == 0:
        X_train, Y_train, X_test, Y_test = train_dev_split(train_features, train_labels, split_ratio = 0.8)
    else:
        test_labels = np.nonzero(test_labels)[1]
        X_train, Y_train, X_test, Y_test = train_features, train_labels, test_features, test_labels

    clf = BaggingClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1))

    clf.fit(X_train, Y_train)
    preds = clf.predict_proba(X_test)
    precision, recall, fscore = evaluate_result(preds, Y_test)
    print(precision, recall, fscore)

def run_adaboost(train_features, train_labels, test_features=[], test_labels=[]):
    train_labels = np.nonzero(train_labels)[1]
    if len(test_features) == 0:
        X_train, Y_train, X_test, Y_test = train_dev_split(train_features, train_labels, split_ratio = 0.8)
    else:
        test_labels = np.nonzero(test_labels)[1]
        X_train, Y_train, X_test, Y_test = train_features, train_labels, test_features, test_labels
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, Y_train)
    preds = clf.predict_proba(X_test)
    precision, recall, fscore = evaluate_result(preds, Y_test)
    print(precision, recall, fscore)


def do_experiment(args):
    labeler = BinaryLabels()
    train_parser = Parser(args.training, args.train_size, args.article_length)
    train_labels = labeler.process(args.labels, args.train_size)
    if args.test_labels != None and args.test_data != None:
        test_labels = labeler.process(args.test_labels, args.test_size)
        test_parser = Parser(args.test_data, args.test_size, args.article_length)

    if args.model == 0:
        train_parser.map_features_parallel()
        if args.test_data != None:
            test_parser.map_features_parallel(vocab = train_parser.vocab)
    else:
        train_parser.map_general_features()
        if args.test_data != None:
            test_parser.map_general_features(vocab = train_parser.vocab,
                                            hyperlinks = train_parser.hyperlinks)

    if args.model == 0:
        run_lstm_exp(train_parser.X, train_labels, train_parser.vocab, args.batch, args.epoch,
                    args.layers, args.bi, args.attention, args.dropout_attention, args.dropout_rate,
                    test_parser.X, test_labels)
    elif args.model == 1:
        run_rf_exp(train_parser.all_feats, train_labels, train_parser.feature_dict)
        # run_rf_exp(train_parser.all_feats, train_labels, train_parser.feature_dict, test_parser.all_feats, test_labels)
    elif args.model == 2:
        run_adaboost(train_parser.all_feats, train_labels)
        # run_adaboost(train_parser.all_feats, train_labels, test_parser.all_feats, test_labels)
    elif args.model == 3:
        run_rf_exp_bagging(train_parser.all_feats, train_labels, train_parser.feature_dict)
        # run_rf_exp_bagging(train_parser.all_feats, train_labels, train_parser.feature_dict, test_parser.all_feats, test_labels)
    else:
      print("Choose model option from 0-3")






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=argparse.FileType('rb'), help="Training articles")
    parser.add_argument("labels", type=argparse.FileType('rb'), help="Training article labels")
    parser.add_argument("-l", "--article_length", type=int, metavar="N", help="Number of words to consider in article", default=200)
    parser.add_argument("-m", "--model", type=int, metavar="N", help="0: LSTM, 1: RF, 2: Adaboost", default=1)
    parser.add_argument("-e", "--epoch", type=int, metavar="N", help="Number of epochs", default=50)
    parser.add_argument("-b", "--batch", type=int, metavar="N", help="Batch size", default=50)
    parser.add_argument("-a", "--attention", metavar="N", help="Attention", default="simple")
    parser.add_argument("--bi", type=int, metavar="N", help="Bidirectional LSTM", default=1)
    parser.add_argument("--dropout_rate", type=float, metavar="N", help="Dropout rate", default=0.3)
    parser.add_argument("--layers", type=int, metavar="N", help="Number of layers", default=2)
    parser.add_argument("--dropout_attention", type=float, metavar="N", help="Dropout attention rate", default=0.5)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances.", default=None)
    parser.add_argument("--test_size", type=int, metavar="N", help="Only test on the first N instances.", default=None)
    parser.add_argument("--test_labels", type=argparse.FileType('rb'), help="Test article labels", default=None)
    parser.add_argument("-t", "--test_data", type=argparse.FileType('rb'), metavar="FILE", default=None)

    args = parser.parse_args()
    do_experiment(args)

    for fp in (args.training, args.labels, args.test_data, args.test_labels): fp.close()
