"""
Different from parser, methods in the file aims to use the ELMo embeddings, learned
from the internal state of a bi-LSTM and represent contextual features of the input
text, as described in AllenNLP's paper "Deep contextualized word representations"
"""
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.engine import Layer
from utils import do_xml_parse, extract_text


class ElmoParser:
    def __init__(self, file_name, train_size, max_len):
        self.articles = do_xml_parse(file_name, 'article', max_elements=train_size)
        self.num_articles = train_size
        if max_len != None:
            self.all_text = [extract_sentences(article)[:max_len] for article in self.articles]
        else:
            self.all_text = [extract_sentences(article) for article in self.articles]



class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)
