"""
Code adapted from Swarthmore College's CS67: Natural Language Processing
"""
from abc import ABC, abstractmethod
from collections import Counter
import nltk
from nltk.corpus import stopwords
import numpy as np
from utils import *

class HNLabels(ABC):
    def __init__(self):
        """
        Inilizes the object's labels and _label_list fields
        """
        self.labels = None
        self._label_list = None

    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    def process(self, label_file, max_instances=None):
        """
        Returns the numpy array of indices corresponding to all articles' labels
        """
        articles = do_xml_parse(label_file, 'article', max_elements=max_instances)
        y_labeled = list(map(self._extract_label, articles))

        if self.labels is None:
            self._label_list = sorted(set(y_labeled))
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)])

        y = [self.labels[x] for x in y_labeled]
        binary_matrix = np.zeros((len(y),2))
        for i,label in enumerate(y):
            binary_matrix[i][label] = 1

        return binary_matrix

    @abstractmethod
    def _extract_label(self, article):
        """ Return the label for this article """
        return "Unknown"

class BinaryLabels(HNLabels):
    def __init__(self):
        HNLabels.__init__(self)

    def _extract_label(self, article):
        return article.get('hyperpartisan')
