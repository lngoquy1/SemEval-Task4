from HyperpartisanNewsReader import do_xml_parse, ParserVocab
import argparse
from html import unescape
from dateutil.parser import parse
import re
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
import multiprocessing
from multiprocessing import Pool, freeze_support
from itertools import repeat


class Parser:
    """
    Each parser returns a word embedding for an article
    """

    def __init__(self, file_name, train_size, max_len):
        self.X = []
        self.y = []
        self.all_feats = None
        self.vocab = {}
        self.num_banned = []
        self.corpus = []
        self.hyperlinks = []
        self.all_hyperlinks = {}
        self.title_features = []
        self.bow = []
        self.MAX_LEN = max_len
        self.MAX_NUM_LINKS = 3
        self.articles = do_xml_parse(file_name, 'article', max_elements=train_size)
        self.feature_dict = {}
        self.num_articles = train_size
        self.censored = open("banned_by_google.txt","r").read().splitlines()





    def uncensor_word(self, word, num_banned):
        numLettersFromStart = word.find("*")
        numLettersFromEnd = len(word) - word.rfind("*") - 1

        # with open("banned_by_google.txt","r") as censor_file:
        #     censored = censor_file.read().splitlines()

        if word in self.censored:
            num_banned += 1

        for item in self.censored:
            if numLettersFromEnd != 0:
                censored_regex = r"^" + re.escape(item[0:numLettersFromStart]) + r"\*+" + re.escape(item[-1*numLettersFromEnd:]) + r"$"
            else:
                censored_regex = r"^" + re.escape(item[0:numLettersFromStart]) + r"\*+$"
            # print(censored_regex)
            if re.match(censored_regex, word):
                num_banned += 1
                return item, num_banned
        return word, num_banned


    def add_feature_dict(self, values):
        """
        Having self.vocab, self.all_hyperlinks, self.title_features and self.num_banned,
        we want to be able to make a dictionary of index mapping with which vocab
        """
        for value in values:
            key = len(self.feature_dict)
            self.feature_dict[key] = value



    def map_features_parallel(self, vocab=None):
        num_cpu = multiprocessing.cpu_count()
        chunk_size = self.num_articles//num_cpu
        all_text = []
        small_chunk = []
        for article in self.articles:
            if len(small_chunk) == chunk_size:
                all_text.append(small_chunk)
                small_chunk = []
            small_chunk.append(extract_text(article))
            # all_links.append(self.get_hyperlink_feat(article))
        all_text.append(small_chunk)


        parsed_articles = Pool(num_cpu).starmap(feat_processing_wrapper, zip(all_text, repeat(self.MAX_LEN)))
        if vocab == None:
            word_freq = Counter([w for articles in parsed_articles for article in articles for w in article]).most_common()
            self.vocab = {word_freq[i][0] : i+1 for i in range(len(word_freq))}
        else:
            self.vocab = vocab
        mapped_articles = Pool(num_cpu).starmap(map_word_index_wrapper, zip(parsed_articles, repeat(self.vocab), repeat(self.MAX_LEN)))
        # Flatten the chunks out
        flattened = [article for chunk in mapped_articles for article in chunk]
        self.X = np.array(flattened)




    def map_general_features(self, vocab=None, hyperlinks=None):
        num_cpu = multiprocessing.cpu_count()
        chunk_size = self.num_articles//num_cpu
        all_text = []
        all_hyperlinks = []
        all_title_features = []
        small_chunk = []
        for article in self.articles:
            if len(small_chunk) == chunk_size:
                all_text.append(small_chunk)
                small_chunk = []
            small_chunk.append(extract_text(article))
            all_hyperlinks.append(get_hyperlinks(article))
            all_title_features.append(get_title_features(article))
        all_text.append(small_chunk)

        # Get bow
        parsed_articles = Pool(num_cpu).starmap(feat_processing_wrapper, zip(all_text, repeat(self.MAX_LEN)))
        if vocab == None:
            self.add_feature_dict(["zero_padding"])
            word_freq = Counter([w for articles in parsed_articles for article in articles for w in article]).most_common()
            sorted_words = [word for word,_ in word_freq]
            self.vocab = {sorted_words[i] : i+1 for i in range(len(sorted_words))}
            self.add_feature_dict(sorted_words)
        else:
            self.vocab = vocab

        all_bow = Pool(num_cpu).starmap(map_bow_features_wrapper, zip(parsed_articles, repeat(self.vocab), repeat(len(self.vocab)+1)))
        flattened_bow_features = np.array([bow_vecs for chunk in all_bow for bow_vecs in chunk])



        if hyperlinks == None:
            link_freq = Counter([link for three_links in all_hyperlinks for link in three_links]).most_common()
            sorted_links = [link for link,_ in link_freq]
            self.hyperlinks = {sorted_links[i] : i for i in range(len(sorted_links))}
            self.add_feature_dict(sorted_links)
        else:
            self.hyperlinks = hyperlinks
        hyperlink_features = map_bow_features_wrapper(all_hyperlinks, self.hyperlinks, len(self.hyperlinks))

        # title_features
        self.add_feature_dict(["title_length", "all_caps", "is_question", "is_exclamation", "stop_words"])

        title_features = np.array(all_title_features)

        print("flattened_bow_features", flattened_bow_features.shape)
        print("hyperlink_features", hyperlink_features.shape)
        print("title_features", title_features.shape)
        self.all_feats = np.concatenate([flattened_bow_features, hyperlink_features, title_features], axis=1)

        print("Shape of all_feats", self.all_feats.shape)







########################### HELPER FUNCTIONS ###########################
def extract_text(article):
    return unescape("".join([x for x in article.find("spacy").itertext()])).split()

def extract_tag(self, article):
    return unescape("".join([x for x in article.find("tag").itertext()])).split()

def get_title_features(article):
    title = article.get("title").split()
    if len(title) == 0:
        return [0]*5
    # get title length
    title_length = len(title)//5
    # get num allcaps
    all_caps = round(sum([1 for word in title if word.isupper()])/len(title),4)
    # get if question mark
    is_question = 1 if "?" in title else 0
    # get if question mark
    is_exclamation = 1 if "!" in title else 0
    # get num stop words
    stop_words = round(sum([1 for w in title if w in set(stopwords.words('english'))])/len(title),4)
    return np.array([title_length, all_caps, is_question, is_exclamation, stop_words])

def get_bow(words, vocabulary_size):
    bow_vec = np.zeros(vocabulary_size) # for padding 0
    cnt = Counter(words).most_common()
    for word_index, count in cnt:
        bow_vec[word_index] = count
    return bow_vec

def chop_link(link):
    index = 0
    for i in range(3):
        index = link.find('/', index)+1
    return link[:index]

def feat_processing_wrapper(parsed_articles, max_len):
    return [parse_article(article, max_len) for article in parsed_articles]

def parse_article(text, max_len):
    text = [w for w in text if w.isalpha() and w not in set(stopwords.words('english'))]
    parsed_text = []
    i = 0
    while i < len(text) and len(parsed_text) <= max_len:
        if i < len(text)-1 and text[i] == '#' and text[i+1].isdigit() != True:
            parsed_text += ['<hashtag>', text[i+1].lower(), '</hashtag>']
            i += 2
        elif text[i].isupper():
            parsed_text += [text[i], '<allcaps>']
            i += 1
        elif i < len(text)-1 and is_date(' '.join([text[i], text[i+1]])):
            parsed_text.append('<date>')
            i += 2
        elif i < len(text)-2 and is_date(' '.join([text[i], text[i+1], text[i+2]])):
            parsed_text.append('<date>')
            i += 3
        else:
            parsed_text.append(text[i].lower())
            i += 1
    parsed_text = parsed_text[:50]
    return parsed_text


def is_date(string):
    # Code from https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    try:
        parse(string)
        return True
    except:
        return False


def map_word_index_wrapper(articles, word_index, max_len):
    mapped = [[word_index[w] if w in word_index else 0 for w in article] for article in articles]
    padded_mapped = [[0]*(max_len - len(xi)) + xi for xi in mapped]
    return padded_mapped


def get_hyperlinks(article):
    hyperlinks = article.find('a')
    if hyperlinks is not None:
        hyperlinks = hyperlinks.xpath("//@href")
        shortened = [chop_link(link) for link in hyperlinks if len(link) > 7]
        top_links = [link for link,_ in Counter(shortened).most_common(3)]
        for i in range(3-len(top_links)):
            top_links.append("<empty_link>")
        return top_links
    else:
        return ["<empty_link>"]*3

def map_bow_features_wrapper(articles, vocab, vocabulary_size):
    mapped =  [[vocab[w] if w in vocab else 0 for w in article] for article in articles]
    bow_vecs = [get_bow(xi, vocabulary_size) for xi in mapped]
    return np.array(bow_vecs)
# def do_experiment(args):
#     vocab = HNVocab(args.vocabulary, args.vocab_size, args.stop_words)
#     parser = Parser(vocab)
#     parser.process(args.training, args.train_size)
#     print(parser.X)








# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("training", type=argparse.FileType('rb'), help="Training articles")
#     # parser.add_argument("labels", type=argparse.FileType('rb'), help="Training article labels")
#     parser.add_argument("vocabulary", type=argparse.FileType('r'), help="Vocabulary")
#     # parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
#     parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
#     parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file (after stop words)", default=None)
#     parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances.", default=None)
#     # parser.add_argument("--test_size", type=int, metavar="N", help="Only test on the first N instances.", default=None)
#
#     # eval_group = parser.add_mutually_exclusive_group(required=True)
#     # eval_group.add_argument("-t", "--test_data", type=argparse.FileType('rb'), metavar="FILE")
#     # eval_group.add_argument("-x", "--xvalidate", type=int)
#
#     args = parser.parse_args()
#     do_experiment(args)
