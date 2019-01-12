import sys
from itertools import islice
from html import unescape
from scipy import sparse
from lxml import etree

def extract_text(article):
    return unescape("".join([x for x in article.find("spacy").itertext()])).split()

def extract_sentences(article):
    return unescape("".join([x for x in article.find("spacy").itertext()]))

def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """
    Parses cleaned up spacy-processed XML files. This functions is more
    memory efficient (see comments on yield), and more CPU-efficient (see
    comments on islice)
    """
    fp.seek(0)
    # islice slices the elements from a list till stop point (max_elements),
    # or all elements if stop is not specified. The for-loop belows only iterates
    # through max_elements instead of the entire list.
    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
    for i, (event, elem) in elements:
        # Returns a generator containing chunks, allowing less memory waste
        # because we can access one element at a time, instead of keeping
        # all elements in memory at once
        yield elem
        # clear() empties the given list without returning the value
        elem.clear()
        if progress_message and (i % 1000 == 0):
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)
