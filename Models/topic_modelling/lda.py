### Running LDA analysis on summarized (condensed) text generated from articles.
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import re
import numpy as np
np.random.seed(1729)
import nltk
import pandas as pd
from nltk.corpus import stopwords



def pre_process_text(ORIGINAL_TEXT):
    """Polishes text"""
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.add("-")
    frp = []
    for i, c in enumerate(ORIGINAL_TEXT):
        reg = c.lower()
        reg = ' '.join(reg)
        reg = ' '.join([word for word in c.split() if word not in STOPWORDS])
        reg = re.sub('[^a-zA-Z]', ' ', reg)
        reg = re.sub(r'\s+', ' ', reg)
        frp.append(reg)
    return frp


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return result


def generate_frequencies(sentence):
    """
    Generates Frequencies and Occurances of words from a sentence:
    1. Parameters : sentence (lemmantized version of the sentence, in list type).
    2. Returns : 
        A. TDM (pandas Series object)
        B. DataFrame (pandas TDM representation)
    """
    occurance = {}
    frequency = {}
    for i, word in enumerate(sentence):
        if word not in frequency.keys():
            frequency[word] = 1
        else:
            frequency[word] += 1
    max_word_frequency = max(frequency.values())
    for word in frequency.keys():
        occurance[word] = frequency[word] / max_word_frequency
    df = pd.DataFrame(data=[list(frequency.keys()), list(frequency.values()), list(occurance.values())]).T
    df.columns = ['Word', 'Occurance', 'Frequency']
    return df, frequency, occurance

def generate_tf_idf(paragraph, ORIGINAL_TEXT):
    tf_idf = []
    for i, lem in enumerate(paragraph):
        pd_df, frequency_words, occurance_words = generate_frequencies(lem)
        temp = []
        for occur, freq in zip(frequency_words.values(), occurance_words.values()):
            temp.append(freq * np.log10(len(ORIGINAL_TEXT)/occur))
        tf_idf.append(temp)
    return tf_idf 




def main(ORIGINAL_TEXT):
    POLISHED_TEXT = pre_process_text(ORIGINAL_TEXT)
    #Incorporating stemming instead of lemmatization because of performance and speed.
    stemmer = SnowballStemmer('english')

    lemmantized = list(pd.Series(ORIGINAL_TEXT).map(preprocess))

    dictionary = gensim.corpora.Dictionary(lemmantized)
    dictionary.filter_tokens()

    ### TD-IDF generation

    tf_idf = generate_tf_idf(lemmantized, ORIGINAL_TEXT)


    # Create a corpus from a list of texts
    common_dictionary = Dictionary(common_texts)
    lemmy_BOW = [dictionary.doc2bow(text) for text in lemmantized]

    lda_model = gensim.models.LdaMulticore(lemmy_BOW, minimum_probability=15, num_topics=10, id2word=dictionary, passes=3, workers=3)


    topics = []
    for idx, topic in lda_model.print_topics(-1):
        topics.append(topic)
        print(topic)
    return topics

if __name__ == "__main__":
    LDA_analysis = main()