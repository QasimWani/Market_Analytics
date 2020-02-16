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




def main():
    ORIGINAL_TEXT = ["In fact, Tesla CEO Elon Musk said in April that Autopilot can help reduce accidents by as much as 50%.\n\nBut just like any system, it's not perfect",
 '\n\nHowever, it should be noted that these sensors can be thrown off by things like debris covering them',
 "These sensors help the car understand its environment so that it can safely steer itself in most highway situations.\nThe hardware that makes up Tesla's self-driving system includes a forward radar, a forward-looking camera, a high-precision digitally-controlled electric assist braking system, and 12 long-range ultrasonic sensors placed around the car",
 "\n\nOn Thursday, regulators revealed an investigation into a possible tie between Tesla's Autopilot system and a fatal accident.\n\nWhile few details about the collision have been revealed, Tesla has said that the car was in Autopilot mode when the car crashed.\n\nHere's a closer look at how Autopilot works to help you better understand how it should be used.\nTesla's Autopilot system is made up of multiple sensors placed all around the car",
 '\n\n\nThese ultrasonic sensors are strategically placed around the car so that they can sense 16 feet around the car in every direction, at any speed.\nThe senors enable the vehicle to sense when something is too close and gauge the appropriate distance so that it can do things like safely change lanes',
 'And it requires a human to pay attention at all times',
 '\n\nThe radar enables detection of cars and other moving objects.\nThe forward-facing camera is located on the top windshield',
 "\n    It's been shown time and time again to help people avoid accidents"]
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