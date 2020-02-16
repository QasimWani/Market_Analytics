### Part 1. Text Summarization

### This is text summarization.
### Objective, to summarize an article and make sense off of it.
from bs4 import BeautifulSoup
import sys
import requests
from nltk.corpus import stopwords
import nltk
import re
import collections
import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
np.random.seed(1729)

def calculate_sentence_frequency(sentence, average_sentence_word_count):
    """
    Calculates the weighted frequency of a single sentence.
    Parameters:
    1. sentence. A string containing multiple words.
    Returns : word_frequencies (type = dict) list of words and associative weights.
    """
    word_frequencies = {}
    if len(sentence.split(" ")) < average_sentence_word_count:
        for word in nltk.word_tokenize(sentence):
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        max_word_frequency = max(word_frequencies.values()) if len(word_frequencies.values()) > 0 else 1
        for word in word_frequencies.keys():
            word_frequencies[word] /= max_word_frequency
    return word_frequencies

def get_text_weighted_score(paragraph, average_word_count):
    """
    Generates the weighted score of the entire text.
    Uses calculate_sentence_frequency(paragraph[i]).
    Parameters:
    1. paragraph. A list of sentences.
    Returns:
    1. sentence_scores (type = dict) list of sentence and associative weights.
    """
    sentence_scores = {}
    for i, sent in enumerate(paragraph):
        word_frequencies = calculate_sentence_frequency(paragraph[i], average_word_count)
        for word in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] =  word_frequencies[word]
            else:
                sentence_scores[sent] += word_frequencies[word]
    return sentence_scores

def text_summarization_main(ORIGINAL_TEXT):
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.add("-")
    
    intermedia_text = ORIGINAL_TEXT.lower().replace(". ", " qwertyuiop")
    intermedia_text = re.sub('[^a-zA-Z]', ' ', intermedia_text )
    intermedia_text = re.sub(r'\s+', ' ', intermedia_text)
    intermedia_text = intermedia_text.split(" qwertyuiop")

    average_sentence_word_count = len(intermedia_text)
    sum_word_count = 0
    for c,text in enumerate(intermedia_text):
        intermedia_text[c] = ' '.join([word for word in text.split() if word not in STOPWORDS])
        sum_word_count += len(intermedia_text[c].split(" "))

    average_sentence_word_count = sum_word_count / average_sentence_word_count
    
    sentence_scores = get_text_weighted_score(intermedia_text, average_sentence_word_count)
    original_dict = {}
    ORIGINAL_TEXT = ORIGINAL_TEXT.split(". ")
    for i, sentences in enumerate(sentence_scores.items()):
        original_dict[ORIGINAL_TEXT[i]] = sentences[1]
    sorted_sentences = sorted(original_dict.items(), key=lambda x: x[1], reverse=True)
    final_list = []
    for i, s in enumerate(sorted_sentences):
        if i < 10:
            final_list.append(s[0])
    return final_list

### Part 2. LDA Generation.

### Running LDA analysis on summarized (condensed) text generated from articles.
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




def latent_dirichlet_allocation(summarized_text):
    ORIGINAL_TEXT = summarized_text
    POLISHED_TEXT = pre_process_text(ORIGINAL_TEXT)
    #Incorporating stemming instead of lemmatization because of performance and speed.
    stemmer = SnowballStemmer('english')

    lemmantized = list(pd.Series(ORIGINAL_TEXT).map(preprocess))

    dictionary = gensim.corpora.Dictionary(lemmantized)
    dictionary.filter_tokens()

    ### TD-IDF generation

    tf_idf = generate_tf_idf(lemmantized, ORIGINAL_TEXT)


    # Create a corpus from a list of texts
    lemmy_BOW = [dictionary.doc2bow(text) for text in lemmantized]

    lda_model = gensim.models.LdaMulticore(lemmy_BOW, minimum_probability=15, num_topics=10, id2word=dictionary, passes=3, workers=3)


    topics = []
    for idx, topic in lda_model.print_topics(-1):
        topics.append(topic)
    return topics



### Part 3. Search Algorithm Implementation.

def search_Parent_BING(QUERY):
    ### First subroutine : 1.News Lead Generation...
    ### Important : Only works for Bing as Root node.

    link = "https://www.bing.com/news/search?q="+ QUERY.lower() +"&FORM=HDRSC6"
    html_data = requests.get(link).content
    soup = BeautifulSoup(html_data, 'html.parser')

    TITLE = str(soup.title).split("<title>")[1].split("-")[0].split("|")[0].strip()

    title_pages, links_pages = getLinkTitle(soup)
    children_text = tree_ChildrenText(links_pages) #uses first subrotine to extrapolate data from children.

    summaries = mult_text_sum_impl(children_text) ### uses second subrotine to summarize text

    LDA_mult = mult_LDA(summaries) ### third subroutine done to implement Dirichlet.

    return children_text, summaries, LDA_mult, title_pages, links_pages ### last two are for references (meta)

def getLinkTitle(soup):
    """
    This function takes in the query to find the list of all anchor tags and associative links.
    Parameters:
    1. soup object.
    Returns:
    1. List of anchor tag.
    2. List of website titles.
    """
    arefs = list(soup.find_all("a", class_="title"))
    title_pages = [str(x).split(">")[1].split("</")[0].replace(" › ", "/") for x in arefs]
    links_pages = [str(x).split("href=\"")[1].split("\"")[0].replace(" › ", "/") for x in arefs]
    return title_pages[:3], links_pages[:3]

def get_NODE_ChildrenText(page_link):
    
    child = requests.get(page_link).content
    soup_child = BeautifulSoup(child, 'html.parser')
    site_text = ""
    if str(soup_child.text).find("ERROR: The request could not be satisfied") == -1:
        for i, txt in enumerate(soup_child.find_all("p")):
            if(str(txt).find("|") == -1 or str(txt).find("FREE") == -1):
                if(str(txt).find("class=") != -1 or str(txt).find("<span>") != -1 or str(txt).find("style=") != -1 or str(txt).find("content=") != -1):
                    site_text += txt.text
    return site_text

def tree_ChildrenText(multiple_pages):
    page_mult_text = []
    for i, links in enumerate(multiple_pages):
        page_mult_text.append(get_NODE_ChildrenText(links))
    return page_mult_text

def mult_text_sum_impl(arrText):
    summaries = []
    for i, data in enumerate(arrText):
        summaries.append(text_summarization_main(data))
    return summaries

def mult_LDA(arrSum):
    LDAs = []
    for i, data in enumerate(arrSum):
        LDAs.append(latent_dirichlet_allocation(data))
    return LDAs


"""
Main Implementation. 
Ensemble
"""
def main(QUERY):
    _, summaries, LDA_mult, title_pages, links_pages = search_Parent_BING(QUERY)
    print(summaries)
    print("\n\n\n")
    print(LDA_mult)
    print("\n\n\n")
    print(title_pages)
    print("\n\n\n")
    print(links_pages)

if __name__ == "__main__":
    main(sys.argv[1:])
