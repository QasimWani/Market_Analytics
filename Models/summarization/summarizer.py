### This is text summarization.
### Objective, to summarize an article and make sense off of it.

from nltk.corpus import stopwords
import nltk
import re
import collections

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
        max_word_frequency = max(word_frequencies.values())   
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

def main():
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.add("-")
    
    ORIGINAL_TEXT = """
    In this paper we consider the problem of modeling text corpora and other collections of discrete
data. The goal is to find short descriptions of the members of a collection that enable efficient
processing of large collections while preserving the essential statistical relationships that are useful
for basic tasks such as classification, novelty detection, summarization, and similarity and relevance
judgments.
Significant progress has been made on this problem by researchers in the field of information retrieval (IR) (Baeza-Yates and Ribeiro-Neto, 1999). The basic methodology proposed by
IR researchers for text corpora—a methodology successfully deployed in modern Internet search
engines—reduces each document in the corpus to a vector of real numbers, each of which represents ratios of counts. In the popular tf-idf scheme (Salton and McGill, 1983), a basic vocabulary
of “words” or “terms” is chosen, and, for each document in the corpus, a count is formed of the
number of occurrences of each word. After suitable normalization, this term frequency count is
compared to an inverse document frequency count, which measures the number of occurrences of a

c 2003 David M. Blei, Andrew Y. Ng and Michael I. Jordan.
BLEI, NG, AND JORDAN
word in the entire corpus (generally on a log scale, and again suitably normalized). The end result
is a term-by-document matrix X whose columns contain the tf-idf values for each of the documents
in the corpus. Thus the tf-idf scheme reduces documents of arbitrary length to fixed-length lists of
numbers.
While the tf-idf reduction has some appealing features—notably in its basic identification of sets
of words that are discriminative for documents in the collection—the approach also provides a relatively small amount of reduction in description length and reveals little in the way of inter- or intradocument statistical structure. To address these shortcomings, IR researchers have proposed several
other dimensionality reduction techniques, most notably latent semantic indexing (LSI) (Deerwester
et al., 1990). LSI uses a singular value decomposition of the X matrix to identify a linear subspace
in the space of tf-idf features that captures most of the variance in the collection. This approach can
achieve significant compression in large collections. Furthermore, Deerwester et al. argue that the
derived features of LSI, which are linear combinations of the original tf-idf features, can capture
some aspects of basic linguistic notions such as synonymy and polysemy.
To substantiate the claims regarding LSI, and to study its relative strengths and weaknesses, it is
useful to develop a generative probabilistic model of text corpora and to study the ability of LSI to
recover aspects of the generative model from data (Papadimitriou et al., 1998). Given a generative
model of text, however, it is not clear why one should adopt the LSI methodology—one can attempt
to proceed more directly, fitting the model to data using maximum likelihood or Bayesian methods.
A significant step forward in this regard was made by Hofmann (1999), who presented the
probabilistic LSI (pLSI) model, also known as the aspect model, as an alternative to LSI. The pLSI
approach, which we describe in detail in Section 4.3, models each word in a document as a sample
from a mixture model, where the mixture components are multinomial random variables that can be
viewed as representations of “topics.” Thus each word is generated from a single topic, and different
words in a document may be generated from different topics. Each document is represented as
a list of mixing proportions for these mixture components and thereby reduced to a probability
distribution on a fixed set of topics. This distribution is the “reduced description” associated with
the document.
    """
    
    TESLA_TEXT = ORIGINAL_TEXT.lower().replace(". ", " qwertyuiop")
    TESLA_TEXT = re.sub('[^a-zA-Z]', ' ', TESLA_TEXT )
    TESLA_TEXT = re.sub(r'\s+', ' ', TESLA_TEXT)
    TESLA_TEXT = TESLA_TEXT.split(" qwertyuiop")

    average_sentence_word_count = len(TESLA_TEXT)
    sum_word_count = 0
    for c,text in enumerate(TESLA_TEXT):
        TESLA_TEXT[c] = ' '.join([word for word in text.split() if word not in STOPWORDS])
        sum_word_count += len(TESLA_TEXT[c].split(" "))

    average_sentence_word_count = sum_word_count / average_sentence_word_count
    
    sentence_scores = get_text_weighted_score(TESLA_TEXT, average_sentence_word_count)
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

if __name__ == "__main__":
    sorted_sentences = main()