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

def main(ORIGINAL_TEXT):
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
            print(final_list[-1])
    return final_list

if __name__ == "__main__":
    sorted_sentences = main()