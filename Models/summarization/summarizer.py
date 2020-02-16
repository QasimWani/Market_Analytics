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
    
    ORIGINAL_TEXT = """It's been shown time and time again to help people avoid accidents. In fact, Tesla CEO Elon Musk said in April that Autopilot can help reduce accidents by as much as 50%.

But just like any system, it's not perfect. And it requires a human to pay attention at all times. 

On Thursday, regulators revealed an investigation into a possible tie between Tesla's Autopilot system and a fatal accident.

While few details about the collision have been revealed, Tesla has said that the car was in Autopilot mode when the car crashed.

Here's a closer look at how Autopilot works to help you better understand how it should be used.
Tesla's Autopilot system is made up of multiple sensors placed all around the car. These sensors help the car understand its environment so that it can safely steer itself in most highway situations.
The hardware that makes up Tesla's self-driving system includes a forward radar, a forward-looking camera, a high-precision digitally-controlled electric assist braking system, and 12 long-range ultrasonic sensors placed around the car. 


These ultrasonic sensors are strategically placed around the car so that they can sense 16 feet around the car in every direction, at any speed.
The senors enable the vehicle to sense when something is too close and gauge the appropriate distance so that it can do things like safely change lanes. 

However, it should be noted that these sensors can be thrown off by things like debris covering them. 

The radar enables detection of cars and other moving objects.
The forward-facing camera is located on the top windshield. A computer inside the camera helps the car understand what obstacles are ahead of the car.
The camera is basically the system's eyes. It enables the car to detect traffic, pedestrians, road signs, lane markings, and anything else that might be in front of the vehicle. This information is then used to help the car drive itself."""
    
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