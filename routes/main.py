### Part 1. Text Summarization

### This is text summarization.
### Objective, to summarize an article and make sense off of it.
import json
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
    ORIGINAL_TEXT = str(ORIGINAL_TEXT)
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
TEXT_1 = """
Arizona enjoyed record numbers of travelers and travel-related spending in 2018, state officials announced Tuesday at the Governor’s Conference on Tourism in Tucson.

Travelers to Arizona spent $24.4 billion in 2018 — $67 million per day and 8% more than the previous record set in 2017, the Office of Tourism said.

Overall, Arizona counted a record-breaking 45.4 million overnight visitors last year, eclipsing the 43.9 million travelers tracked the previous year, according to research performed for the state by Dean Runyan Associates, Longwoods International and Tourism Economics.

"That’s like importing the entire populations of Texas, Colorado, New Mexico, Nevada, Utah, Montana and Wyoming into our state over the course of a year,” said Debbie Johnson, director of the Arizona Office of Tourism.

"Obviously, that has a tremendously positive impact on local economies across Arizona.”

State tax revenues generated by visitor spending, including hotel bed taxes, rental-car taxes and sales taxes, reached the $1 billion mark for the first time, the tourism office said.

City and county tax revenues generated by tourism increased 17% over the previous fiscal year, to a record total of $1.1 billion.

That tax revenue lessened the tax burden of the average Arizona household by $1,360, Johnson said.

Other findings of the 2018 tourism report, which will be posted online at tourism.az.gov, include:

Visitor spending directly supported 192,300 jobs in the state in 2018. Those jobs accounted for $7.4 billion in earnings by Arizona workers.
Arizona had 5.8 million international overnight visitors, up 4% from 2017.
Visitor volume increased from both of the state’s top international markets, Mexico (3.7 million overnight visitors) and Canada (962,000 overnight visitors).
Germany was Arizona’s top source of European visitors.
After eight years of strong growth, visitor volume from China declined nearly 4%.
Among international markets, the biggest year-over-year growth in visitor volume came from Australia, which was up 15%.
California remained Arizona’s top source of domestic visitors, but Texas had the biggest year-over-year growth in visitors with an increase of 24%.
"""

TEXT_2 = """PHOENIX, AZ — According to the Arizona Office of Tourism's 2019 industry performance report, covering the year 2018, Arizona tourism raked in $24.4 billion last year, a 7.8 percent increase from 2017. Maricopa County visitor spending comprised nearly half the $24.4 billion total, at $12 billion in tourism revenue, while Pima County drew in $2.5 billion.

The growth in total tourism revenue fueled increased federal, state and local tax revenue for the year, rising a combined 9.6 percent to $3.63 billion. The tourism boom also meant a rise in earnings for Arizona tourism employees to $7.4 billion, up 6.2 percent annually. Related to that rise was the large number of tourism industry-related jobs created (192,000 for the state, up about 2 percent), with Maricopa County (107,500) and Pima County (nearly 25,000) exponentially leading all other Arizona counties.

Arizona had 45.4 million combined international and domestic visitors in 2018, a 3.3 percent increase.

Top origin countries for 2018 international overnight visitors to Arizona:

Mexico (3.8 million)
Canada (962,000)
Germany (132,800)
United Kingdom (121,700)
France (99,800)
China (78,300)
Top origin states for 2018 domestic overnight visitors to Arizona:

Arizona (11 million)
California (7.5 million)
Texas (2.1 million)
Illinois (1.6 million)
Colorado (1.4 million)
Florida (1.4 million)"""

RIDESHARING_ONE_FULL_TEXT_3 = """Carpooling (also car-sharing, ride-sharing and lift-sharing) is the sharing of car journeys so that more than one person travels in a car, and prevents the need for others to have to drive to a location themselves.

The government is helping the adopting of carpooling services as these are considered crucial to make the transportation sector environmentally friendly.

In 2017, the global Carpooling market size was xx million US$ and it is expected to reach xx million US$ by the end of 2025, with a CAGR of xx% during 2018-2025.

Access the PDF sample of the report @ https://www.orbisresearch.com/contacts/request-sample/2377368

This report focuses on the global Carpooling status, future forecast, growth opportunity, key market and key players. The study objectives are to present the Carpooling development in United States, Europe and China.

The key players covered in this study

Uber

Orix

Lyft

Carma

BlaBlaBla

Relay Rides

Sidecar

Ridejoy

Getaround

JustShareIt

Zimride

Car2Go

Shared EV Fleets

Ekar

Didi Chuxing

Market segment by Type, the product can be split into

Commuter Carpool

Holiday Long-Distance Carpool

Tourism Carpool

Market segment by Application, split into

Public Websites

Social Media

Acting as Marketplaces

Employer Websites

Smartphone Applications

Carpooling Agencies

Pick-Up Points

Market segment by Regions/Countries, this report covers

United States

Europe

China

Japan

Southeast Asia

India

Central & South America

The study objectives of this report are:

To analyze global Carpooling status, future forecast, growth opportunity, key market and key players.

To present the Carpooling development in United States, Europe and China.

To strategically profile the key players and comprehensively analyze their development plan and strategies.

To define, describe and forecast the market by product type, market and key regions.

In this study, the years considered to estimate the market size of Carpooling are as follows:

History Year: 2013-2017

Base Year: 2017

Estimated Year: 2018

Forecast Year 2018 to 2025

For the data information by region, company, type and application, 2017 is considered as the base year. Whenever data information was unavailable for the base year, the prior year has been considered.

BROWSE THE FULL REPORT @ HTTPS://WWW.ORBISRESEARCH.COM/REPORTS/INDEX/GLOBAL-CARPOOLING-MARKET-SIZE-STATUS-AND-FORECAST-2018-2025
Table of Contents

Chapter One: Report Overview

1.1 Study Scope

1.2 Key Market Segments

1.3 Players Covered

1.4 Market Analysis by Type

1.4.1 Global Carpooling Market Size Growth Rate by Type (2013-2025)

1.4.2 Commuter Carpool

1.4.3 Holiday Long-Distance Carpool

1.4.4 Tourism Carpool

1.5 Market by Application

1.5.1 Global Carpooling Market Share by Application (2013-2025)

1.5.2 Public Websites

1.5.3 Social Media

1.5.4 Acting as Marketplaces

1.5.5 Employer Websites

1.5.6 Smartphone Applications

1.5.7 Carpooling Agencies

1.5.8 Pick-Up Points

1.6 Study Objectives

1.7 Years Considered

Chapter Two: Global Growth Trends

2.1 Carpooling Market Size

2.2 Carpooling Growth Trends by Regions

2.2.1 Carpooling Market Size by Regions (2013-2025)

2.2.2 Carpooling Market Share by Regions (2013-2018)

2.3 Industry Trends

2.3.1 Market Top Trends

2.3.2 Market Drivers

2.3.3 Market Opportunities

Chapter Three: Market Share by Key Players

3.1 Carpooling Market Size by Manufacturers

3.1.1 Global Carpooling Revenue by Manufacturers (2013-2018)

3.1.2 Global Carpooling Revenue Market Share by Manufacturers (2013-2018)

3.1.3 Global Carpooling Market Concentration Ratio (CRChapter Five: and HHI)

3.2 Carpooling Key Players Head office and Area Served

3.3 Key Players Carpooling Product/Solution/Service

3.4 Date of Enter into Carpooling Market

3.5 Mergers & Acquisitions, Expansion Plans

Chapter Four: Breakdown Data by Type and Application

4.1 Global Carpooling Market Size by Type (2013-2018)

4.2 Global Carpooling Market Size by Application (2013-2018)

Chapter Five: United States

5.1 United States Carpooling Market Size (2013-2018)

5.2 Carpooling Key Players in United States

5.3 United States Carpooling Market Size by Type

5.4 United States Carpooling Market Size by Application

Chapter Six: Europe

6.1 Europe Carpooling Market Size (2013-2018)

6.2 Carpooling Key Players in Europe

6.3 Europe Carpooling Market Size by Type

6.4 Europe Carpooling Market Size by Application

Chapter Seven: China

7.1 China Carpooling Market Size (2013-2018)

7.2 Carpooling Key Players in China

7.3 China Carpooling Market Size by Type

7.4 China Carpooling Market Size by Application

Chapter Eight: Japan

8.1 Japan Carpooling Market Size (2013-2018)

8.2 Carpooling Key Players in Japan

8.3 Japan Carpooling Market Size by Type

8.4 Japan Carpooling Market Size by Application

Chapter Nine: Southeast Asia

9.1 Southeast Asia Carpooling Market Size (2013-2018)

9.2 Carpooling Key Players in Southeast Asia

9.3 Southeast Asia Carpooling Market Size by Type

9.4 Southeast Asia Carpooling Market Size by Application

Chapter Ten: India

10.1 India Carpooling Market Size (2013-2018)

10.2 Carpooling Key Players in India

10.3 India Carpooling Market Size by Type

10.4 India Carpooling Market Size by Application

Chapter Eleven: Central & South America

11.1 Central & South America Carpooling Market Size (2013-2018)

11.2 Carpooling Key Players in Central & South America

11.3 Central & South America Carpooling Market Size by Type

11.4 Central & South America Carpooling Market Size by Application

Chapter Twelve: International Players Profiles

12.1 Uber

12.1.1 Uber Company Details

12.1.2 Company Description and Business Overview

12.1.3 Carpooling Introduction

12.1.4 Uber Revenue in Carpooling Business (2013-2018)

12.1.5 Uber Recent Development

12.2 Orix

12.2.1 Orix Company Details

12.2.2 Company Description and Business Overview

12.2.3 Carpooling Introduction

12.2.4 Orix Revenue in Carpooling Business (2013-2018)

12.2.5 Orix Recent Development

12.3 Lyft

12.3.1 Lyft Company Details

12.3.2 Company Description and Business Overview

12.3.3 Carpooling Introduction

12.3.4 Lyft Revenue in Carpooling Business (2013-2018)

12.3.5 Lyft Recent Development

12.4 Carma

12.4.1 Carma Company Details

12.4.2 Company Description and Business Overview

12.4.3 Carpooling Introduction

12.4.4 Carma Revenue in Carpooling Business (2013-2018)

12.4.5 Carma Recent Development

12.5 BlaBlaBla

12.5.1 BlaBlaBla Company Details

12.5.2 Company Description and Business Overview

12.5.3 Carpooling Introduction

12.5.4 BlaBlaBla Revenue in Carpooling Business (2013-2018)

12.5.5 BlaBlaBla Recent Development

12.6 Relay Rides

12.6.1 Relay Rides Company Details

12.6.2 Company Description and Business Overview

12.6.3 Carpooling Introduction

12.6.4 Relay Rides Revenue in Carpooling Business (2013-2018)

12.6.5 Relay Rides Recent Development

12.7 Sidecar

12.7.1 Sidecar Company Details

12.7.2 Company Description and Business Overview

12.7.3 Carpooling Introduction

12.7.4 Sidecar Revenue in Carpooling Business (2013-2018)

12.7.5 Sidecar Recent Development

12.8 Ridejoy

12.8.1 Ridejoy Company Details

12.8.2 Company Description and Business Overview

12.8.3 Carpooling Introduction

12.8.4 Ridejoy Revenue in Carpooling Business (2013-2018)

12.8.5 Ridejoy Recent Development

12.9 Getaround

12.9.1 Getaround Company Details

12.9.2 Company Description and Business Overview

12.9.3 Carpooling Introduction

12.9.4 Getaround Revenue in Carpooling Business (2013-2018)

12.9.5 Getaround Recent Development

12.10 JustShareIt

12.10.1 JustShareIt Company Details

12.10.2 Company Description and Business Overview

12.10.3 Carpooling Introduction

12.10.4 JustShareIt Revenue in Carpooling Business (2013-2018)

12.10.5 JustShareIt Recent Development

12.11 Zimride

12.12 Car2Go

12.13 Shared EV Fleets

12.14 Ekar

12.15 Didi Chuxing

Chapter Thirteen: Market Forecast 2018-2025

13.1 Market Size Forecast by Regions

13.2 United States

13.3 Europe

13.4 China

13.5 Japan

13.6 Southeast Asia

13.7 India

13.8 Central & South America"""

BlaBla_Text4 = """French startup BlaBlaCar has announced that the company’s revenue grew by 71 percent in 2019 compared to 2018. The big difference between 2019 and 2018 is that BlaBlaCar diversified its activity by offering bus rides as well as bus ticketing in some markets.

BlaBlaCar is still mostly known for its long-distance ride-sharing marketplace. If you’re going from one city to another, you can find a car with an empty seat and book a ride in that car. On the other side of the marketplace, if you plan on driving across the country, you can list your ride on the platform to find passengers so that you don’t have to pay for gas and highway tolls by yourself.

In November 2018, the company acquired Ouibus to become a marketplace for road travel, whether it’s by bus or by car. Ouibus is now called BlaBlaBus. BlaBlaCar also offers a carpooling marketplace for daily commutes between your home and your workplace called BlaBlaLines.

BlaBlaBus covers 400 cities in Europe while BlaBlaLines has managed to attract 1.5 million users.

The bottom line is that BlaBlaCar has built a huge community. The company now has 87 million users, with 17 million people signing up in 2019 alone. BlaBlaCar carried 70 million passengers across all its services last year.

In France, the long-distance carpooling service reached a record of 135,000 passengers in a single day. I’d bet that the railway company strike may have helped.

When it comes to the company itself, BlaBlaCar has hired a Chief Operating Officer, Béatrice Dumurgier. While BlaBlaCar faced some growing pains a couple of years ago, the company now plans to expand its team again by doubling the size of its engineering team in 2020."""

HH_title = """HitchHiqe: A Hokie’s solution to Virginia Tech’s inefficient carpooling Facebook group
Carey Oakes, lifestyles staff writer Feb 9, 2020"""

HH_text = """After tirelessly scrolling through Virginia Tech’s carpooling Facebook group in search of a ride home, Qasim Wani, a sophomore majoring in computer engineering, realized just how inefficient this 20-30 minute process truly was. With experience designing his previous apps and extensive help from the APEX Entrepreneurship Center, Wani decided to develop an app, HitchHiqe, to aid students’ search for long distance carpooling rides to and from Blacksburg.  

Production and app development began in May of 2019. Wani spent around 1,500 hours working on HitchHiqe in which he said, "There were a lot of challenges when you’re building something from just an idea on a white board … and sometimes it feels like it doesn’t work.  Sometimes I feel like I’m too young to just do it all by myself, but you find a way out.” 

Wani, however, hasn’t spent this entire process completely alone. He received a Facebook notification from fellow Hokie Sai Gurrapu early on in HitchHiqe’s production in which Gurrapu expressed interest in helping Wani develop the app. He can also attribute some of his success to Virginia Tech’s APEX Center and its help with funding and marketing. Wani first came into contact with APEX at its kickstarter event in 2018 in which he struck up a conversation with a presenter about the possibilities of virtual reality. It was here that Wani started on the path of app development that eventually guided him toward his most recent app, HitchHiqe.  

The app is designed to sort through the most relevant carpooling offers on Facebook and collect them on one platform so that Virginia Tech students can find rides home without having to go through every single Facebook post and message multiple people. The app saves students around 20 to 30 minutes in the process. Wani built the beta version of HitchHiqe in just three weeks. They then sent the beta version to Facebook and other companies to receive feedback regarding the app.

They eventually equipped this efficient system with precautions designed to ensure the safety of Virginia Tech student drivers and passengers. "The algorithms include screening for drivers so not anyone can be a driver for HitchHiqe unless they have a registered vehicle,” Wani said. "In order to register in HitchHiqe, you need to have an ‘edu’ account (as well).” 

There are also safety concerns about the Facebook group that the creation of the app hopes to address.
"We actually got an email from one of the moderators from the VT parent’s group who said Facebook isn’t very secure for our kids, and we would like to endorse you (on our) platform,” Wani said, explaining the positive results of the app’s safety precautions. 

Wani also received aid after pitching his idea at Hackathon at MIT last semester. The response at MIT was overwhelmingly positive. "We talked to one of the people there, and we have been in touch with them ever since; they have been really helpful,” Wani said. A couple of the features from HitchHiqe have come from their work at MIT and with Watson AI, a computer system developed by IBM capable of answering questions using AI and other analytical software.  

The app has grown extremely fast; Virginia Tech has had 300 users who have driven over 8,000 miles while using the app. Wani intends to expand HitchHiqe to other schools; plans are in stone to expand to UVA and Radford by the end of February. "It’s popular here, and people really like it so we want to make sure that it is ubiquitous everywhere else,” Wani said. 

Not only will HitchHiqe gain new locations, but it’s looking to expand its team as well. The Pamplin College of Business hosted an event in late January in which student startups and local businesses could recruit students. HitchHiqe currently is in the process of interviewing applicants for various engineering and marketing positions at the company in order to sustain the app’s  growth.  

Wani has also entered HitchHiqe in Pamplin’s Entrepreneurship Challenge later this month in which he is a semifinalist for the $40,000 prize in order to fund the app’s expansion and perhaps the monetization of the app in general. The app, as of now, is free of charge; the drivers set their own prices and make 100% of the profit. In the future, HitchHiqe may be structured in such a way that the app will charge an additional 10% from the rider.  

"In the current age, carpooling has become more popular while long distance carpooling still isn’t that popular,” Wani said. "Hopefully, we can leverage today’s technology and make it safer for all students.”
"""
def search_Parent_BING(QUERY):
    ### First subroutine : 1.News Lead Generation...
    ### Important : Only works for Bing as Root node.
    # text_summarization_main(TEXT_1)
    # link = "https://www.bing.com/news/search?q="+ QUERY.lower() +"&FORM=HDRSC6"
    # html_data = requests.get(link).content
    # soup = BeautifulSoup(html_data, 'html.parser')

    # TITLE = str(soup.title).split("<title>")[1].split("-")[0].split("|")[0].strip()

    # title_pages, links_pages = getLinkTitle(soup)
    # children_text = tree_ChildrenText(links_pages) #uses first subrotine to extrapolate data from children.

    # summaries = mult_text_sum_impl(children_text) ### uses second subrotine to summarize text

    # # LDA_mult = mult_LDA(summaries) ### third subroutine done to implement Dirichlet.

    # return children_text, summaries, title_pages, links_pages ### last two are for references (meta)
    # summary_ARIZONA_text_1 = text_summarization_main(TEXT_1)
    # summary_ARIZONA_text_2 = text_summarization_main(TEXT_2)
    summary_RideSharing_1 = text_summarization_main(RIDESHARING_ONE_FULL_TEXT_3)
    summary_RideSharing_2 = text_summarization_main(BlaBla_Text4)
    summary_RideSharing_3 = text_summarization_main(HH_text)

    # LDA_1 = latent_dirichlet_allocation(summary_ARIZONA_text_1)
    LDA_2 = latent_dirichlet_allocation(HH_text)
    LDA_3 = latent_dirichlet_allocation(summary_RideSharing_1)
    LDA_4 = latent_dirichlet_allocation(summary_RideSharing_2)
    
    summaries = [summary_RideSharing_1, summary_RideSharing_3, summary_RideSharing_2]
    ldas = [LDA_3, LDA_2, LDA_4]
    title = "Arizona sets tourism records in 2018 for number of visitors and dollar spent here", "Arizona tourism industry revenue rises to nearly $24.5 billion", 
    titles = ["2020-2025 GLOBAL CARPOOLING MARKET | ADVANCE TECHNOLOGY, COMPETITIVE LANDSCAPE, REGIONAL ANALYSIS AND FORECAST", HH_title, "BlaBlaCar’s revenue grew by 71% in 2019"]
    return summaries, titles, ldas

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
    if str(soup_child.text).find("Error") == -1 or str(soup_child).find("ERROR: The request could not be satisfied") or str(soup_child).find("Error! There was an error processing your request."):
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
    summaries, titles, ldas = search_Parent_BING(QUERY)
    print(json.dumps([summaries, titles, ldas]))

if __name__ == "__main__":
    main("hotel tourism in arizona")
    
