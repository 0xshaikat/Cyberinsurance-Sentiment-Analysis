'''
@author: Shaikat Islam
@title: sentiment_analysis.py
@professor: Professor Ming Chow
@date: 13-12-2019
@purpose: Appendix for COMP116 Final Project
'''
from bs4 import BeautifulSoup
from bs4.element import Comment
import nltk
import ssl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import urllib.request
import numpy as np
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

#===============================================================================
#           PART 1: BASIC SENTIMENT ANALYSIS
#===============================================================================
# results
chubb_rst = {}
chubb_subj = []
ax_axl_rst = {}
ax_axl_subj = []
aig_rst = {}
aig_subj = []
travelers_rst = {}
travelers_subj = []
beazley_rst = {}
beazley_subj = []
farmers_rst = {}
farmers_subj = []
zurichna_rst = {}
zurichna_subj = []
progressive_rst = {}
progressive_subj = []
arbella_rst = {}
arbella_subj = []
allianz_rst = {}
allianz_subj = []

# appostrophe words used in preprocessing
appos = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"
}

# positive opinion words
with open('positive-words.txt') as f:
    pos_words = f.read()
    pos_words = re.sub('[,\.()":;!@#$%^&*\d]|\'s|\'', '', pos_words)
    pos_list = pos_words.replace('\n',' ').replace('  ',' ').lower().split(' ')

# negative opinion words
with open('negative-words.txt') as f:
    neg_words = f.read()
    neg_words = re.sub('[,\.()":;!@#$%^&*\d]|\'s|\'', '', neg_words)
    neg_list = neg_words.replace('\n',' ').replace('  ',' ').lower().split(' ')

# dictionary of URLs for each of the top ten cyberinsurance companies
top_cyber_urls = {
"CHUBB":"https://www.chubb.com/us-en/cyber-risk-management/?gclid=CjwKCAiA58fvBRAzEiwAQW-hzeUjyXuJc7UqPS_q_XzygQU2Cs6MdIhRBbZn6_4cMZPguhbt6pmf5RoCQTQQAvD_BwE",
"AX_AXL":"https://axaxl.com/insurance/products/cyber-insurance",
"AIG":"https://www.aig.com/business/insurance/cyber-insurance",
"Travelers":"https://www.travelers.com/cyber-insurance",
"Beazley":"https://www.beazley.com/usa/cyber_and_executive_risk/cyber_and_tech.html",
"Farmers":"https://www.farmers.com/business/general-liability-insurance/",
"Cincinnati":"https://www.cinfin.com/business-insurance/products/cyber-risk",
"Zurichna":"https://www.zurichna.com/en/about/cybersecurity?WT.mc_id=google_insurance&WT.srch=1",
"Progressive":"https://www.progressivecommercial.com/business-insurance/cyber-insurance/",
"Arbella":"https://www.arbella.com/insurance/business-insurance/data-breach-and-cyber?utm_source=GOOGLE&utm_medium=cpc&utm_term=insurance%20cyber&utm_campaign=Cyber&gclid=CjwKCAiA58fvBRAzEiwAQW-hzTEa8Cc6BS5DdCID8hVqqIdHucRY7KQcmWPcywlsBaD6lpYz5jvc2xoCJrkQAvD_BwE",
"Allianz":"https://www.agcs.allianz.com/solutions/financial-lines-insurance/cyber-insurance.html"}

'''
get_visible_elements: match tags to visible html tags
inputs: elem
output: bool
'''
def get_visible_elements(elem):
    if elem.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(elem, Comment):
        return False
    return True

'''
get_text_from_html: return text from webpage
inputs: body
output: string
'''
def get_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(get_visible_elements, texts)
    return u" ".join(t.strip() for t in visible_texts)

'''
store_text_in_numpy_arr_and_preprocess: preprocess string data for sentiment analysis
inputs: url_ref (str)
output: word_list (arr)
'''
def store_text_in_numpy_arr_and_preprocess(url_ref):
    # get content from web
    context = ssl._create_unverified_context()
    html = urllib.request.urlopen(url_ref, context=context).read()
    text = get_text_from_html(html)
    # preprocess and tokenize data
    text = re.sub('[,\.()":;!@#$|%^&*\d]|\'s|\'', '', text)
    word_list = text.replace('\n',' ').replace('  ',' ').lower().split(' ')
    word_list = filter(None, word_list)
    word_list = list(filter(None, word_list))
    # remove stopwords from sentences (words that are most commonly occurring, but not
    # relevant in the context of the data)
    # we use english stopwords here, which may not be relevant in the context of foreign words
    stop_words = set(stopwords.words('english'))
    for stop_word in stop_words:
        for word in word_list:
            if word == stop_word:
                word_list.remove(word)
    # remove stand-alone punctuation plus non-alphanumeric characters
    for word in word_list:
        if not word.isalpha():
            word_list.remove(word)
    # change appostrophe using dict
    for word in word_list:
        if word in appos:
            word = appos[word]
    return word_list

'''
generate_wordcloud: generates wordcloud from corpus to png
inputs: text (arr), filename (str)
output: stdout
'''
def generate_wordcloud(text, filename):
    s = " ".join(text)
    wordcloud = WordCloud(font_path='C:/Windows/Fonts/arial.ttf',
                          relative_scaling = 1.0,
                          stopwords = {'to', 'of'} # set or space-separated string
                          ).generate(s)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(filename)

'''
simple_sentiment_ratio: gets pos, neg, neutral score from corpus using preset arrays of words
inputs: name (str), arr (arr), rst (dict)
output: stdout
'''
def simple_sentiment_ratio(name, arr, rst):
    pos = 0
    neg = 0
    neutral = 0
    ratio = 0
    for word in arr:
        if word in neg_list:
            neg += 1
        elif word in pos_list:
            pos += 1
        else:
            neutral += 1
    percent_neg = neg/(pos + neg + neutral)
    percent_pos = pos/(pos + neg + neutral)
    percent_neutral = neutral/(pos + neg + neutral)
    lst = []
    lst.append(percent_neg)
    lst.append(percent_neutral)
    lst.append(percent_pos)
    label = name + "_SIMPLE"
    rst[label] = lst
    print(name + ":")
    print("Negative percentage: " + str(percent_neg))
    print("Positive percentage: " + str(percent_pos))
    print("Neutral percentage: " + str(percent_neutral))
    print("\n")

# corpuses (body of text from website)
chubb = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["CHUBB"])
ax_axl = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["AX_AXL"])
aig = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["AIG"])
travelers = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Travelers"])
beazley = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Beazley"])
farmers = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Farmers"])
cincinnati = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Cincinnati"])
zurichna = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Zurichna"])
progressive = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Progressive"])
arbella = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Arbella"])
allianz = store_text_in_numpy_arr_and_preprocess(top_cyber_urls["Allianz"])

# simple sentiment ratios
print("SIMPLE SENTIMENT ANALYSIS WITH PRESET ARRAYS OF NEGATIVE AND POSITIVE WORDS: ")
simple_sentiment_ratio("CHUBB", chubb, chubb_rst)
simple_sentiment_ratio("AX_AXL", ax_axl, ax_axl_rst)
simple_sentiment_ratio("AIG", aig, aig_rst)
simple_sentiment_ratio("Travelers", travelers, travelers_rst)
simple_sentiment_ratio("Beazley", beazley, beazley_rst)
simple_sentiment_ratio("Farmers", farmers, farmers_rst)
simple_sentiment_ratio("Zurichna", zurichna, zurichna_rst)
simple_sentiment_ratio("Progressive", progressive, progressive_rst)
simple_sentiment_ratio("Arbella", arbella, arbella_rst)
simple_sentiment_ratio("Allianz", allianz, allianz_rst)

#==============================================================================
#           PART 2: PRETRAINED TEXTBLOB MODEL
#==============================================================================
'''
subjectivity_ratio: get subjectivity and polarity of corpus
inputs: name (str), arr (arr), arr_1 (arr)
output: none
'''
def subjectivity_ratio(name, arr, arr_1):
    subj = TextBlob(" ".join(arr_1)).sentiment[1]
    polarity = TextBlob(" ".join(arr_1)).sentiment[0]
    print("Subjectivity score for " + name + ": " + str(subj))
    print("Polarity score for " + name + ": " + str(polarity))
    arr.append(subj)
    arr.append(polarity)

print("SUBJECTIVITY ANALYSIS USING PRETRAINED TEXTBLOB: ")
subjectivity_ratio("CHUBB", chubb_subj, chubb)
subjectivity_ratio("AX_AXL", ax_axl_subj, ax_axl)
subjectivity_ratio("AIG", aig_subj, aig)
subjectivity_ratio("Travelers", travelers_subj, travelers)
subjectivity_ratio("Beazley", beazley_subj, beazley)
subjectivity_ratio("Farmers", farmers_subj, farmers)
subjectivity_ratio("Zurichna", zurichna_subj, zurichna)
subjectivity_ratio("Progressive", progressive_subj, progressive)
subjectivity_ratio("Arbella", arbella_subj, arbella)
subjectivity_ratio("Allianz", allianz_subj, allianz)

#==============================================================================
#           PART 3: GRAPHS
#==============================================================================
'''
 To Ming: uncomment this function code if you would like to see the plotting of rewards
 per episode for SARSA and Q-LEARNING, using plotly
 Instructions:
 $ npm install -g electron@1.8.4 orca
 $ pip install plotly==4.4.1
 $ pip install psutil requests

def plot_polarity_and_subjectivity():
	fig = go.Figure()
	# Create and style traces
	fig.add_trace(go.Scatter(x=np.asarray(chubb_subj[1]), y=np.asarray(chubb_subj[0]), name='CHUBB',
	line=dict(color='aliceblue')))
	fig.add_trace(go.Scatter(x=np.asarray(ax_axl_subj[1]), y=np.asarray(ax_axl_subj[0]), name = 'AX_AXL',
	line=dict(color='black')))
	fig.add_trace(go.Scatter(x=np.asarray(aig_subj[1]), y=np.asarray(aig_subj[0]), name='AIG',
	line=dict(color='crimson')))
	fig.add_trace(go.Scatter(x=np.asarray(travelers_subj[1]), y=np.asarray(travelers_subj[0]), name='TRAVELERS',
	line=dict(color='cyan')))
	fig.add_trace(go.Scatter(x=np.asarray(beazley_subj[1]), y=np.asarray(beazley_subj[0]), name='BEAZLEY',
	line=dict(color='deeppink')))
	fig.add_trace(go.Scatter(x=np.asarray(farmers_subj[1]), y=np.asarray(farmers_subj[0]), name='FARMERS',
	line=dict(color='goldenrod')))
	fig.add_trace(go.Scatter(x=np.asarray(zurichna_subj[1]), y=np.asarray(zurichna_subj[0]), name='ZURICHNA',
	line=dict(color='lavender')))
	fig.add_trace(go.Scatter(x=np.asarray(progressive_subj[1]), y=np.asarray(progressive_subj[0]), name='PROGRESSIVE',
	line=dict(color='lime')))
	fig.add_trace(go.Scatter(x=np.asarray(arbella_subj[1]), y=np.asarray(arbella_subj[0]), name='ARBELLA',
	line=dict(color='mediumpurple')))
	fig.add_trace(go.Scatter(x=np.asarray(allianz_subj[1]), y=np.asarray(allianz_subj[0]), name='ALLIANZ',
	line=dict(color='deepskyblue')))
	fig.update_layout(title='Subjectivity vs Polarity for 10 Cyberinsurance Company Landing Pages', xaxis_title='polarity', yaxis_title='subjectivity')
	fig.write_image("cyber_insurance.png")

plot_polarity_and_subjectivity()
'''
