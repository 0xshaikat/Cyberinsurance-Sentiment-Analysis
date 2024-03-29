Cyberinsurance Sentiment Analysis
---------------------------------
  Performing various sentiment analyses on scraped content on the landing pages of the top ten Cyberinsurance companies. Initially created for use as an appendix for a final paper in COMP116: Introduction to Cybersecurity at Tufts University, taught by Professor Ming Chow.

  A mature market for cyberinsurance has yet to emerge for a number of reasons, including lack of actuarial data, accounting difficulties, and a lack of legislature regulating the industry, but one interesting reason that Bandyopadhyay, Mookerjee, and Rao posit within their paper “A Model to Analyze the Unfulfilled Promise of Cyber Insurance: The Impact of Secondary Loss” is that companies may be afraid of revealing that they have undergone a breach to their insurer for fear of ruining their reputation in the public spheere

  As a result, I believed it would be interesting to perform a sentiment analysis on the landing pages of the ten major cyberinsurance firms, which includes Chubb, AX_AXL, AIG, Travelers, Beazley, Farmers, Zurichna, Progressive, Arbella, and Allianz. Considering the lack of faith of many experts within the current state of the art of the industry, I thought that a sentiment analysis on the landing pages of these firms would resemble a scenario in which an entity in crisis were searching for cyberinsurance as a means of security. As for my methodology, I scraped the readable text-data from each of the landing pages, preprocessed them for NLP tasks by removing punctuation, lowercasing input, as well as removing any conjunctions and stopwords (words that provide no value to the semantic of a sentence, such as articles). After doing this for each of 10 websites, I converted each text block into a word list, which was matched against an opinion word list provided by Minqing Hu and Bing Liu, as part of an appendix for their paper, "Mining and Summarizing Customer Reviews.", published at UIUC. After finding the positive, negative, and neutral proportion values for each of the webpages, I created word clouds for each webpage, and then ran a sentiment/polarity assessment on each webpage using Textblob, a pre trained NLP package for python.
  
Requirements
------------

You need Python 3.5 or later to run this. 

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

For other Linux flavors, macOS and Windows, packages are available at

  http://www.python.org/getit/

Packages 
--------
    $ pip install bs4
    $ pip install ntlk
    $ pip install matplotlib
    $ pip install plotly
    $ pip install wordcloud
    $ pip install textblob
    $ pip install numpy

Usage
-----
    $ python sentiment_analysis.py
 
