'''
NLTK  Natural Language Toolkit
VADER (Valence Aware Dictionary and Sentiment Reasoner) is a pre-trained model that uses rule-based values
tuned to sentiments from social media. It evaluates the text of a message and gives you an assessment of not
just positive and negative,  but the intensity of that emotion as well

 Beautiful Soup parses xml/html,etc type data from websites
 '''

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt



#####################    Parsing html data           #######################################
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'FB']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})    # if header isnt specified then access is forbidden
                                                                # user agent allows access and ensures we are accessing data from identified browser myapp is name of proj
    response = urlopen(req)                                     # returns a http response object
    html = BeautifulSoup(response, features='html.parser')      #print(html) source code of website; bs parses as html doc
    news_table = html.find(id='news-table')                     #news is present in table with id=news-table
    news_tables[ticker] = news_table                            #ticker is company symbol.. add all news related to it in its key



########################    Parsing and Manipulating Finviz Data  ##########################
parsed_data = []

for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):                        #finds all tr tags of the html (table row)

        title = row.a.text                                      #a= anchor tag ..extracting actual text of each row ; print(title)
        date_data = row.td.text.split(' ')                      #extracting timestamp

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])


###################   Sentiment Analysis #####################################
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

vader = SentimentIntensityAnalyzer()
#create a function to extract compound score of a text
f = lambda title: vader.polarity_scores(title)['compound']  #vader.polarity_scores('text') gives sentiment scores b/w +1 and -1 (compound score is overall score)
df['compound'] = df['title'].apply(f)                       #apply that func on title column and create col compound to store those scores
df['date'] = pd.to_datetime(df.date).dt.date


#################   Visualizing data  #######################################


mean_df = df.groupby(['ticker', 'date']).mean().unstack()  #group df cols and calculate mean of each day
#print(mean_df)
mean_df = mean_df.xs('compound', axis="columns")

mean_df.plot(kind='bar')
plt.ylim([-1, 1])
plt.xlabel('last 7 days')
plt.ylabel('sentiment score')
plt.grid(True)
plt.title('Sentiment Analysis of 7 days')
plt.show()
