# sentimental

Simple sentiment analysis with Python

## How it works

Telegram messages from https://t.me/CryptoComOfficial from May 1 to
and including May 15, 2021.

Pre-processing the data by remove non-English messages. From these, keep only messages
that mention either “SHIB” or “DOGE.”

Analysing the data with help library with nltk. sentiment.vader.SentimentIntensityAnalyzer.
We assign the polarity of the sentiment score with -1 for the negative and 1 positive and 0 for neutral sentiments.

We plot the data plotly which is as follows (Ploting the number of messages per day and the average sentiment per day using the)

## Documentation

getCategory() clasifies data into SHIB or DOGE:
and calls the cleanData to clean th data which only returns Engilsh literals (ASCII 65 - 90 or 97 -122 ) or special chars ',', '.', '?', '!', ' '

analyzetext() analyses the data and assigns a polarity  sentiment score as positive(1), negative(-1) or neutral(0) to the each text in the messages.


## Results


## how to run code 
install nltk, plotly, tqdm, pandas

pip3 install plotly
pip3 install nltk
pip3 install tqdm
pip3 install pandas


