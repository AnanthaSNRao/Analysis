import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import pandas as pd
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

nltk.download('vader_lexicon')

def getCategory(msgs):
    msg_list = {'shiba':[], 'doge':[]}
    
    for msg in tqdm(msgs, desc="cleaning data"):
        mt = msg['text']
        if type(mt) == list:
            k = ''
            for a in mt:
                if type(a) == str:
                    k +=a
            mt = k
        
        if "SHIB" in mt.upper():
            msg['text'] =  cleanData(mt)
            msg_list['shiba'].append(msg)
        elif "DOGE" in mt.upper():
            msg['text'] = cleanData(mt)
            msg_list['doge'].append(msg)
    return msg_list

def cleanData(text):
    s = ''
    special_char = [',', '.', '?', '!', ' ']
    for ch in text:
        if  65 <ord(ch)< 90 or 97 <ord(ch)< 122 or ch in special_char :
            s+=ch
    return s

def getDate(date):
    d = date.split('T')
    return d[0]

def analyzetext(msgs, coin):
    msg_list = []
    for msg in tqdm(msgs, desc="Analysis of text "+ coin):
        mt = msg['text']
       
        score = SentimentIntensityAnalyzer().polarity_scores(mt)
        neg = score['neg']
        
        pos = score['pos']
       
        if neg > pos:
            msg['polarity'] = -1
        elif pos > neg:
            msg['polarity'] = 1
        elif pos == neg:
            msg['polarity'] = 0
        msg['date'] = getDate(msg['date'])
        msg_list.append(msg)
        
    return msg_list

def main():
    msgs = []
    try:
        
        f = open('result.json')
        msgs = json.load(f)
        f.close()
        
        msg_list = getCategory(msgs['messages'])
        
        shiba= pd.DataFrame(analyzetext(msg_list['shiba'], 'SHIB'))
        doge= pd.DataFrame(analyzetext(msg_list['doge'], 'DOGE'))
        d1 = doge.groupby(['date', 'polarity'], as_index=False).size().agg(list)
        d = shiba.groupby(['date', 'polarity'], as_index=False).size().agg(list)

        da1 = doge.groupby(['date'], as_index=False).mean().agg(list)
        da = shiba.groupby(['date'], as_index=False).mean().agg(list)
        

        fig_shib = make_subplots(rows=2, cols=1)
        fig_shib.append_trace(go.Bar(x=d['date'], y=d['size']), row=1, col=1)
        fig_shib.append_trace(go.Scatter(x=da['date'], y=da['polarity']), row=2, col=1)
        fig_shib.update_layout(height=1000, width=1000, title_text="Shib Subplots number of messages(top) Average sentiment(bottom) v/s date")
        fig_shib.show()

        fig_doge = make_subplots(rows=2, cols=1)
        fig_doge.append_trace(go.Bar(x=d1['date'], y=d['size']), row=1, col=1)
        fig_doge.append_trace(go.Scatter(x=da1['date'], y=da1['polarity']), row=2, col=1)
        fig_doge.update_layout(height=1000, width=1000, title_text="Doge Subplots number of messages(top) Average sentiment(bottom) v/s date")
        fig_doge.show()
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()
