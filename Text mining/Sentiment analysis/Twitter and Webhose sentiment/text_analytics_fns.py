#-------------------------------------------------------------------
from tweepy import *
# miscellenuous libraries
import pandas as pd
import numpy as np
import pickle,os,json,random,nltk,re,io
from IPython.display import *
from geo_api import *
import re as regex


import utils_1
#%matplotlib inline
#unsupervised sentiment model
from encoder import Model
from afinn import Afinn
affn=Afinn(emoticons=True)
model=Model()

from pytrends.request import TrendReq
pytrend = TrendReq(hl='en-US', geo='India')

from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers import DocumentParser

import webhoseio
from datetime import *
import time
from collections import  deque
#-------------------------------------------------------------------
def get_sentiment_affn(text):
    """
    return vector of sentiments
    text : english text for which sentiment have to be defined
    """
        # tokenizing into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenized_text=tokenizer.tokenize(text)
    sentiments_vec=[affn.score(i) for i in tokenized_text]
    
    return sentiments_vec

def get_sentiment_dnn(text):
    """
    return vector of sentiments
    text : english text for which sentiment have to be defined
    """
        # tokenizing into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenized_text=tokenizer.tokenize(text)


    #getting sentiment vectors
    sentiments_vec=model.transform(tokenized_text)

    # getting sentiment value from classifier neuron
    neuron_index=2388
    sentiments=[i[neuron_index] for i in sentiments_vec]
    return sentiments

def get_sentiment_stat(text,method='affin'):
    """
    return statistics of sentiment 
    text : english text for which sentiment have to be defined
    method : affin or dnn
    """
    if method=='affin':
        sentiments_list=get_sentiment_affn(text)
    elif method=='dnn':
        sentiments_list=get_sentiment_dnn(text)
    else:
        print("error in correct input")
        return 
    #sentiments_posts.append(sentiments_list)
    #print(summarize(post['text']))
    #print("overall sentiment median(+ve frequency) = "+str(np.median(sentiments_list)),\
            #"overall sentiment mean(average) = "+str(np.mean(sentiments_list)),\
            #"overall sentiment sum(Strongess) = "+str(np.sum(sentiments_list)))
    #display(text2sentiment_heatmap(post['text'],sentiments_list))
    #time.sleep(30)
    return {"overall sentiment median(+ve frequency)" : np.median(sentiments_list),
            "overall sentiment mean(average)":np.mean(sentiments_list),
            "overall sentiment sum(Strongess)":np.sum(sentiments_list)}

#-------------webhose Functions------------------------------------------------------------
def preprocess_webhose_op(json_op):
    """
    returns webhose output in preprocess 
    json_op: json_op from webhose data funtion
    """
    l1=[]
    for i in json_op['posts']:
        l1.append(i['thread'].values())

    meta_keys=['ord_in_thread', 'entities', 'external_links', 'author', 'rating','crawled', 'highlightText', 'highlightTitle', 'language']\


    l2=[]
    for i in json_op['posts']:
        l2.append([i[k] for k in meta_keys])

    l3=[i['text'] for i in json_op['posts']]

    meta_data=pd.DataFrame(l2,columns=meta_keys)

    thread=pd.DataFrame(l1,columns=list(json_op['posts'][0]['thread'].keys()))

    text=pd.DataFrame(l3,columns=['text'])

    concat_tab=pd.concat([thread,meta_data,text],axis=1)
    return concat_tab


def get_data_webhose(keyword=["modi","bjp"],location="India",sort=None,realtime_flag=True,timeduration=30):
    """
    return tabular output from webhose
    keyword:  api will look for all keywords in text space
    location: country detail
    realtime_flag=if true only one query will be done for those keywords, only latest data will be picked up(maximum one day old)
    timeduration= if realtime_flag is False how long in past to look max 30 days
    
    """
    tokens=['d4dfcbf5-ae63-4b33-a2f1-f13d0648cd8f','3025d12f-e409-42a3-83f0-8c992864a154','f414cd7e-a1d2-4483-af41-893ccc6f07']
    # append more tokens if required 
    tokens=deque(tokens)
    keyword=" ".join(keyword)
    q=keyword+" location:"+location+" site_type:news OR site_type:blogs OR site_type:discussions"
    output=None
    if realtime_flag:
        time1=datetime.now()-timedelta(days=3)
        time1=int(time.mktime(time1.timetuple())*1e3 + time1.microsecond/1e3)
    else:
        time1=datetime.now()-timedelta(days=min(30,timeduration))
        time1=int(time.mktime(time1.timetuple())*1e3 + time1.microsecond/1e3)        
    while True:
        try:
            webhoseio.config(token=tokens[0])


            query_params = {
            "q": q,
            #"sort": "social.facebook.shares",
            "ts":str(time1)
            }

            ## getting thread data
            output = webhoseio.query("filterWebContent", query_params)
            break
        except Exception as e:
            print(e)
            if ((output is None) or (output['requestsLeft']<=10) ):
                print("less than 10 requests are left") 
                tokens.rotate(-1)
    if ((output is None) or (output['requestsLeft']<=10) ):
        print("less than 10 requests are left")    
        tokens.rotate(-1)
        
    op1=preprocess_webhose_op(output)
    if not realtime_flag:
        op_all=[op1]
        for i in range(1,int(output['totalResults']/100)):
            op_all.append(preprocess_webhose_op(webhoseio.get_next()))
        op_all=pd.concat(op_all)
        return op_all
    else:
        return op1
    
def get_blog_data_realtime(keyword=["modi","bjp"],location="India",sort=None):
    """
    return realtime tabular output from webhose
    keyword:  api will look for all keywords in text space
    location: country detail
    sort : refer https://docs.webhose.io/v1.0/docs/get-parameters
    """
    return get_data_webhose(keyword,location,sort,realtime_flag=True)
def get_blog_data_historical(keyword=["modi","bjp"],location="India",sort=None,timeduration=10):
    """
    return historical tabular output from webhose
    keyword:  api will look for all keywords in text space
    location: country detail
    sort : refer https://docs.webhose.io/v1.0/docs/get-parameters
    timeduration : duration in number of days
    """
    return get_data_webhose(keyword,location,sort,realtime_flag=False,timeduration=timeduration)
    

def get_tfidf_keywords(corpus,scoring_method='mean'):
    """
    returns keywords score based on scoring method provided
    corpus : corpus of text
    scoring_method=mean/sum/
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf1=TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
    tfidf_matrix=tfidf1.fit_transform(corpus)
    feature_names = tfidf1.get_feature_names() 
    #dense = tfidf_matrix.todense()
    if scoring_method=='mean':
        return pd.DataFrame({"feature_names":feature_names,"scores":tfidf_matrix.mean(axis=0).tolist()[0]}).sort_values('scores',ascending=False)
    if scoring_method=='sum':
        return pd.DataFrame({"feature_names":feature_names,"scores":tfidf_matrix.sum(axis=0).tolist()[0]}).sort_values('scores',ascending=False)
    else:
        print('wrong inputs')



def summarize_text(x,summarizer='SumBasicSummarizer',sentence_count=5):
    """
    returns summarized output of text
    x: list of sentences or paragraphs
    summarizer=['LsaSummarizer','LuhnSummarizer',\
       'EdmundsonSummarizer','LexRankSummarizer','TextRankSummarizer',\
      'SumBasicSummarizer','KLSummarizer']

    """
    LANGUAGE = "english"
    stemmer = Stemmer(LANGUAGE)
    l=[]
    for i in x:
        try:
            summarizer_fn=eval(summarizer+'(stemmer)')
            parser=PlaintextParser(i,Tokenizer(LANGUAGE))
            summary=summarizer_fn(parser.document, sentence_count)
            l.append(" ".join([str(k) for k in summary]))
            
            #print(l)
        except Exception as e:
            print(e)
            pass
    if len(l)>1:    
        return ' '.join(l) 
    else:
        return l[0]
def summarize_combined(x,sentence_count=4):
    """
    returns summarized output of text thru TextRankSummrizer
    x: list of sentences or paragraphs
    sentence_count : number of sentences in summary
    """
    return summarize_text([summarize_text(x,sentence_count=sentence_count+4)],summarizer='TextRankSummarizer',sentence_count=sentence_count)

    
#get_name_only=lambda name:''.join([i for i in name if not i.isdigit()]).replace("_","").lower()
get_name_only=lambda name:re.sub('[^A-Za-z]+', '', name).lower()
get_name_only.__name__='get_name_only'

def get_twitter_trend(location_name):
    """
    
    returns trending twitter topic w.r.t to a location from twitter
    location_name: name of location
    """
    c = geoAPI()
    loc=c.search(location_name)['result']['results'][0]['geometry']['location']

    ckey = ['Qdv3DjMTmJOsgZtv4JjRJM11H','GEjIsTwSBSUcdlDlwvE12uM1u']
    consumer_secret =['kC9MhLADtAeSt98iC9SEYpgxkP9bvGb8EU5FY8MFDdGEZfhEIx','MkAZnDFFMyOnrWLXFxqybRbfqYdVXFKvrCa8GJRCd6KARwJf2m']
    access_token_key = ["823437635322609665-hLf1jbSpyUFA9CySBujiS7Uu8dXqYpi",'904567451911933952-0Bq9OpbDnC7dwl0auFJYEQ9ae1JgHUt']
    access_token_secret = ["D0CfF8Pq7ssUz5ejaTZWDqtXPk453FdaH1wn7ao1IW6j4",'TfDJ0ln7iOnHvCMfAHA4Pk3PwkCGS3X2rtuvAw2I2T42R']
    for i in range(2):
        try:
            auth = OAuthHandler(ckey[i], consumer_secret[i]) #OAuth object
            auth.set_access_token(access_token_key[i], access_token_secret[i])
            #get_name_only=lambda name:''.join([i for i in name if not i.isdigit()]).replace("_","").lower()
            api = API(auth)
            loc_id=api.trends_closest(loc['lat'],loc['lng'])[0]['parentid']
            print(location_name,loc_id)
            get_name_only=lambda name:re.sub('[^A-Za-z]+', '', name).lower()
            get_name_only.__name__='get_name_only'

            api = API(auth)
            t=api.trends_place(loc_id)
            break
        except Exception as e:
            print(e)
        
    return [get_name_only(i['name']) for i in t[0]['trends']]
def get_sentiment_blog(op_all,keywords,source="blogs",take_all_data=False):
    """
	develop sentiment analytics in webhose data 
	op_all : output dataframe from webhose
    source : blogs/news/discussions
    take_all_data : if you want to use all data
	"""
    print(op_all.shape)
    op_all['status']=op_all['text'].\
    str.contains("".join(["(?=.*?"+i+"[^$]*)" for i in keywords]),case=False)

    op_all=op_all[op_all['status']]
    if not take_all_data:
        op_all=op_all[op_all['site_type']==source]
    print(op_all.shape)
    sentiments_stat=op_all['text'].apply(lambda x: get_sentiment_stat(x))
    op_all['published']=pd.DatetimeIndex(op_all['published'])
    op_all['stats']=pd.DataFrame(sentiments_stat)   
    op_all['median_sentiment_level']=op_all['stats'].apply(lambda x :x['overall sentiment median(+ve frequency)'])
    op_all['date']=op_all['published'].apply(lambda x: x.date())
    return op_all

#----------------------twitter streaming functions ----------------------------------------------------------
class listener(StreamListener):
    """
    Custom listerner class for twitter  data listening
    """

    def __init__(self, start_time, time_limit=60):
        self.time = start_time
        self.limit = time_limit
        self.tweet_data = []

    def on_data(self, data):

        saveFile = io.open('raw_tweets.json', 'a', encoding='utf-8')

        while (time.time() - self.time) < self.limit:

            try:
                self.tweet_data.append(data)
                return True
            except Exception as e:
                print ('failed ondata,', str(e))
                time.sleep(5)
                pass
        
        saveFile = io.open('raw_tweets.json', 'w', encoding='utf-8')
        saveFile.write(u'[\n')
        saveFile.write(','.join(self.tweet_data))
        saveFile.write(u'\n]')
        saveFile.close()
        sys.exit()

    def on_error(self, status):
        print (status)

def get_tweets(keywords=['modi','NDA','kejriwal','iot'],time_limit=5,location='India'):
    """
    returns tweets for keywords list passed
    keywords: tweets filtering criteria
    time_limit= time limit in seconds for which stream has to be play
    location : location for which filtering have to be done
    """
    ckey = 'Qdv3DjMTmJOsgZtv4JjRJM11H'
    consumer_secret ='kC9MhLADtAeSt98iC9SEYpgxkP9bvGb8EU5FY8MFDdGEZfhEIx'
    access_token_key = "823437635322609665-hLf1jbSpyUFA9CySBujiS7Uu8dXqYpi"
    access_token_secret = "D0CfF8Pq7ssUz5ejaTZWDqtXPk453FdaH1wn7ao1IW6j4"
    c = geoAPI()
    pos=c.search(location)
    bbox=pos['result']['results'][0]['geometry']['bounds']
    l=[]
    #Correct format is: SouthWest Corner(Long, Lat), NorthEast Corner(Long, Lat)
    for k,v in bbox.items():
        l+=[v['lat'],v['lng']]
    l.reverse()
    print('bounding box for', location, l)
    start_time = time.time() #grabs the system time
    auth = OAuthHandler(ckey, consumer_secret) #OAuth object
    auth.set_access_token(access_token_key, access_token_secret)
    try:
        twitterStream = Stream(auth, listener(start_time, time_limit=time_limit)) #initialize Stream object with a time out limit
        twitterStream.filter(track=keywords ,languages=['en'],locations=l)  #call the filter method to run the Stream Object
    except:
        twitterStream.disconnect()

    with open('raw_tweets.json') as data_file:    
        data = json.load(data_file)
    tweets=[i['text'] for i in data]
    return pd.DataFrame({'text':tweets})

def get_hist_tweets(q='modi',count=100,result_type='recent',location='bangalore',distance='100km',days=7):
    """
    It will return last 7 days tweets 
    https://dev.twitter.com/rest/reference/get/search/tweets
    get tweets with given keyword, maximum count per tweet
    q : keywords. or query parameter
    result_type : recent/popular
    location : location name
    distance : distance from location
    count : maximum count for topic related search(max is 100)
    days : number of days to cover in twitter stream 
    """
    date=datetime.now()-timedelta(days=days)
    consumer_key = 'Qdv3DjMTmJOsgZtv4JjRJM11H'
    consumer_secret ='kC9MhLADtAeSt98iC9SEYpgxkP9bvGb8EU5FY8MFDdGEZfhEIx'
    access_token = "823437635322609665-hLf1jbSpyUFA9CySBujiS7Uu8dXqYpi"
    access_token_secret = "D0CfF8Pq7ssUz5ejaTZWDqtXPk453FdaH1wn7ao1IW6j4"    
    
    c = geoAPI()
    loc=c.search(location)['result']['results'][0]['geometry']['location']
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = API(auth)
    l=[]

    for tweet in Cursor(api.search,q=q,result_type=result_type,count=count,\
                               lang="en",geocode=str(loc['lat'])+','+str(loc['lng'])+','+str(distance),\
                               since_id=str(date.year)+'-'+str(date.month)+'-'+str(date.day)).items():
        l.append([tweet.created_at, tweet.text])
#         try:
#             while True:
#                 print(tweet.pop())
#         except:
#             pass
        
    return pd.DataFrame(l,columns=['created_at','text'])
        #csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
     ## data cleaning api for tweets
def clean_tweets(tweets_df):
    """
    return clean tweets 
    tweets_df : tweet dataframe
    """
    ## cleaning twitter data
    cleanuper=TwitterCleanuper()
    tweets_cleaned=tweets_df.copy()
    for cleanup_method in cleanuper.iterate():
        #print(cleanup_method)
        tweets_cleaned = cleanup_method(tweets_cleaned)
    return tweets_cleaned
    

def plot_twitter_stats(tweets_df):
    """
    return sentiment dataframe, and its statistics and plot visualization of tweets dataframe(made for jupyter notebook) 
    tweets_df : tweets dataframe
    
    """
    ## cleaning twitter data
    cleanuper=TwitterCleanuper()
    tweets_cleaned=tweets_df.copy()
    for cleanup_method in cleanuper.iterate():
        #print(cleanup_method)
        tweets_cleaned = cleanup_method(tweets_cleaned)
        ##using affinn based approach
    tweets_df_word_li=tweets_df.copy()
    tweets_df_word_li['sentiment']=tweets_cleaned['text'].apply(lambda x: affn.score(x))
    tweets_df_word_li['sentiment_tag']=tweets_df_word_li['sentiment'].apply(lambda x:sentiment_tag(x) )
    return (tweets_df_word_li,get_sentiment_statistics(plot_title='based on word list',tweets_df=tweets_df_word_li)) 

#------------------cleaning generator for twitter data--------------------
class TwitterCleanuper:
    """
    cleaning iterator for twitter data.
    """
    def iterate(self):
        for cleanup_method in [self.remove_urls,
                               self.remove_usernames,
                               self.remove_na,
                               self.remove_special_chars,
                               self.remove_numbers]:
            yield cleanup_method

    @staticmethod
    def remove_by_regex(tweets, regexp):
        tweets.loc[:, "text"].replace(regexp, "", inplace=True)
        return tweets

    def remove_urls(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"http.?://[^\s]+[\s]?"))

    def remove_na(self, tweets):
        return tweets[tweets["text"] != "Not Available"]

    def remove_special_chars(self, tweets):  # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            tweets.loc[:, "text"].replace(remove, "", inplace=True)
        return tweets

    def remove_usernames(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"@[^\s]+[\s]?"))

    def remove_numbers(self, tweets):
        return TwitterCleanuper.remove_by_regex(tweets, regex.compile(r"\s?[0-9]+\.?[0-9]*"))
# class TwitterData_Cleansing(TwitterData_Initialize):
#     def __init__(self, previous):
#         self.processed_data = previous.processed_data
        
#     def cleanup(self, cleanuper):
#         t = self.processed_data
#         for cleanup_method in TwitterCleanuper.iterate():
#             if not self.is_testing:
#                 t = cleanup_method(t)
#             else:
#                 if cleanup_method.__name__ != "remove_na":
#                     t = cleanup_method(t)

#         self.processed_data = t
#----------miscellenious function for twitter sentiment-------------------------------
def sentiment_tag(sentiment_val):
    """
    sentiment_val : returns sentiment_tag based on sentiment value
    """
    if sentiment_val<0:
        return 'Negative'
    elif sentiment_val>0:
        return 'Positive'
    else:
        return 'Neutral'
def get_sentiment_statistics(plot_title,tweets_df,plot=True):
    """develop statistics for twitter feed used for notebook only
       plot_title : title for plots
       tweets_df : dataframe of tweets 
       plot : bool, return plot or not
    """
    num_positive=sum(tweets_df['sentiment']>0)
    num_negative=sum(tweets_df['sentiment']<0)
    num_neutral=sum(tweets_df['sentiment']==0)
    tweets_df1=tweets_df.sort_values('sentiment')
    most_positive_sentiment=tweets_df1.tail(1)['text']
    most_negative_sentiment=tweets_df1.head(1)['text']
    
    if plot:
        ax=tweets_df['sentiment_tag'].value_counts().plot(kind='bar')
        tweets_df['sentiment'].plot(kind='density',color='r',title=plot_title+\
                                        '\n number of positive sentiment='\
                                        +str(num_positive)+"\n number of negative sentiment="\
                                        +str(num_negative)+"\n number of neutral sentiment="\
                                        +str(num_neutral),figsize=(15,7),ax=ax,secondary_y=True)
    return pd.DataFrame([num_negative,num_neutral,num_positive,most_negative_sentiment\
                         ,most_positive_sentiment],index=['number of negative revies'\
                         ,'number of neutral reviews','number of positive reviews'\
                          ,'most negative sentiment','most positive sentiment'])
