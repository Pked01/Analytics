

```python
import text_analytics_fns as taf
```

    load params
    loading model
    creating tensorflow session
    create model
    embd function
    loading Parameters
    create mlstm
    loading Parameters
    loading Parameters
    loading Parameters
    loading Parameters
    loading Parameters
    loading Parameters
    loading Parameters
    loading Parameters
    loading Parameters
    fully connected layer creation
    loading Parameters
    loading Parameters


#### get_last_week_tweets


```python
help(taf.get_hist_tweets)
```

    Help on function get_hist_tweets in module text_analytics_fns:
    
    get_hist_tweets(q='modi', count=100, result_type='recent', location='bangalore', distance='100km', days=7)
        It will return last 7 days tweets 
        https://dev.twitter.com/rest/reference/get/search/tweets
        get tweets with given keyword, maximum count per tweet
        q : keywords. or query parameter
        result_type : recent/popular
        location : location name
        distance : distance from location
        count : maximum count for topic related search(max is 100)
        days : number of days to cover in twitter stream
    


#### get_blog_data_historical


```python
help(taf.get_blog_data_historical)
```

    Help on function get_blog_data_historical in module text_analytics_fns:
    
    get_blog_data_historical(keyword=['modi', 'bjp'], location='India', sort=None, timeduration=10)
        return historical tabular output from webhose
        keyword:  api will look for all keywords in text space
        location: country detail
        sort : refer https://docs.webhose.io/v1.0/docs/get-parameters
        timeduration : duration in number of days
    


#### get_blog_data_realtime


```python
help(taf.get_blog_data_realtime)
```

    Help on function get_blog_data_realtime in module text_analytics_fns:
    
    get_blog_data_realtime(keyword=['modi', 'bjp'], location='India', sort=None)
        return realtime tabular output from webhose
        keyword:  api will look for all keywords in text space
        location: country detail
        sort : refer https://docs.webhose.io/v1.0/docs/get-parameters
    


#### get_sentiment_affn


```python
help(taf.get_sentiment_affn)
```

    Help on function get_sentiment_affn in module text_analytics_fns:
    
    get_sentiment_affn(text)
        return vector of sentiments
        text : english text for which sentiment have to be defined
    


#### get_sentiment_dnn


```python
help(taf.get_sentiment_dnn)
```

    Help on function get_sentiment_dnn in module text_analytics_fns:
    
    get_sentiment_dnn(text)
        return vector of sentiments
        text : english text for which sentiment have to be defined
    


#### get_sentiment_on_blog


```python
help(taf.get_sentiment_blog)
```

    Help on function get_sentiment_blog in module text_analytics_fns:
    
    get_sentiment_blog(op_all, keywords)
        develop analytics in webhose data
        op_all : output dataframe from webhose
    


#### get_sentiment_stat


```python
help(taf.get_sentiment_stat)
```

    Help on function get_sentiment_stat in module text_analytics_fns:
    
    get_sentiment_stat(text, method='affin')
        return statistics of sentiment 
        text : english text for which sentiment have to be defined
        method : affin or dnn
    


#### get_sentiment_statistics


```python
help(taf.get_sentiment_statistics)
```

    Help on function get_sentiment_statistics in module text_analytics_fns:
    
    get_sentiment_statistics(plot_title, tweets_df, plot=True)
        develop statistics for twitter feed used for notebook only
        plot_title : title for plots
        tweets_df : dataframe of tweets 
        plot : bool, return plot or not
    


#### get_tfidf_keywords


```python
help(taf.get_tfidf_keywords)
```

    Help on function get_tfidf_keywords in module text_analytics_fns:
    
    get_tfidf_keywords(corpus, scoring_method='mean')
        returns keywords score based on scoring method provided
        scoring_method=mean/sum/
    


#### get_tweets


```python
help(taf.get_tweets)
```

    Help on function get_tweets in module text_analytics_fns:
    
    get_tweets(keywords=['modi', 'NDA', 'kejriwal', 'iot'], time_limit=5, location='India')
        returns tweets for keywords list passed
        keywords: tweets filtering criteria
        time_limit= time limit in seconds for which stream has to be play
        location : location for which filtering have to be done
    


#### get_twitter_trend


```python
help(taf.get_twitter_trend)
```

    Help on function get_twitter_trend in module text_analytics_fns:
    
    get_twitter_trend(location_name)
        returns trending twitter topic w.r.t to a location from twitter
        location_name: name of location
    


#### preprocess_webhose_op


```python
help(taf.preprocess_webhose_op)
```

    Help on function preprocess_webhose_op in module text_analytics_fns:
    
    preprocess_webhose_op(json_op)
        returns webhose output in preprocess 
        json_op: json_op from webhose data funtion
    


#### summary_text


```python
help(taf.summarize_text)
```

    Help on function summarize_text in module text_analytics_fns:
    
    summarize_text(x, summarizer='SumBasicSummarizer', SENTENCES_COUNT=10)
        returns summarized output of text
        x: list of sentences
        summarizer=['LsaSummarizer','LuhnSummarizer',       'EdmundsonSummarizer','LexRankSummarizer','TextRankSummarizer',      'SumBasicSummarizer','KLSummarizer']
    


#### sentiment_tag


```python
help(taf.sentiment_tag)
```

    Help on function sentiment_tag in module text_analytics_fns:
    
    sentiment_tag(sentiment_val)
        sentiment_val : returns sentiment_tag based on sentiment value
    


#### plot_twitter_stats


```python
help(taf.plot_twitter_stats)
```

    Help on function plot_twitter_stats in module text_analytics_fns:
    
    plot_twitter_stats(tweets_df)
        visualization of tweets dataframe(made for jupyter notebook)
        tweets_df : tweets dataframe
    


#### clean_tweets


```python
help(taf.clean_tweets)
```

    Help on function clean_tweets in module text_analytics_fns:
    
    clean_tweets(tweets_df)
        return clean tweets 
        tweets_df : tweet dataframe
    

