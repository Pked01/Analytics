def text2sentiment_heatmap(text,sentiments_list=None):
    """
    text: raw text for which sentiment has to be analyzed
    sentiment list: sentiment of each sentence
    """
    # tokenizing into sentences
    if sentiments_list is not None:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        tokenized_text=tokenizer.tokenize(text)


        #getting sentiment vectors
        sentiments_vec=model.transform(tokenized_text)

        # getting sentiment value from classifier neuron
        neuron_index=2388
        sentiments=[i[neuron_index] for i in sentiments_vec]
    else:
        sentiments =sentiments_list

    # splitting sentences into words and diving sentiments thus word to sentiment mapping
    # this is done solely for purpose of plotting
    words=[]
    values=[]
    for idx,sent in enumerate(tokenized_text):
        word_li=tokenized_text[idx].split(" ")
        words+=word_li
        values+=[sentiments[idx]]*len(word_li)
    del word_li

    #plotting heatmap for reviews
    num_words=10
    # words_df=[]
    # values_df=[]
    for i in range(0,len(words),num_words):
        #words_df.append(words[i:i+num_words])
        #values_df.append(values[i:i+num_words])
        fig, ax = plt.subplots(figsize=(num_words*2,0.5))
        ax=sns.heatmap(np.array(values[i:i+num_words]).reshape(1,len(values[i:i+num_words]))\
                       ,fmt='',annot=np.array(words[i:i+num_words]).reshape(1,len(words[i:i+num_words]))\
                       ,annot_kws={"size":15}, vmin=-1, vmax=1, cmap='RdYlGn')

    # words_df=pd.DataFrame(words_df)
    # values_df=pd.DataFrame(values_df)
