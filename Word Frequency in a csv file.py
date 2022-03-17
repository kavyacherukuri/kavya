#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import datetime
import time
import csv


# In[6]:


#to convert create date into month date format
def convertTime(date_to_convert):
     return datetime.datetime.strptime(date_to_convert, '%a %b %d %H:%M:%S +0000 %Y').strftime('%b %d')


# In[12]:


def mainFunction():
    
    #--------------------------------------------------Task 1-----------------------------------------------------
    #reading amazon file into string
    tweets = pd.read_csv('Amazon_Responded_Oct05.csv', on_bad_lines='skip') 
    og_tweets = tweets #storing the original copy for task 2
    
    #Step-1 : removing records which are not verified
    tweets = tweets.drop(tweets.index[tweets['user_verified'] == False])
    print('The number of records which are user_verified: ',len(tweets))
    
    #Step-2 : group by create date and getting the max count 
    tweets['created_date'] = tweets.tweet_created_at.apply(convertTime) #adding create date column
    max_tweets = tweets
    countByDate = tweets.groupby('created_date').count()
    maxTweetsValue = countByDate[['id_str']].idxmax()[0] #getting the max number of tweets date.
    print('\nMax Tweets date: ',maxTweetsValue)
    
    #Step-3:
    #Data cleaning:
    #filtering the max tweets date
    max_tweets = max_tweets.drop(max_tweets.index[tweets['created_date'] != 'Jan 03'])
    #getting the total count
    max_tweets['total_count'] = max_tweets['favorite_count'] + max_tweets['retweet_count']
    #getting the top 100 total count tweets
    max_tweets = max_tweets.sort_values(by='total_count', ascending=False).head(100) 
    tweet_values= max_tweets.text_
    
    #calcuating the word frequency:
    words = []
    #splitting the sentences in text_ column into list of words
    for tweet in tweet_values:
        new_words = tweet.split()
        words.extend(new_words)
        
    #removing the the user name, end tag which starts with '^' and the url   
    words = [x for x in words if not x.startswith('@') and not x.startswith('https://') and not x.startswith('^')]
    
    nText = []
    #remvoing the characters and numbers form the list of words and converting everything to lower case
    for word in words:
        getVals = list([val for val in word if val.isalpha() or '#' in val])
        result = ("".join(getVals)).lower()
        nText.append(result)
    nText = list(filter(None,nText))
    
    uniqueWords = list(set(nText)) #getting a list of unique words
    wordCount =  pd.DataFrame(uniqueWords)
    wordCount.rename(columns = {0: 'Word'}, inplace= True)
    wordCount["Frequency"] = ""
    #getting the frequency of each unique word
    for i in range(len(uniqueWords)):
        wordCount.iloc[i]['Frequency'] = nText.count(wordCount.loc[i]['Word']) 
    #sorting the frequencies in descending order
    wordCount = wordCount.sort_values(by='Frequency', ascending=False)
    print('\nTask 1 final output: display the frequency of words \n', wordCount)
    
    #Writing the data in the csv file
    wordCount.to_csv('wordFrequency.csv',index=False)
    
    #----------------------------------------------------------------------------------------------------
    #-----------------------------------------Task: 2-------------------------------------------------
    
    #loading the findtext into a dataframe
    findText =  pd.read_csv('find_text.csv', on_bad_lines='skip')
    findText.text = findText.text.fillna('')
    
    #dropping all the columns except id_str and text from amazon data set.
    og_tweets = og_tweets.drop(columns=['tweet_created_at', 'user_screen_name', 'user_id_str',
       'user_statuses_count', 'user_favourites_count', 'user_protected',
       'user_listed_count', 'user_following', 'user_description',
       'user_location', 'user_verified', 'user_followers_count',
       'user_friends_count', 'user_created_at', 'tweet_language', 'text_',
       'favorite_count', 'favorited', 'in_reply_to_screen_name',
       'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'retweet_count',
       'retweeted'])
    og_tweets.text = og_tweets.text.fillna('')
    
    #getting the common id_str from both the files and assigning text to it
    mergedValues = pd.merge(og_tweets,findText,on='id_str')
    mergedValues = mergedValues.drop(columns=['text_y'])
    mergedValues.rename(columns = {'text_x': 'text'}, inplace= True)
    
    print('\nTask 2: add text to find_text file based on the id_str \n', mergedValues)
    
    #Writing the data in the csv file
    mergedValues.to_csv('find_text.csv',index=False)
    
    
mainFunction()


# In[ ]:




