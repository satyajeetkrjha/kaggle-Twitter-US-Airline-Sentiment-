# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 22:27:53 2017

@author: Satyajeet jha
"""
#import the necessary stuff
import sklearn
import matplotlib.pyplot as plt 
import pandas as pd
import numpy

#read the datset 
tweet=pd.read_csv('Tweets.csv')
tweet.head()
#looking at dataset we see that the column negativereason_gold ,airline_sentiment_gold and tweet_coord are empty and thus useless for us 
del tweet['tweet_coord']
del tweet['airline_sentiment_gold']
del tweet['negativereason_gold']
#cool , we now count the number of positive ,negative and neutrl tweets from airline_sentiment column
Mood_count=tweet['airline_sentiment'].value_counts()#Returns object containing counts of unique values.
print(Mood_count)

#we now do the visualization 
x=[10,20,30]#at these x cordinates
plt.bar(x,Mood_count)#at x=10 we get 
plt.xticks(x,['negative','neutral','positive'],degree=45)
plt.xlabel('x-coordinte')#
plt.ylabel('Mood of tweets count')#y-coordinate is labelled this 
plt.title('Count of Moods')#this will give title to graph
#on visulaiztion we see that there are more negtive tweets in comparsion to other tweets 
airline_count=tweet['airline'].value_counts()#this tells us about number of tweets from united airwys ,virgin etc,
print(airline_count)

#now we look at which airline user has done more neagtive tweets
#https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
#we write a function and pass the name of each airline 
def fun(Airline):
    data_frame=tweet[tweet['airline']==Airline]#so u created a dataframe with the airline that will be passed on to the function
    count=data_frame['airline_sentiment'].value_counts()#we now count the number of positive and negative tweets by this airline by looking at airline_sentiment column
    x=[10,20,30]#at x=10 ,20 and 30 we get negative neutral and positive tweets plotted
    plt.bar(x,count)
    plt.xticks(x,['negative','neutral','positive'])
    plt.ylabel('Mood Count')
    plt.xlabel('Mood')
    plt.title('Mood count for airline  '+Airline)
plt.figure(figsize=(15,15))    
plt.subplot(2,3,1)    
fun('US Airways')
plt.subplot(2,3,2)
fun('United')    
plt.subplot(2,3,3)
fun('American')
plt.subplot(2,3,4)
fun('Southwest')
plt.subplot(2,3,5)
fun('Delta')
plt.subplot(2,3,6)
fun('Virgin America')
plt.show()
#we see that Us airways ,United and American had tweets which were more negative but forb the other three ,they contain ech kind of tweet
negative_dict=dict(tweet['negativereason'].value_counts(sort=False))#this is a dictionary with a key and a map
negative_count=tweet['negativereason'].value_counts()
print(negative_dict)

def fun1(Airline):
    if 'Airline'=='All':#in this case we take the entire tweet dataframe as our datfrme
        data_frame=tweet
    else:
         data_frame=tweet[tweet['airline']==Airline]#else we take out this column
    count=dict(data_frame['negativereason'].value_counts()) #we creted a dictionary where key is the negative reason and value is the number of  tweets which says the same reason
    #print(count)
    Unique_reason=list(tweet['negativereason'].unique())#we get a list of all the unique negtive reasons .We willl use it for data visualization part
    #print(Unique_reason)#we print the listed but our list conatains nan as well and we
    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']#we removed nan from our list.
    #print(Unique_reason)
    Reason_frame=pd.DataFrame({'Reasons':Unique_reason})
    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])
    #print(Reason_frame)#our reason frame conations reasons and count of each reason
    return Reason_frame 

def plottingpart(Airline):
    data_frame=fun1(Airline)
    #print(data_frame)#this our reason data frame 
    #note instead of [10,20,30] as indexes we want to use reason as our axis for data visualization
    count=data_frame['count']
    print(count)
    Index = range(1,(len(data_frame)+1))
    plt.bar(Index,count)
    plt.xticks(Index,data_frame['Reasons'],rotation=90)
    plt.ylabel('Count')
    plt.xlabel('Reason')
    plt.title('Count of Reasons for '+Airline)
        
plottingpart('US Airways')    #we see that it has more customer service issue
plottingpart('United')#it has more customer service issue and late flight
plottingpart('American')#customer service issue is main issue 
plottingpart('Southwest')#customer service  issue is the main reason here 
plottingpart('Delta')   #late flight is main reason (leding)alongwith customer service issue
plottingpart('Virgin America')#customer service issue is main issue here
#preprocessing part 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#now we convert tweet to words nd we ue stopword here in this function
def tweet_to_words(inputtweet):
     letters_only = re.sub("[^a-zA-Z]", " ",inputtweet) #now our tweet doesn't conatin symobols and are just english letters 
     print(letters_only)
     words = letters_only.lower().split()#we convert all to lowerscase and use split
     print(words)
     stops = set(stopwords.words("english"))                  
     meaningful_words = [w for w in words if not w in stops] 
     return( " ".join( meaningful_words )) 
 #this function returns the length of tweet and is same as above
def tweet_length(inputtweet):
    letters_only = re.sub("[^a-zA-Z]", " ",inputtweet) #now our tweet doesn't conatin symobols and are just english letters 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words)) 
   
 
tweet['sentiment']=tweet['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)   #we now got a new column in our twet table where sentiment is 1 if positive else 0
tweet['clean_tweet']=tweet['text'].apply(lambda x: tweet_to_words(x))#we created a new column 'clean-teet' which stoores the text but i
tweet['Tweet_length']=tweet['text'].apply(lambda x: tweet_length(x))
from sklearn.model_selection import train_test_split
train,test = train_test_split(tweet,test_size=0.2,random_state=42)

train_clean_tweet=[]
for tweet in train['clean_tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['clean_tweet']:
    test_clean_tweet.append(tweet)
    
#now we do the vectoriztion thing .
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(analyzer = "word")
train_features= v.fit_transform(train_clean_tweet)
test_features=v.transform(test_clean_tweet)    

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
     
#instead of writing every single classifier  as done in titanic dataset ,we take it all together
Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    GaussianNB()]
#we use above classifiers in our dataset
dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))    




 
    
    


    
    
    
    

