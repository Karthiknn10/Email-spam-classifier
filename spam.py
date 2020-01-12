# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:59:58 2020

@author: Karthi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("emails.csv")
df.describe()
df.info()
#23% spam so balanced dataset
#no null values


#looking how spam and ham looks like
sample_spam = df["text"][df["spam"]==1].head(5)
#lots of misspellings, words like click, money, dollar, some offer kinda thing repeated
#no emphasis on punctuation

sample_ham = df["text"][df["spam"]==0].head(5)

##Getting the length of messages
df['length'] = df['text'].apply(len)
df['length'].max()
#some mail contains 44000 characters!
len("Hello world")

df['length'].plot(bins=100, kind='hist') 
#but many contains less than 10000 characters
describe = df.length.describe()
#min - 13 characters

# Let's see the longest message 43952
df[df['length'] == 43952]['text'].iloc[0]
df[df['length'] == 43952]['spam'].iloc[0]

#Dividing dataframe into spam and ham
ham = df[df['spam']==0]
spam = df[df['spam']==1]


spam['length'].plot(bins=40, kind='hist')
plt.title("Spam distribution")

spam['length'].plot(bins=40, kind='hist')
plt.title("Ham distribution")

#almost same kind of distribution

sns.countplot(df['spam'], label = "Count") 

#Removing punctuation
import string
string.punctuation

sample = 'Loyalty is a two way goddamn @$($ street!!'
sample_punc_removed = [char for char in sample if char not in string.punctuation]
sample_punc_removed

#joining the characters to form a string without punctuation
sample_wo_punc = "".join(sample_punc_removed)

#Removing stopwords
from nltk.corpus import stopwords
stopwords.words('english')

#removing stopwords from the above sentence
sample_clean = [word for word in sample_wo_punc.split() \
                if word.lower() not in stopwords.words("english")]
print(sample_clean)

#Implementing punctuation, stop words in the dataframe
def clean_words(text):
    sample_punc_removed =[char for char in text if char not in string.punctuation]
    sample_wo_punc = "".join(sample_punc_removed)
    sample_clean = [word for word in sample_wo_punc.split() \
                if word.lower() not in stopwords.words("english")]
    return sample_clean

df_clean = df["text"].apply(clean_words)

#Counting occurences of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = clean_words)
spamham_countvectorizer = vectorizer.fit_transform(df['text'])
spamham_countvectorizer.toarray()

X = spamham_countvectorizer
X.shape
type(X)
#n>> (5728,37229)
y = df['spam'].values

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_train = model.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

y_predict_test = model.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

#Report
print(classification_report(y_test, y_predict_test))
