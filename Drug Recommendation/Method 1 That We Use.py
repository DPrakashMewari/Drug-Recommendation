#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
stops = set(stopwords.words("english")) 


# In[67]:


df=pd.read_csv("Drugs_Reviews.csv")


# In[68]:


df.head()


# In[69]:


df.shape


# In[70]:


dataset=df[['drugName','review','rating','condition','usefulCount']].copy()


# In[71]:


dataset.head()


# In[72]:


dataset.isnull().sum()


# In[73]:


dataset.dropna(inplace=True)


# In[74]:


dataset.shape


# In[75]:


#Removing Spaces


# In[76]:


blanks=[]
for i,dn,rv,rt,cn,uc in dataset.itertuples():  # iterate over the DataFrame
    if (type(rv)==str or type(dn)==str or type(cn)==str):            # avoid NaN values
        if rv.isspace() or dn.isspace() or dn.isspace() or cn.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

dataset.drop(blanks, inplace=True)


# In[77]:


dataset.shape


# In[78]:


from sklearn.preprocessing import MinMaxScaler


# In[79]:


scaler=MinMaxScaler()


# In[80]:


dataset[['rating','usefulCount']]=scaler.fit_transform(dataset[['rating','usefulCount']])


# In[81]:


dataset.head()


# In[82]:


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk


# In[83]:



def remove_numbers(review):
    return re.sub('[^A-Za-z]',' ',review)

def to_lower(review):
    return review.lower()  

def tokenize(review):
    return nltk.tokenize.word_tokenize(review)

def lemmatizer(review):
    
    wnl=WordNetLemmatizer()
    review=[wnl.lemmatize(word) for word in review if not word in stops]
    review=' '.join(review)
    
    return review


# In[84]:


dataset['processed_review']=dataset['review'].apply(remove_numbers)


dataset['processed_review']=dataset['processed_review'].apply(to_lower)


dataset['processed_review']=dataset['processed_review'].apply(tokenize)


dataset['processed_review']=dataset['processed_review'].apply(lemmatizer)


# In[85]:


corpus=[]
lenp=[]
for i,dn,rv,rt,cn,uc,pr in dataset.itertuples():  # iterate over the DataFrame
    corpus.append(pr)
    lenp.append(len(pr))
    
 
        


# In[86]:


max(lenp)


# In[87]:


dataset.head()


# In[88]:


rr=''    
for i in corpus:
    rr+=i+' '
wordcloud=WordCloud().generate(rr)  
plt.imshow(wordcloud)


# In[89]:


tags=nltk.pos_tag(corpus[0].split())
nltk.ne_chunk(tags)


# In[90]:


import nltk
nltk.download('vader_lexicon')


# In[91]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[92]:


sid=SentimentIntensityAnalyzer()


# In[93]:


dataset['scores'] = dataset['processed_review'].apply(lambda review: sid.polarity_scores(review))



# In[96]:


dataset.head()


# In[97]:


dataset['compound']  = dataset['scores'].apply(lambda score_dict: score_dict['compound'])

dataset.head()


# In[98]:


dataset['comp_score'] = dataset['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

dataset.head()


# In[101]:


X=dataset['processed_review']


# In[52]:





# In[100]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()


# In[102]:


## TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


# In[103]:


y=dataset['comp_score']


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[36]:


from sklearn.naive_bayes import GaussianNB


# In[37]:


classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[38]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred)


# In[45]:


pd.DataFrame(classification_report(y_test, y_pred,output_dict=True))


# In[42]:


from sklearn.linear_model import LogisticRegression


# In[43]:


lr_classifier=LogisticRegression()
lr_classifier.fit(X_train, y_train)


# In[44]:


y_pred = classifier.predict(X_test)


# In[ ]:


condition=input('Enter The condition')


# In[ ]:


condition


# In[ ]:


df_drugs=dataset[dataset['condition'].str.lower().str.contains(condition.lower())]


# In[ ]:


df_drugs['Valuation']=df_drugs['rating']+df_drugs['usefulCount']+df_drugs['compound']


# In[ ]:


df_drugs.sort_values(by='Valuation',ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




