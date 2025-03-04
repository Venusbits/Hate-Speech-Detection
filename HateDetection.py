#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("HateSpeechData.csv")


# In[4]:


df


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df["labels"]=df["class"].map({0: "Hate Speech",
                            1: "Offensive Language",
                            2: "No hate or Offensive Language"})


# In[9]:


df


# In[10]:


data= df[["tweet","labels"]]


# In[11]:


data


# In[12]:


import re
import nltk
nltk.download('stopwords')

# In[13]:


#importing stopwords
from nltk.corpus import stopwords
stopwords=set(stopwords.words("english"))


# In[14]:


#importing stemming
stemmer=nltk.SnowballStemmer("english")


# In[15]:


#!pip install string


# In[16]:


#data cleaning
import string
def clean_data(text):
    text=str(text).lower()
    text=re.sub('https/://\S+|www\.S+', '',text)
    text=re.sub('\[.*?\]','',text)
    texxt=re.sub('<-*?>+','',text)
    text=re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    #stopwords removal
    text=[word for word in text.split(' ') if word not in stopwords]
    text=" ".join(text)
    #stemming the text
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[17]:


data["tweet"]= data["tweet"].apply(clean_data)


# In[18]:


data


# In[19]:


X=np.array(data["tweet"])
Y=np.array(data["labels"])


# In[20]:


X


# In[21]:


Y


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[23]:


cv= CountVectorizer()
X= cv.fit_transform(X)


# In[24]:


X


# In[25]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
X_train


# In[26]:


#building out ML model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)


# In[31]:


dt_pred=dt.predict(X_test)
dt_pred


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test, dt_pred)


# In[34]:


#confusionmatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,dt_pred)
cm


# In[39]:


import seaborn as sns
import matplotlib.pyplot as ply
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


sns.heatmap(cm, annot=True,fmt=".1f",cmap="YlGnBu")


# In[35]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
rf_pred


# In[36]:


rf_accuracy = accuracy_score(Y_test, rf_pred)
rf_accuracy


# In[37]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
svm_pred = svm.predict(X_test)
svm_pred


# In[38]:


svm_accuracy = accuracy_score(Y_test, svm_pred)
svm_accuracy


# In[ ]:




