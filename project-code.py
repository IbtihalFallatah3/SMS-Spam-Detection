#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# # Load the Dataset

# In[5]:


df = pd.read_csv(r"C:\Users\remas alhazmi\OneDrive - University of Prince Mugrin\Desktop\DM\spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df.head()


# # Encode Labels

# In[7]:


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# # Visualize Class Distribution

# In[9]:


sns.countplot(x='label', data=df)
plt.title("Distribution of Spam vs Ham")
plt.xticks([0,1], ['Ham', 'Spam'])
plt.show()


# # Pie Chart

# In[16]:


labels = ['Ham', 'Spam']
sizes = df['label'].value_counts()
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(5, 5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Message Distribution")
plt.axis('equal')
plt.show()


# # WordCloud

# In[21]:


def clean_text(msg):
    msg = msg.lower()
    msg = re.sub(r'\d+', '', msg)
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    msg = msg.strip()
    stop_words = set(stopwords.words('english'))
    msg = ' '.join([word for word in msg.split() if word not in stop_words])
    return msg

df['cleaned'] = df['message'].apply(clean_text)


# In[23]:


from wordcloud import WordCloud

ham_text = " ".join(df[df['label'] == 0]['cleaned'])
spam_text = " ".join(df[df['label'] == 1]['cleaned'])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=400, height=300, background_color='white').generate(ham_text))
plt.title("Ham Messages Word Cloud")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=400, height=300, background_color='black', colormap='Pastel1').generate(spam_text))
plt.title("Spam Messages Word Cloud")
plt.axis("off")

plt.tight_layout()
plt.show()


# # Text Preprocessing

# In[9]:


def clean_text(msg):
    msg = msg.lower()
    msg = re.sub(r'\d+', '', msg)  
    msg = msg.translate(str.maketrans('', '', string.punctuation))   
    msg = msg.strip()
    stop_words = set(stopwords.words('english'))
    msg = ' '.join([word for word in msg.split() if word not in stop_words])
    return msg

df['cleaned'] = df['message'].apply(clean_text)
df[['message', 'cleaned']].head()


# # Vectorization with TF-IDF

# In[10]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']


# # Split Dataset

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train

# In[15]:


# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


# # Evaluate

# In[16]:


def evaluate_model(name, y_true, y_pred):
    print(f"🔹 {name} Evaluation:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("-" * 50)

evaluate_model("Naive Bayes", y_test, nb_preds)
evaluate_model("Logistic Regression", y_test, lr_preds)
evaluate_model("Random Forest", y_test, rf_preds)


# In[17]:


results = pd.DataFrame({'message': df['message'][y_test.index],
                        'actual': y_test,
                        'predicted': lr_preds})
wrong = results[results['actual'] != results['predicted']]
wrong.head()


# In[ ]:




