import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import streamlit as st
import nltk
from nltk.corpus import stopwords
import random

#loading the dataset
data_fake=pd.read_csv('C:/cogentinfo/nlp-learning/fake-news/data/Fake.csv')
data_true=pd.read_csv('C:/cogentinfo/nlp-learning/fake-news/data/True.csv')

#previewing data
#st.subheader("Previewing datasets:")

#st.write(data_fake)

data_fake["class"]=0
data_true['class']=1
#st.write(data_fake.shape)

#st.write(data_true)
#st.write(data_true.shape)

data_merge=pd.concat([data_fake, data_true], axis = 0)
#st.subheader("After merging datasets:")
#st.write(data_merge)
#st.write(data_merge.shape)

#"title", "subject" and "date" columns is not required for detecting the fake news, so I am going to drop the columns.
data=data_merge.drop(['title','subject','date'], axis = 1)

#random shuffling the dataset
data = data.sample(frac = 1)
#st.subheader("After shuffling:")

#indexing the shuffled dataset in order
data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)
#st.write(data)


#PRE PROCESSING TEXT
#Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

data['text'] = data['text'].apply(wordopt)

# Download stopwords from nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(data, text_column):
    
    if text_column not in data.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")
    
    def clean_text(text):
        if isinstance(text, str):
            words = text.split()
            cleaned_text = ' '.join(word for word in words if word.lower() not in stop_words)
            return cleaned_text
        else:
            return text

    data[text_column] = data[text_column].apply(clean_text)
    data = data.dropna(subset=[text_column])
    data = data[data[text_column].str.strip().astype(bool)]
    return data
data = remove_stopwords(data, 'text')



#Defining dependent and independent variable as x and y
x = data['text']
y = data['class']



st.subheader("dataset after preprocessing text:")
st.write(data)

#Training the model
#Splitting the dataset into training set and testing set.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

#Extracting Features from the Text
#Convert text to vectors

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#LOGISTIC REGRESSION CLASSIFIER

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)
st.subheader("Logistic Regression Report")

report = classification_report(y_test, pred_lr, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the classification report in Streamlit

st.dataframe(report_df)


#DECISION TREEE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
red_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)

st.subheader("Decision Tree Report")

report = classification_report(y_test, red_dt, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the classification report in Streamlit

st.dataframe(report_df)


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)

pred_rf = RF.predict(xv_test)
RF.score(xv_test, y_test)

st.subheader("Random forest Report")

report = classification_report(y_test, pred_rf, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the classification report in Streamlit

st.dataframe(report_df)


#TESTING THE MODEL

def output_label(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return (
        f"LR Prediction: {output_label(pred_LR[0])}\n"
        f"DT Prediction: {output_label(pred_DT[0])}\n"
        f"RF Prediction: {output_label(pred_RF[0])}"
    )

#Model Testing With random Entry

st.title("Fake News Detection")


random_index = random.randint(0, len(data) - 1)
news = data.iloc[random_index]['text']
st.write(f"Random News: {news}")
result = manual_testing(news)
st.write(result)
