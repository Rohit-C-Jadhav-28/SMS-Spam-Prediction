# Import Seaction... ✔️
import re
import pandas
import numpy
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Helper Function...
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r"@\w+", '', text)
    # Remove hashtags but keep the text
    text = re.sub(r"#(\w+)", r'\1', text)
    # Remove special characters and numbers
    text = re.sub(r"[^A-Za-z\s]", '', text)
    # Remove extra spaces
    text = re.sub(r"\s+", ' ', text).strip()
    return text.lower()

# Data Fatching
'''
    1. Encode it in Latin-1.
    2. Drop The unwanted column. 
    3. Change the Name of the Title.
'''
Data_Path = r'Data\spam.csv'
dataframe = pandas.read_csv(Data_Path,encoding='latin1') 
drop_column_name = ['Unnamed: 2','Unnamed: 3','Unnamed: 4']
dataframe = dataframe.drop(drop_column_name, axis=1)
Category = dataframe['v1'].unique()
dataframe.rename(columns={'v1': 'Target', 'v2':'Text'}, inplace=True)
dataframe.head()


# Text Preprocessing.
'''
    1. Label Encoder the Convert Ham : 0 & Spam : 1 
    2. Cleane the text
    3. Convert it Into Lower Case.
    4. apply to every single row's of Perticular Column
'''
LE = LabelEncoder()
dataframe['Target_LE'] = LE.fit_transform(dataframe['Target'])
dataframe.head()

dataframe['Cleaned_Text'] = dataframe['Text'].apply(lambda x : clean_text(x))


# Vectorizing...
'''
    1. Convert the Text into Number 
    2. Convert it into toarray() function.
    3. Create model of TFIDF Vectorizer.
'''
Tfidf = TfidfVectorizer()
Vectorize = Tfidf.fit_transform(dataframe['Cleaned_Text']).toarray()
pickle.dump(Tfidf, open('Spam_SMS_Tfidf.pkl', 'wb'))


# Train Test Split...
X = Vectorize
y = dataframe['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

Classifiers = [KNeighborsClassifier(), LogisticRegressionCV(), LogisticRegression(), RandomForestClassifier(), MultinomialNB()]

Classifiers_Name = ['KNeighborsClassifier', 'LogisticRegressionCV', 'LogisticRegression', 'RandomForestClassifier', 'MultinomialNB']

Accuracy_list = list()

i = 0
for classifier in Classifiers:
    Fitting = classifier.fit(X_train, y_train)
    class_prediction = Fitting.predict(X_test)
    Accuracy = accuracy_score(y_test, class_prediction)
    Accuracy_list.append(Accuracy)
    pickle.dump(Fitting, open(Classifiers_Name[i]+'.pkl', 'wb'))
    i = i+1

for i in range(len(Accuracy_list)-1):
    print(f'{Classifiers_Name[i]} Accuracy :: {Accuracy_list[i]}')