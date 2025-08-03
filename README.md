# SMS-Spam-Detection-using-NLP-Machine-Learning
📩 SMS Spam Detection using NLP & Machine Learning
This notebook demonstrates how to build an end-to-end SMS spam classification system using Natural Language Processing and Machine Learning.

1️⃣ Data Preprocessing
Loaded dataset with pandas
Checked for null values and duplicates
Removed duplicates to clean the dataset
2️⃣ Text Cleaning
Converted text to lowercase
Removed special characters, numbers, and punctuation
Tokenized messages into individual words
Removed stopwords using nltk.corpus.stopwords
Applied stemming using PorterStemmer
3️⃣ Exploratory Data Analysis
Analyzed the distribution of spam vs ham messages
Visualized message lengths and word counts
4️⃣ Word Cloud Visualization
Created word clouds separately for:
Spam messages
Ham messages
Helped visualize the most frequently used words in each category
5️⃣ Text Vectorization (Feature Extraction)
Tried two vectorizers:
CountVectorizer() → Bag of Words
TfidfVectorizer(max_features=2000) ✅
Chose TF-IDF due to better results and controlled vocabulary size
6️⃣ Model Training & Evaluation
Tested 3 Naive Bayes models:

GaussianNB()
MultinomialNB() ✅ (Best performer)
BernoulliNB()
Best Performance (MultinomialNB + TF-IDF):

Accuracy: 97.67%
Precision: 1.00
🏁 Conclusion
Applied real-world NLP techniques: tokenization, stopword removal, stemming, vectorization
MultinomialNB with TF-IDF outperformed other models with high accuracy and perfect precision
📚 Citation
This notebook was created as a learning project inspired by this YouTube tutorial.

Importing Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk 
import seaborn as sns
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")
spam_data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin1')
spam_data.head()
v1	v2	Unnamed: 2	Unnamed: 3	Unnamed: 4
0	ham	Go until jurong point, crazy.. Available only ...	NaN	NaN	NaN
1	ham	Ok lar... Joking wif u oni...	NaN	NaN	NaN
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...	NaN	NaN	NaN
3	ham	U dun say so early hor... U c already then say...	NaN	NaN	NaN
4	ham	Nah I don't think he goes to usf, he lives aro...	NaN	NaN	NaN
spam_data.shape
(5572, 5)
spam_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   v1          5572 non-null   object
 1   v2          5572 non-null   object
 2   Unnamed: 2  50 non-null     object
 3   Unnamed: 3  12 non-null     object
 4   Unnamed: 4  6 non-null      object
dtypes: object(5)
memory usage: 217.8+ KB
Data Cleaning
spam_data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
spam_data.head()
v1	v2
0	ham	Go until jurong point, crazy.. Available only ...
1	ham	Ok lar... Joking wif u oni...
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
3	ham	U dun say so early hor... U c already then say...
4	ham	Nah I don't think he goes to usf, he lives aro...
spam_data.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
spam_data.head()
target	text
0	ham	Go until jurong point, crazy.. Available only ...
1	ham	Ok lar... Joking wif u oni...
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
3	ham	U dun say so early hor... U c already then say...
4	ham	Nah I don't think he goes to usf, he lives aro...
Encoding Target
encoder = LabelEncoder()
spam_data['target'] = encoder.fit_transform(spam_data['target']) # ham = 0 & spam = 1
spam_data.head()
target	text
0	0	Go until jurong point, crazy.. Available only ...
1	0	Ok lar... Joking wif u oni...
2	1	Free entry in 2 a wkly comp to win FA Cup fina...
3	0	U dun say so early hor... U c already then say...
4	0	Nah I don't think he goes to usf, he lives aro...
spam_data.isna().sum()
target    0
text      0
dtype: int64
Removing Duplicates
spam_data.duplicated().sum()
403
spam_data = spam_data.drop_duplicates(keep = 'first')
spam_data.duplicated().sum()
0
Performing EDA
spam_data['target'].value_counts()
target
0    4516
1     653
Name: count, dtype: int64
plt.pie(spam_data['target'].value_counts(), labels = ['ham', 'spam'], autopct = '%0.2f')
plt.title('Data is Imbalanced')
plt.show()

nltk.download('punkt')
[nltk_data] Downloading package punkt to /usr/share/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
True
spam_data['number_of_characters'] = spam_data['text'].apply(len)
spam_data.head()
target	text	number_of_characters
0	0	Go until jurong point, crazy.. Available only ...	111
1	0	Ok lar... Joking wif u oni...	29
2	1	Free entry in 2 a wkly comp to win FA Cup fina...	155
3	0	U dun say so early hor... U c already then say...	49
4	0	Nah I don't think he goes to usf, he lives aro...	61
spam_data['number_of_words'] = spam_data['text'].apply(lambda x: len(nltk.word_tokenize(x)))
spam_data.head()
target	text	number_of_characters	number_of_words
0	0	Go until jurong point, crazy.. Available only ...	111	24
1	0	Ok lar... Joking wif u oni...	29	8
2	1	Free entry in 2 a wkly comp to win FA Cup fina...	155	37
3	0	U dun say so early hor... U c already then say...	49	13
4	0	Nah I don't think he goes to usf, he lives aro...	61	15
spam_data['number_of_sentences'] = spam_data['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
spam_data[['number_of_characters', 'number_of_words', 'number_of_sentences']].describe()
number_of_characters	number_of_words	number_of_sentences
count	5169.000000	5169.000000	5169.000000
mean	78.977945	18.455794	1.965564
std	58.236293	13.324758	1.448541
min	2.000000	1.000000	1.000000
25%	36.000000	9.000000	1.000000
50%	60.000000	15.000000	1.000000
75%	117.000000	26.000000	2.000000
max	910.000000	220.000000	38.000000
sns.histplot(spam_data[spam_data['target'] == 0]['number_of_characters'])
sns.histplot(spam_data[spam_data['target'] == 1]['number_of_characters'], color='red')
<Axes: xlabel='number_of_characters', ylabel='Count'>

sns.histplot(spam_data[spam_data['target'] == 0]['number_of_words'])
sns.histplot(spam_data[spam_data['target'] == 1]['number_of_words'], color='red')
<Axes: xlabel='number_of_words', ylabel='Count'>

sns.pairplot(spam_data, hue='target')
<seaborn.axisgrid.PairGrid at 0x7bc4a78bb950>

sns.heatmap(spam_data.corr(numeric_only = True), annot = True)
<Axes: >

Data Preprocessing
### Lower Case
### Tokenization
### Removing stop words and punctuation
### Stemming
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # print(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y).strip()
transform_text(spam_data['text'][0])
'go jurong point crazi avail bugi n great world la e buffet cine got amor wat'
spam_data['transformed_text'] = spam_data['text'].apply(transform_text)
spam_data.head()
target	text	number_of_characters	number_of_words	number_of_sentences	transformed_text
0	0	Go until jurong point, crazy.. Available only ...	111	24	2	go jurong point crazi avail bugi n great world...
1	0	Ok lar... Joking wif u oni...	29	8	2	ok lar joke wif u oni
2	1	Free entry in 2 a wkly comp to win FA Cup fina...	155	37	2	free entri 2 wkli comp win fa cup final tkt 21...
3	0	U dun say so early hor... U c already then say...	49	13	1	u dun say earli hor u c alreadi say
4	0	Nah I don't think he goes to usf, he lives aro...	61	15	1	nah think goe usf live around though
wc = WordCloud(width = 500,height = 500, min_font_size = 10, background_color = 'white')
spam_wc = wc.generate(spam_data[spam_data['target'] == 1]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize=(14,6))
plt.imshow(spam_wc)
<matplotlib.image.AxesImage at 0x7bc4a2865510>

ham_wc = wc.generate(spam_data[spam_data['target'] == 0]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize=(14,6))
plt.imshow(ham_wc)
<matplotlib.image.AxesImage at 0x7bc4a4477bd0>

spam_msg = []
for sms in spam_data[spam_data['target'] == 1]['transformed_text'].tolist():
    for word in sms.split():
        spam_msg.append(word)
        
    
len(spam_msg)
9939
word_freq = Counter(spam_msg).most_common(30)
df = pd.DataFrame(word_freq, columns=['word', 'count'])
plt.figure(figsize=(12, 6))
sns.barplot(x='word', y='count', data=df)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()

ham_msg = []
for sms in spam_data[spam_data['target'] == 0]['transformed_text'].tolist():
    for word in sms.split():
        ham_msg.append(word)
        
    
word_freq = Counter(ham_msg).most_common(30)
df = pd.DataFrame(word_freq, columns=['word', 'count'])
plt.figure(figsize=(12, 6))
sns.barplot(x='word', y='count', data=df)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()

Model Building
Text Vectorization
# Here TfidfVectorizer along with max_features = 3000 works better then CountVectorizer.
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 2000)
X = tfidf.fit_transform(spam_data['transformed_text']).toarray()
X.shape
(5169, 2000)
y = spam_data['target'].values
Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
Since we're working with text data, Naive Bayes models are a popular choice. So here, I’m experimenting with three variants: GaussianNB, MultinomialNB, and BernoulliNB.
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
Gaussian Naive Bayes
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,y_pred1))
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred1))
print("Precision Score: ", precision_score(y_test,y_pred1))
Accuracy Score:  0.851063829787234
Confusion Matrix:  [[765 131]
 [ 23 115]]
Precision Score:  0.46747967479674796
Bernoulli Naive Bayes
bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,y_pred3))
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred3))
print("Precision Score: ",precision_score(y_test,y_pred3))
Accuracy Score:  0.9864603481624759
Confusion Matrix:  [[895   1]
 [ 13 125]]
Precision Score:  0.9920634920634921
Multinomial Naive Bayes
mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,y_pred2))
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred2))
print("Precision Score: ",precision_score(y_test,y_pred2))
Accuracy Score:  0.97678916827853
Confusion Matrix:  [[896   0]
 [ 24 114]]
Precision Score:  1.0
📊 Model Evaluation Summary
In the context of spam detection, precision score is especially important because we want to avoid falsely classifying genuine (ham) messages as spam.

Among the models tested:

Multinomial Naive Bayes (MNB) achieved the best results:
Accuracy: 97.67%
Precision Score: 1.00
✅ Since MNB has a perfect precision score, it's highly reliable in minimizing false positives, making it the most suitable choice for this use case.
