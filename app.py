
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

#since this an tsv file using pd.read_table or we can use read_csv also the separotor as '\t'
messages = pd.read_table('SMSSpamCollection', names=["label", "message"])

#print(messages.head())

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#print(stop_words)

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ', messages['message'][i]) #removing any other values than normal english alphabets
    review = review.lower() # making it to lower case
    review = review.split() # spliting it into each individual words
    #review = [ps.stem(word) for word in review if word not in stop_words] #using stemming
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words] #using lemmatizer

    review = ' '.join(review) #joining back as a sentence
    corpus.append(review)

# transforming to integer values
y = pd.get_dummies(messages['label'])
#print(y.head())
#dropping ham column

y = y.drop('ham', axis=1)

#print(y.shape)
#print(y.head())

#Creating Bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

#data after performing bag of words
new_data = pd.DataFrame(X,columns=cv.get_feature_names())

print(new_data.head())
print(cv.get_params())

#Train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(new_data,y,test_size= 0.20,random_state=33)

#Training the model using Multinomial Naive bayes algorithm

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train,y_train.values.ravel())

y_pred = spam_detect_model.predict(X_test)

from sklearn import metrics
accuracy_score = metrics.accuracy_score(y_test,y_pred)
print(accuracy_score)
cm = metrics.confusion_matrix(y_test,y_pred)
print(cm)


