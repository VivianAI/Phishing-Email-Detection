# Natural Language Processing
# Importing the libraries
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Email-trainingdata-20k.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,9999):
    review = re.sub('[^a-zA-Z]', ' ', dataset['email'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('CVWeights.sav', 'wb'))
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = ((cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1]))*100
precision = cm[0,0]/(cm[0,0]+cm[0,1])*100
recall = cm[0,0]/(cm[0,0]+cm[1,0])*100
print('accuracy = ',accuracy, 'precision = ',precision, 'recall = ',recall)

filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

print(accuracy)
