import pickle

import re
from nltk.stem.porter import PorterStemmer
corpus = []
review = re.sub('[^a-zA-Z]', ' ', 'Subject: rig sale because the original dash was based upon a $ 10 million ( gross ) sale price in a deal that fell apart , rac is requiring another dash since the sale to scf will be for $ 9 million gross . the delta net to ena is negative $ . 5 million . dash will be coming to you today . we are set to sign final documents tomorrow . thanks . dick .')
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = ' '.join(review)
corpus.append(review)

weights = pickle.load(open('CVWeights.sav', 'rb'))
X=weights.transform(corpus).toarray()

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
result = loaded_model.predict(X)
if result == 'ham':
    print('Harmless')
else:
    print('Phishing')   