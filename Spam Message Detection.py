import spacy
import pandas as pd 

nlp = spacy.load('en_core_web_sm')

df = pd.read_csv('train.csv')

df.head()

def lemmatization(text):
  doc = nlp(text)
  lemmalist = [token.lemma_ for token in doc]
  return ' '.join(lemmalist)

df['lemma']=df['sms'].apply(lemmatization)

df.head()

def remove_stopwords(text):
  doc = nlp(text)
  no_stopwords = [token.text for token in doc if not token.is_stop and not token.is_punct]
  return ' '.join(no_stopwords)

df['preprocessed'] = df['lemma'].apply(remove_stopwords)

df.head()

X = df['preprocessed']
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


X_train.shape, X_test.shape

!pip install --upgrade scikit-learn

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

model = Pipeline([
    ('cvectorizer_tfidf', TfidfVectorizer()),
    ('Random Forest', RandomForestClassifier())              
])

model.fit(X_train, y_train)

model.score(X_test, y_test) * 100

pred = model.predict(X_test)

y_test[:5]

pred[:5]

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))

import seaborn as sns
sns.set_style('darkgrid')

cf = confusion_matrix(y_test, pred, normalize = 'true')
sns.heatmap(cf, annot=True, cmap = 'Greens')
