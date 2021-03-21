import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer # Portuguese stemmer

nltk.download('stopwords')
nltk.download('rslp')
portuguese_stopwords = stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()

def process_text(text):
    text = ''.join([c for c in text if c not in string.punctuation]) # Remove punctuation
    text = ''.join([c for c in text if not c.isdigit()]) # Remove numbers
    text = text.lower().strip() # Lowercase and strip
    
    # remove stopwords and stem
    text = [stemmer.stem(word) for word in text.split() if word not in portuguese_stopwords] 
    return ' '.join(text)

def make_predict(clf, vectorizer):
    def predict(text):
        transformed_text = process_text(text)
        transformed_text = vectorizer.transform([transformed_text])
        return clf.predict(transformed_text)[0]
    return predict