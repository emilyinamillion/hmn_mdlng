import pandas as pd
from hmn_mdlng import NLPModeler
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from spacy.en import English 
import re
parser = English()

## Quick text cleaner
def cleaner(text):
    text = text.lower()
    separators = ["\r", "\n", "\t", "n't", "'m", "'ll", '[^a-z ]']
    for i in separators:
        sample = re.sub(i, " ", text)
    
    text = re.sub(r'[\w\.-]+@[\w\.-]+', "", text) #sub email addresses
    STOPLIST = set(list(stopwords.words('english')) + list(ENGLISH_STOP_WORDS))
    tokens = parser(sample)
    tokens = [tok.lemma_.strip() for tok in tokens]

    return " ".join([tok for tok in tokens if len(tok) != 1 and tok not in STOPLIST])

def main():
	categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
	print("running test on 20newsgroups data - consider the meaning of life while we clean the text.")
	twenty_train = fetch_20newsgroups(subset='train',
	categories=categories, shuffle=True, random_state=42)
	labels = dict(enumerate(twenty_train.target_names))
	y = [labels.get(i) for i in twenty_train.target]
	X = twenty_train.data
	df = pd.DataFrame(X, y).reset_index()
	df.columns = ["target", "text"]
	df.text = df.text.apply(cleaner)
	NLPModeler(df, X_column_label = "text", y_column_label="target")
	
if __name__ == '__main__':
	main()
