from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TitleVectorizer:
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=False,
            token_pattern=r'\b\w+\b',
            min_df = 2,
            max_df = 0.95,
        )
        self.fitted = False

    def fit(self, titles):
        self.vectorizer.fit(titles)
        self.fitted = True
        print(f"✓ TF-IDF fitted on {len(titles)} titles")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        if len(self.vectorizer.vocabulary_) > 0:
            feature_names = self.vectorizer.get_feature_names_out()
            idfs = self.vectorizer.idf_
            top_idx = np.argsort(idfs)[-20:]  # Top 20 most distinctive
            print(f"✓ Most distinctive keywords: {', '.join(feature_names[top_idx])}")

    def transform(self, title):
        if not self.fitted: raise ValueError("FIT FIRST")
        if title is None or not title.strip(): 
            return np.zeros(len(self.vectorizer.vocabulary_))
        vector = self.vectorizer.transform([title]).toarray()[0]
        return vector
    
    def transform_batch(self, titles):
        if not self.fitted: raise ValueError("FIT FIRST")
        valid = [title if title is not None else "" for title in titles]
        vectors = self.vectorizer.transform(valid).toarray()
        return vectors

