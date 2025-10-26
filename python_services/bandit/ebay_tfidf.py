from .title_vectorizer import TitleVectorizer
from .text_utils import normalize_title

class EbayTfidf:
    def __init__(self, max_features=10000):
        self.title_vectorizer = TitleVectorizer(max_features=max_features)
        self.features = max_features

    def fit(self, records):
        return self.title_vectorizer.fit(records)
    
    def transform(self, records):
        pass

    def transform_batch(self, records):
        pass

    def save_state(self, filepath):
        pass

    def load_state(self, filepath):
        pass

    def extract_features(self, record):
        pass

    def extract_batch_features(self, record):
        pass
