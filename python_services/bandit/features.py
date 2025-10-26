import json
import hashlib
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .title_vectorizer import TitleVectorizer
from .text_utils import normalize_title, create_mock_ebay_title

class RecordFeatureExtractor:
    def __init__(self, 
                 artist_vocab_size = 1000,
                 label_vocab_size = 500,
                 genre_vocab_size = 100,
                 style_vocab_size = 200,
                 embedding_dims = None,
                 title_tfidf_features=10000):
        
        self.artist_vocab_size = artist_vocab_size
        self.label_vocab_size = label_vocab_size
        self.genre_vocab_size = genre_vocab_size
        self.style_vocab_size = style_vocab_size

        self.embedding_dims = embedding_dims or {
            'artist': 64,
            'label': 32,
            'genre': 16,
            'style': 16
        }

        self.title_vectorizer = TitleVectorizer(max_features = title_tfidf_features)
        self.title_tfidf_features = title_tfidf_features

        self.numerical_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()

        self.min_year = None
        self.max_year = None
        self.is_fitted = False

    def get_vocab_sizes(self):
        return {
            'artist': self.artist_vocab_size,
            'label': self.label_vocab_size,
            'genre': self.genre_vocab_size,
            'style': self.style_vocab_size
        }
    
    def get_embedding_dims(self):
        return self.embedding_dims.copy()
    
    def get_feature_info(self):
        return {
            'vocab_sizes': self.get_vocab_sizes(),
            'embedding_dims': self.get_embedding_dims(),
            'categorical_features': 8,  # artist + label + 3 genres + 3 styles
            'numerical_features': 3,   # wants, haves, ratio
            'other_features': 10,      # price + year + condition (8 dims)
            'total_features': 21  
        }
    
    def save_state(self, filepath):
        if not self.is_fitted:
            raise ValueError("Must call fit() before saving state")
        
        state = {
            'vocab_sizes': self.get_vocab_sizes(),
            'embedding_dims': self.embedding_dims,
            'numerical_scaler': self.numerical_scaler,
            'price_scaler': self.price_scaler,
            'min_year': self.min_year,
            'max_year': self.max_year,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.numerical_scaler = state['numerical_scaler']
        self.price_scaler = state['price_scaler'] 
        self.min_year = state['min_year']
        self.max_year = state['max_year']
        self.is_fitted = state['is_fitted']

        if state['vocab_sizes'] != self.get_vocab_sizes():
            raise ValueError("Loaded vocab sizes don't match current configuration")

    def _hash_to_index(self, text, vocab_size):
        if not text or not isinstance(text, str):
            return 0
        
        clean_text = text.lower().strip()
        if not clean_text: return 0

        hash_object = hashlib.md5(clean_text.encode('utf-8'))
        hash_integer = int(hash_object.hexdigest(), 16)

        vocab_index = (hash_integer % (vocab_size - 1)) + 1

        return vocab_index 
    
    def _get_artist_index(self, artist):
        return self._hash_to_index(artist, self.artist_vocab_size)
    
    def _get_label_index(self, label: str) -> int:
        """Get vocabulary index for label"""
        return self._hash_to_index(label, self.label_vocab_size)
    
    def _get_genre_indices(self, genres, max_genres):
        """Get vocabulary indices for genres (up to max_genres)"""
        if not genres:
            return [0] * max_genres
        
        # Take first max_genres, hash each one
        genre_indices = []
        for i in range(max_genres):
            if i < len(genres):
                idx = self._hash_to_index(genres[i], self.genre_vocab_size)
                genre_indices.append(idx)
            else:
                genre_indices.append(0)  # Pad with zeros
        
        return genre_indices
    
    def _get_style_indices(self, styles, max_styles = 3):
        """Get vocabulary indices for styles (up to max_styles)"""
        if not styles:
            return [0] * max_styles
        
        # Take first max_styles, hash each one
        style_indices = []
        for i in range(max_styles):
            if i < len(styles):
                idx = self._hash_to_index(styles[i], self.style_vocab_size)
                style_indices.append(idx)
            else:
                style_indices.append(0)  # Pad with zeros
        
        return style_indices
    
    def _parse_price(self, price_str):
        if not price_str or not isinstance(price_str, str):
            return 0.0
    
        try:
            price_clean = price_str.replace('$', '').replace('€', '').replace('£', '')
            price_clean = price_clean.replace('USD', '').replace('EUR', '').replace('GBP', '')          
            price_parts = price_clean.split(',')
            price_numeric = price_parts[0].strip()
            
            if not price_numeric:
                return 0.0
                
            return float(price_numeric)
            
        except (ValueError, AttributeError, IndexError):
            return 0.0
        
    def _normalize_year(self, year):
        if year is None or year <= 1900 or year > 2025 or self.min_year is None or self.max_year is None:
            return 0.5
        
        year_range = self.max_year - self.min_year
        if year_range == 0:
            return 0.5
        
        clamped = max(self.min_year, min(self.max_year, year))
        return (clamped - self.min_year) / year_range 
    
    def _extract_numerical_features(self, record):
        wants = record.get('wants', 0)
        haves = record.get('haves', 0)

        ratio = wants / max(haves, 1) if haves > 0 else wants

        if self.is_fitted:
            features_scaled = self.numerical_scaler.transform([[wants, haves, ratio]])[0]
            return features_scaled.tolist()
        else:
            return [wants, haves, ratio]
        
    def _encode_condition(self, condition):
        conditions = [
            'Poor', 'Fair', 'Good', 'Good Plus', 
            'Very Good', 'Very Good Plus', 'Near Mint', 'Mint'
        ]
    
        if not condition or not isinstance(condition, str):
            # Return all zeros for missing condition
            return [0.0] * len(conditions)
        
        # Normalize and expand abbreviations
        condition_clean = condition.strip()
        condition_clean = condition_clean.replace('(', '').replace(')', '')
        
        # Common abbreviation mappings
        abbreviations = {
            'P': 'Poor',
            'F': 'Fair', 
            'G': 'Good',
            'G+': 'Good Plus',
            'VG': 'Very Good',
            'VG+': 'Very Good Plus', 
            'NM': 'Near Mint',
            'M': 'Mint'
        }
        
        # Check for exact abbreviation match first
        for abbrev, full in abbreviations.items():
            if condition_clean == abbrev:
                condition_clean = full
                break
        
        # Create one-hot vector
        one_hot = [0.0] * len(conditions)
        
        # Find best match (case insensitive)
        for i, cond in enumerate(conditions):
            if cond.lower() in condition_clean.lower():
                one_hot[i] = 1.0
                break
        
        return one_hot

    def _process_genre_array(self, genres):
        max_genres = 3
        genre_indices = []
        
        if not genres or not isinstance(genres, list):
            return [0] * max_genres
        
        # Take first max_genres, hash each one
        for i in range(max_genres):
            if i < len(genres) and genres[i]:
                idx = self._hash_to_index(genres[i], self.genre_vocab_size)
                genre_indices.append(idx)
            else:
                genre_indices.append(0)  # Pad with zeros for missing genres
        
        return genre_indices

    def _process_style_array(self, styles):
        max_styles = 3
        style_indices = []
        
        if not styles or not isinstance(styles, list):
            return [0] * max_styles
        
        # Take first max_styles, hash each one  
        for i in range(max_styles):
            if i < len(styles) and styles[i]:
                idx = self._hash_to_index(styles[i], self.style_vocab_size)
                style_indices.append(idx)
            else:
                style_indices.append(0)  # Pad with zeros for missing styles
        
        return style_indices
    
    def fit(self, records):
        if not records: raise ValueError("NO RECORDS")

        numerical_features = []
        prices = []
        years = []
        normalized_titles = []

        for record in records:
            wants = record.get('wants', 0)
            haves = record.get('haves', 0)
            ratio = wants / max(haves, 1) if haves > 0 else wants
            numerical_features.append([wants, haves, ratio])

            price = self._parse_price(record.get('record_price', ''))
            prices.append(price)

            year = record.get('year')
            if year is not None:
                years.append(year)
            if record.get('_is_ebay'):
                norm_title = normalize_title(record.get('ebay_title', ''))
            else:
                norm_title = create_mock_ebay_title(record)
            if norm_title: 
                normalized_titles.append(norm_title)

        if numerical_features:
            self.numerical_scaler.fit(numerical_features)
        if prices:
            self.price_scaler.fit([[p] for p in prices])
        if years:
            self.min_year = min(years)
            self.max_year = max(years)
        else:
            self.min_year = 1950
            self.max_year = 2025

        if normalized_titles:
            print(f"\nFitting TF-IDF on {len(normalized_titles)} titles...")
            self.title_vectorizer.fit(normalized_titles)


        self.is_fitted = True
        print(f"Feature extractor fitted on {len(records)} records")

    def extract_features(self, record):
        if not self.is_fitted: raise ValueError("FIT FIRST")

        features = []

        artist_idx = self._get_artist_index(record.get('artist', ''))
        label_idx = self._get_label_index(record.get('label', ''))
        genre_indices = self._process_genre_array(record.get('genres', []))
        style_indices = self._process_style_array(record.get('styles', []))

        features.extend([artist_idx, label_idx])
        features.extend(genre_indices)
        features.extend(style_indices)

        numerical_features = self._extract_numerical_features(record)
        features.extend(numerical_features)

        price = self._parse_price(record.get('record_price', ''))
        price_norm = self.price_scaler.transform([[price]])[0][0]
        features.append(price_norm)

        year_norm = self._normalize_year(record.get('year'))
        features.append(year_norm)

        condition_features = self._encode_condition(record.get('media_condition', ''))
        features.extend(condition_features)

        if record.get('_is_ebay'):
            norm_title = normalize_title(record.get('ebay_title', ''))
        else:
            norm_title = create_mock_ebay_title(record)
        
        title_tfidf = self.title_vectorizer.transform(norm_title)
        features.extend(title_tfidf)

        return np.array(features, dtype=np.float32)
    
    def extract_batch_features(self, records):
        if not self.is_fitted: raise ValueError("FIT FIRST")

        feature_vectors = []
        for record in records:
            features = self.extract_features(record)
            feature_vectors.append(features)
        
        return np.vstack(feature_vectors)