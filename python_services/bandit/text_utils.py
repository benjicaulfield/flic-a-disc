import re

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'this', 'but', 'or', 'records', 'record',
    'vinyl', 'lp', 'album', 'rare', 'grail', 'import', 'promo', 'pressing', 
    'press', 'reissue', 'original', 'og', 'limited', 'edition', 'sealed', 'new',
    'read', 'plz', 'please', 'look', 'mint', 'nm', 'near', 'vg', 'ex', 
    'excellent', 'good', 'fair', 'poor', 'better', 'best', 'condition', 
    'grade', 'graded', '12', '7', '10', '45', 'inch', 'albums'
}

def normalize_title(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    filtered = set([word for word in words if word not in STOP_WORDS and len(word) > 1])
    return ' '.join(filtered)

def create_mock_ebay_title(discogs_record):
    parts = []
    if discogs_record.get('artist'): parts.append(discogs_record['artist'])    
    if discogs_record.get('title'): parts.append(discogs_record['title'])
    if discogs_record.get('label'): parts.append(discogs_record['label'])
    if discogs_record.get('year') and discogs_record['year'] != 0:
        parts.append(str(discogs_record['year']))
    raw = ' '.join(parts)
    return normalize_title(raw)