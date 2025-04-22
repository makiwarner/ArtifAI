import re
import unicodedata

def clean_text(text):
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text
