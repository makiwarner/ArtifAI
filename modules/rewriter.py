import re

def rewrite_to_first_person(text, artist_name):
    name_pattern = re.escape(artist_name)
    text = re.sub(rf"\b{name_pattern}\b", "I", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b{name_pattern}'s\b", "my", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(she|he|they)\b", "I", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(her|his|their)\b", "my", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(him|them)\b", "me", text, flags=re.IGNORECASE)
    return text