import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize_lemmatize(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
