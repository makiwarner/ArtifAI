def get_best_sentence(user_input, sentences, nlp):
    """
    Fallback method for finding the best matching sentence from the general text.
    """
    if not sentences:
        return None
    user_doc = nlp(user_input)
    best_score = 0.0
    best_sentence = None
    for sentence in sentences:
        sentence_doc = nlp(sentence)
        score = user_doc.similarity(sentence_doc)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    if best_score > 0.6:
        return best_sentence
    return None