def get_best_faq_answer(user_input, faq_map, nlp):
    """
    Checks the FAQ mapping for the best matching question and returns its answer 
    if the similarity score is above the threshold.
    """
    if not faq_map:
        return None
    user_doc = nlp(user_input)
    best_score = 0.0
    best_answer = None
    for entry in faq_map:
        question_doc = nlp(entry["question"])
        score = user_doc.similarity(question_doc)
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]
    if best_score > 0.6:
        return best_answer
    return None