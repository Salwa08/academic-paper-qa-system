#### Metrcis for BM25-based Q&A system
### This file will contain functions to evaluate the performance of the BM25 keyword search approach.
# Possible metrics:
# Hit rate: Did the answer appear in top-3 chunks? (Yes/No)
# Average position: Where did relevant info first appear?
# Response time: How fast was retrieval?


def evaluate_bm25(qa_system, test_questions, relevant_keywords):
    """Evaluate BM25 Q&A system using test questions and relevant keywords.
    
    Args:
        qa_system: Initialized BM25 Q&A system.
        test_questions: List of test questions.
        relevant_keywords: Dict mapping questions to relevant keywords.
    
    Returns:
        Dict with evaluation metrics.
    """

    total_questions = len(test_questions)
    hit_count = 0
    total_position = 0
    total_response_time = 0.0

    for question in test_questions:
        result = qa_system.answer_with_timing(question, top_k=3)
        answer = result['answer']
        response_time = result['response_time']
        total_response_time += response_time
        keywords = relevant_keywords.get(question, [])
        hit = False
        position = 4  # Default to 4 (not found in top-3)
        for idx, keyword in enumerate(keywords):
            if keyword.lower() in answer.lower():
                hit = True
                position = idx + 1
                break
        if hit:
            hit_count += 1
            total_position += position

    hit_rate = hit_count / total_questions if total_questions > 0 else 0
    avg_position = total_position / hit_count if hit_count > 0 else 0
    avg_response_time = total_response_time / total_questions if total_questions > 0 else 0.0
    metrics = {
        'total_questions': total_questions,
        'hit_rate': hit_rate,
        'avg_position': avg_position,
        'avg_response_time': avg_response_time
    }
    return metrics

