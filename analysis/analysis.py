from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import datetime

def calculate_daily_scores(diary_entries, nlp_model, expanded_dict, relativeness_threshold, positive_threshold, negative_threshold):
    daily_scores = {}
    for entry_date, entries in sorted(diary_entries.items()):
        day_score = defaultdict(float)
        for entry in entries:
            sentence_vector = nlp_model(entry).vector
            filtered_dict = filter_dict_by_relativeness(expanded_dict, sentence_vector, relativeness_threshold)
            impacting_traits = find_all_similar_traits_with_sentence(sentence_vector, filtered_dict, positive_threshold, negative_threshold)
            for need, divisions in impacting_traits.items():
                for division, traits in divisions.items():
                    for trait, details in traits.items():
                        day_score[trait] += sum(weighted_similarity for _, _, weighted_similarity in details['similar_words'])
        daily_scores[entry_date] = day_score
    return daily_scores

def track_score_changes(daily_scores):
    prev_scores = None
    score_changes = {}
    for date, scores in daily_scores.items():
        if prev_scores:
            score_changes[date] = {trait: scores.get(trait, 0) - prev_scores.get(trait, 0) for trait in scores}
        else:
            score_changes[date] = {trait: 0 for trait in scores}
        prev_scores = scores
    return score_changes

def accumulate_scores(daily_scores):
    accumulated_scores = defaultdict(float)
    for scores in daily_scores.values():
        for trait, score in scores.items():
            accumulated_scores[trait] += score
    return accumulated_scores

def get_top_traits(accumulated_scores, num_traits=5):
    return sorted(accumulated_scores.items(), key=lambda item: item[1], reverse=True)[:num_traits]
