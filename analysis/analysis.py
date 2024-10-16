from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import datetime

# Function to calculate daily emotional scores, incorporating the consolidated module
def calculate_daily_scores(diary_entries, nlp_model, expanded_dict):
    daily_scores = {}
    cumulative_scores = defaultdict(float)  # Initialize cumulative scores for each trait

    for entry_date, entries in sorted(diary_entries.items()):
        day_score = defaultdict(float)
        for entry in entries:
            sentence_vector = nlp_model(entry).vector
            filtered_dict = filter_dict_by_relativeness(expanded_dict, sentence_vector, relativeness_threshold)
            impacting_traits = find_all_similar_traits_with_sentence(sentence_vector, filtered_dict, positive_threshold, negative_threshold)

            # Calculate current impact for the day
            for need, divisions in impacting_traits.items():
                for division, traits in divisions.items():
                    for trait, details in traits.items():
                        for _, _, weighted_similarity in details['similar_words']:
                            day_score[trait] += weighted_similarity

        # Calculate days since entry for age-dependent decay
        days_since_entry = (datetime.date.today() - entry_date).days
        
        # Apply consolidated_module with age-dependent decay
        adjusted_day_score = {}
        for trait_name, current_score in day_score.items():
            cumulative_score = cumulative_scores[trait_name]
            adjusted_score = consolidated_module(trait_name, current_score, cumulative_score, days_since_entry, base_decay=0.8)
            adjusted_day_score[trait_name] = adjusted_score
            cumulative_scores[trait_name] = adjusted_score  # Update cumulative score with adjusted value

        # Store the adjusted cumulative scores for the day
        daily_scores[entry_date] = dict(cumulative_scores)  # Make a copy for each day's cumulative state

    return daily_scores

# Function to track changes in emotional scores
def track_score_changes(daily_scores):
    sorted_dates = sorted(daily_scores.keys())
    previous_day_scores = None
    score_changes = {}
    for date in sorted_dates:
        if previous_day_scores is None:
            score_changes[date] = {trait: 0 for trait in daily_scores[date]}
        else:
            day_changes = {}
            for trait, score in daily_scores[date].items():
                previous_score = previous_day_scores.get(trait, 0)
                day_changes[trait] = score - previous_score
            score_changes[date] = day_changes
        previous_day_scores = daily_scores[date]
    return score_changes

def accumulate_scores(daily_scores):
    accumulated_scores = defaultdict(float)
    for scores in daily_scores.values():
        for trait, score in scores.items():
            accumulated_scores[trait] += score
    return accumulated_scores

def get_top_traits(accumulated_scores, num_traits=5):
    # Sort traits by accumulated scores in descending order and pick the top 'num_traits'
    sorted_traits = sorted(accumulated_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_traits[:num_traits]
