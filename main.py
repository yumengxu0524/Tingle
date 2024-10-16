from analysis.analysis import calculate_daily_scores, track_score_changes, accumulate_scores, get_top_traits
from processing.trait_processing import expand_dictionary_with_synonyms
from utils.utils import load_nlp_model
from analysis.context_preparation import prepare_context, prepare_context_for_agent_2
from analysis.chatgpt_integration import get_analysis_from_chatgpt, determine_tone, follow_up_questions

import datetime

if __name__ == "__main__":
    diary_entries = {
        datetime.date(2024, 6, 20): ["I hope I will be able to confide everything to you..."],
        datetime.date(2024, 6, 21): ["Writing in a diary is a really strange experience..."],
        datetime.date(2024, 7, 8): ["Dearest Kitty, Our entire family was surprised yesterday..."],
        datetime.date(2024, 8, 3): ["Dearest Kitty, The van Daans arrived on the seventh..."],
        datetime.date(2024, 10, 1): ["Dear Kitty, So much has happened it's as if the world..."],
        datetime.date(2024, 10, 16): ["Dear Kitty, Mr. Dussel and I had another of our differences..."],
    }

    # Initialize NLP model and dictionary
    nlp_model = load_nlp_model()
    expanded_dict = expand_dictionary_with_synonyms()

    # Define thresholds
    positive_threshold = 5
    negative_threshold = -1
    relativeness_threshold = 0.2

    # Perform the analysis
    daily_scores = calculate_daily_scores(diary_entries, nlp_model, expanded_dict, relativeness_threshold, positive_threshold, negative_threshold)
    score_changes = track_score_changes(daily_scores)
    accumulated_scores = accumulate_scores(daily_scores)
    top_traits = get_top_traits(accumulated_scores)

    # Prepare context and obtain analysis
    context_agent_1 = prepare_context(diary_entries, daily_scores, accumulated_scores, top_traits)
    latest_entry_date = max(diary_entries.keys())
    tone = determine_tone(daily_scores[latest_entry_date])
    initial_analysis = get_analysis_from_chatgpt(context_agent_1, tone)
    
    print("ChatGPT Analysis:\n", initial_analysis)

    context_agent_2 = prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits)
    follow_up_questions(initial_analysis, diary_entries, daily_scores, score_changes, accumulated_scores, top_traits)

