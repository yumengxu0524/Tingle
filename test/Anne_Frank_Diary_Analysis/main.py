from analysis.analysis import calculate_daily_scores, track_score_changes, accumulate_scores, get_top_traits
from processing.trait_processing import expand_dictionary_with_synonyms
from utils.utils import load_nlp_model
from analysis.context_preparation import prepare_context, prepare_context_for_agent_2
from analysis.chatgpt_integration import get_analysis_from_chatgpt, determine_tone, follow_up_questions

import datetime

if __name__ == "__main__":
    diary_entries = {
        datetime.date(2024, 6, 20): ["I hope I will be able to confide everything to you, as I have never been able to confide in anyone, and I hope you will be a great source of comfort and support."],
        datetime.date(2024, 6, 21): ["Writing in a diary is a really strange experience for someone like me. Not only because I've never written anything before, but also because it seems to me that later on neither I nor anyone else will be interested in the musings of a thirteen-year-old schoolgirl. Oh well, it doesn't matter. I feel like writing."],
        datetime.date(2024, 7, 8): ["Dearest Kitty, Our entire family was surprised yesterday morning by the sudden announcement of my sister Margot that she had received a call-up notice from the SS. Fortunately, it's only Margot who's been called up, but that doesn't mean that I won't get one too. Daddy already has a plan to hide us in the building of the company he works for."],
        datetime.date(2024, 8, 3): ["Dearest Kitty, The van Daans arrived on the seventh. This morning, when I was upstairs making the beds, the van Daans' son, Peter, came in. Peter is almost sixteen, a shy, awkward boy whose company won't amount to much, I thought. Yet, who knows, maybe he'll be a pleasant comrade. He's currently the only young person here."],
        datetime.date(2024, 10, 1): ["Dear Kitty, So much has happened it's as if the whole world had suddenly turned upside down. But as you see, Kitty, I'm still alive, and that's the main thing, Father says. I'm alive all right, but don't ask where or how. You probably don't understand that I'm alive more in spite of everything than because of it."],
        datetime.date(2024, 10, 16): ["Dear Kitty, Mr. Dussel and I had another of our little differences yesterday. I must honestly admit that I don't like him much; he's pedantic, argumentative, and clumsy, old-fashioned, and just a little bit boring. I'm always running into him whenever I turn around, and it annoys me."],
        datetime.date(2024, 10, 16): ["Dear Kitty, I've reached the point where I hardly care whether I live or die. The world will keep on turning without me, and I can't do anything to change events anyway. I'll just let matters take their course and concentrate on studying and hope that everything will be all right in the end."]
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
    context_agent_1 = prepare_context_for_agent_1(diary_entries, daily_scores, accumulated_scores, top_traits)
    latest_entry_date = max(diary_entries.keys())
    tone = determine_tone(daily_scores[latest_entry_date])
    initial_analysis = get_analysis_from_chatgpt(context_agent_1, tone)
    
    print("ChatGPT Analysis:\n", initial_analysis)

    context_agent_2 = prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits)
    follow_up_questions(initial_analysis, diary_entries, daily_scores, score_changes, accumulated_scores, top_traits)
