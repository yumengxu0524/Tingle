def get_analysis_from_chatgpt(context_agent_1, tone):
    # ChatGPT integration to get analysis
    pass

def determine_tone(daily_scores):
    if 'Happiness' in daily_scores and daily_scores['Happiness'] > 15:
        return "enthusiastic and positive"
    elif 'Sadness' in daily_scores and daily_scores['Sadness'] > 15:
        return "empathetic and gentle"
    else:
        return "neutral and informative"

def follow_up_questions(initial_analysis, diary_entries, daily_scores, score_changes, accumulated_scores, top_traits):
    print("Follow-up questions based on analysis.")

