from collections import defaultdict

def determine_tone(daily_scores):
    if 'Happiness' in daily_scores and daily_scores['Happiness'] > 15:
        return "enthusiastic and positive"
    elif 'Sadness' in daily_scores and daily_scores['Sadness'] > 15:
        return "empathetic and gentle"
    else:
        return "neutral and informative"

def generate_reflective_prompt(trait):
    prompts = {
        "Cautiousness": "It seems like you've been thinking a lot about safety and risk. Could you tell me more about what’s been on your mind?",
        "Self-Esteem": "You mentioned some thoughts on self-worth recently. What do you feel most proud of?",
        "Happiness": "It’s great to see moments of joy. What’s something that brought a smile to your face recently?"
    }
    return prompts.get(trait, "Can you share a bit more about your thoughts and feelings?")

def suggest_resources(trait, self_actualization_words):
    for need, divisions in self_actualization_words.items():
        for division, traits in divisions.items():
            if trait in traits:
                return f"For the trait '{trait}', consider exploring activities or resources that align with personal growth in this area."
    return "Reflecting on your experiences can help you grow. Keep it up!"

def calculate_weekly_summary(daily_scores):
    weekly_summary = defaultdict(float)
    for scores in daily_scores.values():
        for trait, score in scores.items():
            weekly_summary[trait] += score
    for trait in weekly_summary:
        weekly_summary[trait] /= 7  # Average score over the week
    return dict(weekly_summary)

def provide_goal_suggestions(top_traits):
    suggestions = {
        "Self-Esteem": "Set a goal that challenges you and reinforces your sense of accomplishment.",
        "Happiness": "Engage in activities that bring you joy and make them a regular part of your routine.",
        "Resilience": "Consider learning a new skill to build on your strengths."
    }
    return [suggestions.get(trait, "Keep up the great work!") for trait, _ in top_traits]

def get_top_traits_with_resources(accumulated_scores, self_actualization_words, num_traits=5):
    sorted_traits = sorted(accumulated_scores.items(), key=lambda item: item[1], reverse=True)[:num_traits]
    top_traits_with_resources = [(trait, score, suggest_resources(trait, self_actualization_words)) for trait, score in sorted_traits]
    return top_traits_with_resources
    
    # Adding top traits with resource suggestions
    top_traits_with_resources = get_top_traits_with_resources(accumulated_scores, original_self_actualization_words)
    context_agent_2 += "\nTop Traits with Resource Suggestions:\n"
    for trait, score, suggestion in top_traits_with_resources:
        context_agent_2 += f"{trait} (Score: {score}): {suggestion}\n"

def prepare_context_for_agent_1(diary_entries, daily_scores, accumulated_scores, top_traits):
    context_agent_1 = "Psychological Analysis of Diary Entries:\n\n"
    context_agent_1 += "Day-to-Day Score Changes:\n"
    for date, scores in daily_scores.items():
        context_agent_1 += f"{date}: {scores}\n"
    
    context_agent_1 += "\nAccumulated Trait Scores:\n"
    for trait, score in accumulated_scores.items():
        context_agent_1 += f"{trait}: {score}\n"
    
    context_agent_1 += "\nTop Traits:\n"
    for trait, score in top_traits:
        context_agent_1 += f"{trait}: {score}\n"
    
    context_agent_1 += "\nAnalysis:\nPlease provide a psychological analysis of the diary writer based on the above information."
    return context_agent_1

# Prepare context for ChatGPT
def prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits):
    context_agent_2 = "Psychological Analysis of Diary Entries:\n\n"
    context_agent_2 += "Day-to-Day Score Changes:\n"
    for date, scores in daily_scores.items():
        context_agent_2 += f"{date}: {scores}\n"
    
    context_agent_2 += "\nAccumulated Trait Scores:\n"
    for trait, score in accumulated_scores.items():
        context_agent_2 += f"{trait}: {score}\n"
    
    # Adding top traits with resource suggestions
    context_agent_2 += "\nTop Traits with Resource Suggestions:\n"
    for trait, score, suggestion in top_traits:
        context_agent_2 += f"{trait} (Score: {score}): {suggestion}\n"

    # Adding weekly summary
    weekly_summary = calculate_weekly_summary(daily_scores)
    context_agent_2 += "\nWeekly Summary:\n"
    for trait, score in weekly_summary.items():
        context_agent_2 += f"{trait}: {score}\n"
    
    # Adding goal suggestions
    goal_suggestions = provide_goal_suggestions(top_traits)
    context_agent_2 += "\nGoals and Recommendations:\n"
    for suggestion in goal_suggestions:
        context_agent_2 += f"{suggestion}\n"        
               
        
    context_agent_2 += "\nAnalysis:\nPlease provide a psychological analysis of the diary writer based on the above information."
    return context_agent_2


