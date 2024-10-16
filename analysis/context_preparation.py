def prepare_context(diary_entries, daily_scores, accumulated_scores, top_traits):
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
    
    context_agent_1 += "\nAnalysis:\nPlease provide a psychological analysis based on the above."
    return context_agent_1

def prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits):
    context_agent_2 = "Psychological Analysis of Diary Entries:\n\n"
    context_agent_2 += "Day-to-Day Score Changes:\n"
    for date, scores in daily_scores.items():
        context_agent_2 += f"{date}: {scores}\n"
    
    context_agent_2 += "\nAccumulated Trait Scores:\n"
    for trait, score in accumulated_scores.items():
        context_agent_2 += f"{trait}: {score}\n"
    
    context_agent_2 += "\nTop Traits:\n"
    for trait, score in top_traits:
        context_agent_2 += f"{trait}: {score}\n"

    context_agent_2 += "\nAnalysis:\nPlease provide insights based on the above."
    return context_agent_2
