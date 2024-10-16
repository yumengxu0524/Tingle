
# Function to get ChatGPT's response
def get_analysis_from_chatgpt(context_agent_1, tone):
    system_prompt = (
        "You are a psychologist analyzing diary entries using a framework that includes tracking emotional scores, "
        "cumulative trait scores, and identifying key psychological traits over time. The framework analyzes traits "
        "such as cautiousness, happiness, sadness, resilience, social connection, and self-esteem, and calculates changes over time. "
        "Please analyze trends in emotional states, identify recurring themes, and provide insights on the writer’s psychological traits and overall mental well-being. "
        "Focus on how the individual’s emotional state fluctuates, suggesting areas for personal growth based on the most prominent traits. "
        "Use the numerical scores provided to interpret the intensity of each trait, noting any significant increases or decreases. "
        "Finally, offer recommendations on managing stress, enhancing well-being, and nurturing positive traits, along with any resources or practices that could aid in personal development."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_agent_1}
        ]
    )
    return response.choices[0].message.content


# Function for follow-up interactions
def follow_up_questions(initial_analysis, diary_entries, daily_scores, score_changes, accumulated_scores, top_traits):
    diary_summary = "Here are the original diary entries for reference:\n"
    for date, entries in diary_entries.items():
        diary_summary += f"{date}:\n"
        for entry in entries:
            diary_summary += f"  - {entry}\n"
            
    # Prepare a comprehensive summary of diary entries and analysis details for context
    context_agent_2 = prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits)
    follow_up_prompt = generate_reflective_prompt(top_traits[0][0])  # Assume top_traits is a list of (trait, score)
    print("\nYou can now ask follow-up questions based on this analysis. Type 'exit' to end the conversation.\n")
    
    # Start interactive loop
    while True:
        user_question = input("Your question: ")
        if user_question.lower() == 'exit':
            print("Ending follow-up session.")
            break
        
        # Sending the follow-up question with full context to ChatGPT
        follow_up_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant helping answer follow-up questions on a psychological analysis of diary entries."},
                {"role": "assistant", "content": initial_analysis},
                {"role": "assistant", "content": diary_summary},
                {"role": "assistant", "content": context_agent_2},
                {"role": "user", "content": user_question}
            ]
        )
        
        # Print the response to the follow-up question
        print("ChatGPT Follow-Up Response:\n", follow_up_response.choices[0].message.content)
