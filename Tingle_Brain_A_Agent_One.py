import os
from openai import OpenAI
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="enter your OPENAI_API_KEY"
)
import datetime
import spacy
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pymongo import MongoClient

# Load the NLP model
nlp = spacy.load('en_core_web_md')
nlp_model = nlp

# MongoDB connection to retrieve diary entries
client = MongoClient("mongodb://localhost:27017")
db = client['mydatabase']
collection = db['entries']


# Function to fetch diary entries from MongoDB
def fetch_diary_entries(user_id: str):
    """Fetch all diary entries for a user from MongoDB."""
    entries = collection.find({"user_id": user_id})
    
    diary_entries = {}
    for entry in entries:
        # Convert the 'time' field from string to date object
        entry_date = datetime.strptime(entry['time'], '%Y-%m-%d').date()
        # Append the content to the respective date in diary_entries
        if entry_date not in diary_entries:
            diary_entries[entry_date] = []
        diary_entries[entry_date].append(entry['content'])
    
    return diary_entries

# Define your Trait data structure and functions
@dataclass
class Trait:
    words: list
    weight: float

# Initialize the multi-layer dictionary with default dictionaries
self_actualization_words = defaultdict(lambda: defaultdict(lambda: defaultdict(Trait)))

# Original self_actualization_words dictionary setup with initial trait definitions
original_self_actualization_words = {
    'Physiological Needs': {
        'Division 3 (Animal)': {
            'Pneumativeness': {'words': ['breathing', 'suffocation', 'air', 'respiration'], 'weight': 15},
            'Alimentiveness': {'words': ['hunger', 'appetite', 'food', 'eating'], 'weight': 15},
            'Sanativeness': {'words': ['injury', 'disease', 'healing', 'recovery'], 'weight': 12},
        },
        'Division 4 (Perceptives)': {
            'Flavor': {'words': ['taste', 'flavor', 'savory', 'sweet', 'bitter'], 'weight': 8},
            'Weight': {'words': ['balance', 'climbing', 'stability', 'equilibrium'], 'weight': 6},
        },
    },
    'Safety Needs': {
        'Division 1 (Domestic)': {
            'Inhabitiveness': {'words': ['home', 'shelter', 'safety', 'security'], 'weight': 13},
        },
        'Division 3 (Animal)': {
            'Cautiousness': {'words': ['danger', 'prudence', 'risk', 'caution'], 'weight': 12},
            'Constructiveness': {'words': ['build', 'invent', 'create', 'construct'], 'weight': 10},
        },
        'Division 4 (Perceptives)': {
            'Order': {'words': ['system', 'organization', 'structure', 'arrangement'], 'weight': 9},
            'Weight': {'words': ['balance', 'climbing', 'weighing', 'scale'], 'weight': 7},
        },
    },
    'Love and Belongingness Needs': {
        'Division 1 (Domestic)': {
            'Amativeness': {'words': ['love', 'romance', 'affection', 'intimacy'], 'weight': 14},
            'Philoprogenitiveness': {'words': ['parental', 'pets', 'children', 'nurturing'], 'weight': 12},
            'Adhesiveness': {'words': ['friendship', 'sociability', 'loyalty', 'bonding'], 'weight': 13},
            'Union for life': {'words': ['commitment', 'marriage', 'partnership', 'union'], 'weight': 11},
        },
        'Division 5 (Self Perfection)': {
            'Agreeableness': {'words': ['pleasing', 'agreeable', 'amicable', 'cordial'], 'weight': 9},
            'Human Nature': {'words': ['kindness', 'empathy', 'compassion', 'generosity'], 'weight': 10},
            'Benevolence': {'words': ['kindness', 'goodness', 'charity', 'benevolence'], 'weight': 10},
            'Imitation': {'words': ['copy', 'imitate', 'emulate', 'mimic'], 'weight': 6},
        },
        'Division 4 (Perceptives)': {
            'Language': {'words': ['speak', 'communication', 'dialogue', 'expression'], 'weight': 11},
        },
    },
    'Esteem Needs': {
        'Division 2 (Aspiring)': {
            'Self-Esteem': {'words': ['dignity', 'respect', 'pride', 'self-respect'], 'weight': 14},
            'Approbativeness': {'words': ['ambition', 'esteem', 'recognition', 'approval'], 'weight': 12},
            'Firmness': {'words': ['perseverance', 'steadfast', 'determination', 'resolve'], 'weight': 10},
            'Conscientiousness': {'words': ['duty', 'right', 'ethics', 'morality'], 'weight': 11},
        },
        'Division 5 (Self Perfection)': {
            'Agreeableness': {'words': ['pleasing', 'agreeable', 'amicable', 'cordial'], 'weight': 8},
            'Veneration': {'words': ['respect', 'revere', 'admire', 'honor'], 'weight': 10},
        },
        'Division 4 (Perceptives)': {
            'Tune': {'words': ['music', 'sing', 'melody', 'harmony'], 'weight': 7},
        },
    },
    'Cognitive Needs': {
        'Division 3 (Animal)': {
            'Constructiveness': {'words': ['build', 'invent', 'create', 'construct'], 'weight': 9},
            'Ideality': {'words': ['beauty', 'refinement', 'perfection', 'idealism'], 'weight': 8},
        },
        'Division 4 (Perceptives)': {
            'Form': {'words': ['shape', 'form', 'structure', 'contour'], 'weight': 7},
            'Individuality': {'words': ['fact', 'identity', 'uniqueness', 'self'], 'weight': 8},
            'Size': {'words': ['length', 'breadth', 'dimension', 'magnitude'], 'weight': 6},
            'Locality': {'words': ['places', 'location', 'site', 'position'], 'weight': 6},
            'Eventuality': {'words': ['events', 'history', 'occurrences', 'developments'], 'weight': 7},
            'Language': {'words': ['speak', 'communication', 'dialogue', 'expression'], 'weight': 9},
            'Color': {'words': ['color', 'hue', 'shade', 'tint'], 'weight': 6},
            'Order': {'words': ['system', 'organization', 'structure', 'arrangement'], 'weight': 8},
            'Number': {'words': ['arithmetic', 'count', 'calculate', 'numerate'], 'weight': 6},
            'Time': {'words': ['time', 'dates', 'duration', 'period'], 'weight': 7},
            'Comparison': {'words': ['compare', 'discriminate', 'differentiate', 'contrast'], 'weight': 8},
            'Causality': {'words': ['cause', 'effect', 'result', 'consequence'], 'weight': 8},
        },
        'Division 5 (Self Perfection)': {
            'Imitation': {'words': ['copy', 'imitate', 'emulate', 'mimic'], 'weight': 5},
        },
    },
    'Self-Actualization': {
        'Division 1 (Domestic)': {
            'Concentrativeness': {'words': ['focus', 'concentrate', 'attention', 'dedication'], 'weight': 11},
        },
        'Division 2 (Aspiring)': {
            'Firmness': {'words': ['perseverance', 'steadfast', 'determination', 'resolve'], 'weight': 12},
            'Conscientiousness': {'words': ['duty', 'right', 'ethics', 'morality'], 'weight': 13},
        },
        'Division 3 (Animal)': {
            'Ideality': {'words': ['beauty', 'refinement', 'perfection', 'idealism'], 'weight': 9},
            'Mirthfulness': {'words': ['wit', 'humor', 'fun', 'jocularity'], 'weight': 7},
        },
        'Division 4 (Perceptives)': {
            'Causality': {'words': ['cause', 'effect', 'result', 'consequence'], 'weight': 8},
            'Tune': {'words': ['music', 'sing', 'melody', 'harmony'], 'weight': 6},
        },
        'Division 5 (Self Perfection)': {
            'Spirituality': {'words': ['faith', 'wonder', 'spirit', 'divinity'], 'weight': 12},
            'Human Nature': {'words': ['kindness', 'empathy', 'compassion', 'generosity'], 'weight': 10},
            'Veneration': {'words': ['respect', 'revere', 'admire', 'honor'], 'weight': 11},
            'Benevolence': {'words': ['kindness', 'goodness', 'charity', 'benevolence'], 'weight': 9},
            'Creed': {'words': ['belief', 'creed', 'faith', 'doctrine'], 'weight': 10},
        },
    },
    'Transcendence': {
        'Division 5 (Self Perfection)': {
            'Spirituality': {'words': ['faith', 'wonder', 'spirit', 'divinity'], 'weight': 13},
            'Creed': {'words': ['belief', 'creed', 'faith', 'doctrine'], 'weight': 11},
        },
    },
    'Emotional Response': {
        'Division 6 (Emotions)': {
            'Anger': {'words': ['angry', 'rage', 'furious'], 'weight': 20},
            'Happiness': {'words': ['happy', 'joy', 'elated'], 'weight': 20},
            'Sadness': {'words': ['sad', 'depressed', 'down'], 'weight': 20},
            'Fear': {'words': ['fear', 'scared', 'frightened'], 'weight': 20},
            'Surprise': {'words': ['surprised', 'shocked', 'astonished'], 'weight': 20},
            'Disgust': {'words': ['disgusted', 'revolted', 'repelled'], 'weight': 20},
        }
    }
}


# Function to find synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

# Function to expand the dictionary with synonyms
def expand_dictionary_with_synonyms(self_actualization_words):
    expanded_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for need, divisions in self_actualization_words.items():
        for division, traits in divisions.items():
            for trait, data in traits.items():
                words = data['words']
                expanded_dict[need][division][trait] = {
                    'words': [],
                    'weight': data['weight']
                }
                for word in words:
                    synonyms = get_synonyms(word)
                    expanded_dict[need][division][trait]['words'].extend(synonyms)
                expanded_dict[need][division][trait]['words'] = list(set(expanded_dict[need][division][trait]['words']))
    return expanded_dict

# Initialize expanded_dict with the provided dictionary structure
expanded_dict = expand_dictionary_with_synonyms(original_self_actualization_words)

# Function to filter dictionary branches based on relativeness threshold
def filter_dict_by_relativeness(expanded_dict, sentence_vector, relativeness_threshold):
    filtered_dict = {}
    for need, divisions in expanded_dict.items():
        need_relative_score = 0
        filtered_divisions = {}
        for division, traits in divisions.items():
            division_relative_score = 0
            filtered_traits = {}
            for trait, data in traits.items():
                trait_relative_score = 0
                for word in data['words']:
                    word_vector_in_dict = nlp(word).vector
                    similarity = cosine_similarity([sentence_vector], [word_vector_in_dict])[0][0]
                    trait_relative_score += similarity
                trait_relative_score /= len(data['words'])
                if trait_relative_score >= relativeness_threshold:
                    filtered_traits[trait] = data
                    division_relative_score += trait_relative_score
            if filtered_traits:
                division_relative_score /= len(filtered_traits)
                if division_relative_score >= relativeness_threshold:
                    filtered_divisions[division] = filtered_traits
                    need_relative_score += division_relative_score
        if filtered_divisions:
            need_relative_score /= len(filtered_divisions)
            if need_relative_score >= relativeness_threshold:
                filtered_dict[need] = filtered_divisions
    return filtered_dict

# Function to find all divisions that are similar above or below the thresholds
def find_all_similar_traits_with_sentence(sentence_vector, expanded_dict, positive_threshold, negative_threshold):
    impacting_traits = {}
    for need, divisions in expanded_dict.items():
        for division, traits in divisions.items():
            for trait, data in traits.items():
                weight = data['weight']
                for word in data['words']:
                    word_vector_in_dict = nlp(word).vector
                    similarity = cosine_similarity([sentence_vector], [word_vector_in_dict])[0][0]
                    weighted_similarity = similarity * weight
                    if weighted_similarity >= positive_threshold or weighted_similarity <= negative_threshold:
                        if need not in impacting_traits:
                            impacting_traits[need] = {}
                        if division not in impacting_traits[need]:
                            impacting_traits[need][division] = {}
                        if trait not in impacting_traits[need][division]:
                            impacting_traits[need][division][trait] = {
                                'similar_words': [],
                                'weight': weight
                            }
                        impacting_traits[need][division][trait]['similar_words'].append((word, similarity, weighted_similarity))
                         # Print calculations specifically for "Cautiousness"
                        #if trait == "Cautiousness":
                          #  print(f"Word: {word}")
                          #  print(f"  Similarity: {similarity}")
                          #  print(f"  Weight: {weight}")
                          #  print(f"  Weighted Similarity: {weighted_similarity}")
    return impacting_traits

# Consolidated module to adjust score using fuzzy logic and time-dependent factors
def consolidated_module(trait_name, current_score, cumulative_score, days_since_entry, base_decay=0.8):
    """
    Adjusts the weight of a trait by applying an age-dependent decay factor for previous cumulative scores,
    so that older data decays faster and scores with less impact decay more quickly.

    Args:
        trait_name (str): The name of the trait.
        current_score (float): The impact score for the current day.
        cumulative_score (float): The cumulative score for the trait up to the current day.
        days_since_entry (int): The number of days since the diary entry.
        base_decay (float): The base decay factor for recent data, modified by the age of the data.

    Returns:
        float: The adjusted cumulative score with accelerated decay for less impacted data.
    """
    # Adjust decay based on current score
    if current_score <= 200:  # Apply more decay for lower current scores
        age_decay_factor = (base_decay ** 2) ** (1 + (days_since_entry / 30))
    else:  # Standard decay for higher current scores
        age_decay_factor = base_decay ** (1 + (days_since_entry / 30))

    decayed_score = cumulative_score * age_decay_factor

    # Combine decayed score with current impact
    combined_score = decayed_score + current_score

    # Apply fuzzy logic adjustments based on the combined score
    if combined_score > 250:  # High combined score, moderate increase
        combined_score *= 1.3  # Increase by 10%
    elif 200 < combined_score <= 250:  # Moderate score, slight increase
        combined_score *= 1.05  # Increase by 5%
    elif combined_score <= 100:  # Low score, slight decrease
        combined_score *= 0.75  # Decrease by 5%

    # Ensure the combined score remains within reasonable bounds, e.g., between 5 and 20
    combined_score = max(5, min(1000, combined_score))
    return combined_score


# Function to calculate daily emotional scores, incorporating the consolidated module
def calculate_daily_scores(diary_entries, nlp_model, expanded_dict):
    daily_scores = {}
    cumulative_scores = defaultdict(float)
    positive_threshold = 5 # can be reset 
    negative_threshold = -1 # can be reset
    relativeness_threshold = 0.2 # can be reset
    for entry_date, entries in sorted(diary_entries.items()):
        day_score = defaultdict(float)
        for entry in entries:
            # Process the entry as text, assuming `entries` contains diary content (strings)
            sentence_vector = nlp_model(entry).vector
            filtered_dict = filter_dict_by_relativeness(expanded_dict, sentence_vector, relativeness_threshold)
            impacting_traits = find_all_similar_traits_with_sentence(sentence_vector, filtered_dict, positive_threshold, negative_threshold)

            # Calculate current impact for the day
            for need, divisions in impacting_traits.items():
                for division, traits in divisions.items():
                    for trait, details in traits.items():
                        for _, _, weighted_similarity in details['similar_words']:
                            day_score[trait] += weighted_similarity

        # Handle the date correctly (no indexing into `entry_date`)
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


def determine_tone(daily_scores):
    if 'Happiness' in daily_scores and daily_scores['Happiness'] > 15:
        return "enthusiastic and positive"
    elif 'Sadness' in daily_scores and daily_scores['Sadness'] > 15:
        return "empathetic and gentle"
    else:
        return "neutral and informative"

# Enhanced function to generate reflective prompts based on top trait
def generate_reflective_prompt(trait):
    prompts = {
        "Cautiousness": "It seems like you've been thinking a lot about safety and risk. Could you tell me more about what’s been on your mind?",
        "Self-Esteem": "You mentioned some thoughts on self-worth recently. What do you feel most proud of?",
        "Happiness": "It’s great to see moments of joy. What’s something that brought a smile to your face recently?"
    }
    return prompts.get(trait, "Can you share a bit more about your thoughts and feelings?")

# Enhanced function to suggest resources based on dynamically linked traits
def suggest_resources(trait, self_actualization_words):
    # Check if the trait exists in the self_actualization_words structure
    for need, divisions in self_actualization_words.items():
        for division, traits in divisions.items():
            if trait in traits:
                # Provide a generic suggestion for any found trait
                return f"For the trait '{trait}', consider exploring activities or resources that align with personal growth in this area."
    
    # If trait not found, return a default message
    return "Remember, reflecting on your experiences can help you grow. Keep it up!"


# Weekly summary calculation
def calculate_weekly_summary(daily_scores):
    weekly_summary = defaultdict(float)
    for scores in daily_scores.values():
        for trait, score in scores.items():
            weekly_summary[trait] += score
    for trait in weekly_summary:
        weekly_summary[trait] /= 7  # Average score over the week
    return dict(weekly_summary)

# Function to provide goal suggestions based on top traits
def provide_goal_suggestions(top_traits):
    suggestions = {
        "Self-Esteem": "You’ve been making great progress. Consider setting a goal that challenges you and reinforces your sense of accomplishment.",
        "Happiness": "To maintain positive emotions, try engaging in activities that bring you joy and make them a regular part of your routine.",
        "Resilience": "Keep building on your strengths. It could be beneficial to set a personal growth goal, like learning a new skill."
    }
    return [suggestions.get(trait, "Keep up the great work!") for trait, _ in top_traits]

# Merged function to get top traits and suggest resources for each
def get_top_traits_with_resources(accumulated_scores, self_actualization_words, num_traits=5):
    # Sort traits by accumulated scores in descending order and pick the top 'num_traits'
    sorted_traits = sorted(accumulated_scores.items(), key=lambda item: item[1], reverse=True)[:num_traits]
    
    # Add resource suggestions for each top trait
    top_traits_with_resources = []
    for trait, score in sorted_traits:
        suggestion = suggest_resources(trait, self_actualization_words)
        top_traits_with_resources.append((trait, score, suggestion))
        
    return top_traits_with_resources

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


# Function to get ChatGPT's response
def get_analysis_from_chatgpt(context_agent_1):
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


# Prepare context for ChatGPT
def prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits, tone):
    context_agent_2 = "Psychological Analysis of Diary Entries:\n\n"

    # Add tone to context
    context_agent_2 += f"\nTone for the analysis is '{tone}'. Please engage with the user based on this tone."  

    context_agent_2 += "Day-to-Day Score Changes:\n"
    for date, scores in daily_scores.items():
        context_agent_2 += f"{date}: {scores}\n"
    
    context_agent_2 += "\nAccumulated Trait Scores:\n"
    for trait, score in accumulated_scores.items():
        context_agent_2 += f"{trait}: {score}\n"
    
    # Adding top traits with resource suggestions
    top_traits_with_resources = get_top_traits_with_resources(accumulated_scores, self_actualization_words, num_traits=5)
    context_agent_2 += "\nTop Traits with Resource Suggestions:\n"
    for trait, score, suggestion in top_traits_with_resources:
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

# Function to get the latest diary entry and tone
def get_latest_entry_and_tone(user_id: str, daily_scores: dict):
    """Fetch the latest diary entry and determine the tone."""
    # Fetch the diary entries for the user
    diary_entries = fetch_diary_entries(user_id)
    if not diary_entries:
        raise ValueError("No diary entries found for the user.")
    # Find the latest diary entry date
    latest_entry_date = max(diary_entries.keys())
    # Check if daily_scores contains data for the latest entry
    if latest_entry_date not in daily_scores:
        raise ValueError(f"No scores found for the latest entry date: {latest_entry_date}")
    # Determine the tone based on the latest entry's daily score
    tone = determine_tone(daily_scores[latest_entry_date])
    return latest_entry_date, tone

# Function for follow-up interactions
def follow_up_questions(initial_analysis, diary_entries, daily_scores, score_changes, accumulated_scores, top_traits, tone):
    diary_summary = "Here are the original diary entries for reference:\n"
    for date, entries in diary_entries.items():
        diary_summary += f"{date}:\n"
        for entry in entries:
            diary_summary += f"  - {entry}\n"

    # Fetch the latest diary entry and tone
    latest_entry_date, tone = get_latest_entry_and_tone(user_id, daily_scores)
    
    # Prepare a comprehensive summary of diary entries and analysis details for context
    context_agent_2 = prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits,tone)

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
