import spacy
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from dataclasses import dataclass

# Load NLP model
nlp = spacy.load('en_core_web_md')

# Define Trait dataclass
@dataclass
class Trait:
    words: list
    weight: float

# Define the original self_actualization_words dictionary
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

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def expand_dictionary_with_synonyms():
    expanded_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for need, divisions in original_self_actualization_words.items():
        for division, traits in divisions.items():
            for trait, data in traits.items():
                words = data['words']
                expanded_dict[need][division][trait] = {
                    'words': [], 'weight': data['weight']
                }
                for word in words:
                    synonyms = get_synonyms(word)
                    expanded_dict[need][division][trait]['words'].extend(synonyms)
                expanded_dict[need][division][trait]['words'] = list(set(expanded_dict[need][division][trait]['words']))
    return expanded_dict

def filter_dict_by_relativeness(expanded_dict, sentence_vector, relativeness_threshold):
    filtered_dict = {}
    for need, divisions in expanded_dict.items():
        filtered_divisions = {}
        for division, traits in divisions.items():
            filtered_traits = {}
            for trait, data in traits.items():
                trait_score = sum(
                    cosine_similarity([sentence_vector], [nlp(word).vector])[0][0]
                    for word in data['words']
                ) / len(data['words'])
                if trait_score >= relativeness_threshold:
                    filtered_traits[trait] = data
            if filtered_traits:
                filtered_divisions[division] = filtered_traits
        if filtered_divisions:
            filtered_dict[need] = filtered_divisions
    return filtered_dict

def find_all_similar_traits_with_sentence(sentence_vector, expanded_dict, positive_threshold, negative_threshold):
    impacting_traits = {}
    for need, divisions in expanded_dict.items():
        for division, traits in divisions.items():
            for trait, data in traits.items():
                weight = data['weight']
                for word in data['words']:
                    similarity = cosine_similarity([sentence_vector], [nlp(word).vector])[0][0]
                    weighted_similarity = similarity * weight
                    if weighted_similarity >= positive_threshold or weighted_similarity <= negative_threshold:
                        if need not in impacting_traits:
                            impacting_traits[need] = {}
                        if division not in impacting_traits[need]:
                            impacting_traits[need][division] = {}
                        if trait not in impacting_traits[need][division]:
                            impacting_traits[need][division][trait] = {'similar_words': [], 'weight': weight}
                        impacting_traits[need][division][trait]['similar_words'].append(
                            (word, similarity, weighted_similarity)
                        )
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

