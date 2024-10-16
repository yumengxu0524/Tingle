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
    # (Put your dictionary content here)
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


