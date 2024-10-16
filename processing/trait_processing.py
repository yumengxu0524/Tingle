from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def expand_dictionary_with_synonyms():
    original_self_actualization_words = {
        # Your original dictionary content here
    }
    expanded_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for need, divisions in original_self_actualization_words.items():
        for division, traits in divisions.items():
            for trait, data in traits.items():
                synonyms = {syn for word in data['words'] for syn in get_synonyms(word)}
                expanded_dict[need][division][trait] = {'words': list(synonyms), 'weight': data['weight']}
    return expanded_dict

def filter_dict_by_relativeness(expanded

