import spacy

def load_nlp_model():
    return spacy.load('en_core_web_md')

def load_diary_entries():
    return {
        datetime.date(2024, 6, 20): ["I hope I will be able to confide everything to you..."],
        datetime.date(2024, 6, 21): ["Writing in a diary is a really strange experience..."],
        # Add more entries as needed
    }

