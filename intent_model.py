import json
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

with open("intents.json", "r") as f:
    intents_data = json.load(f)

intent_patterns = []
intent_tags = []

for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        intent_patterns.append(pattern)
        intent_tags.append(intent["tag"])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(intent_patterns)

def get_pos_tags(sentence):
    doc = nlp(sentence)
    return [(token.text, token.pos_) for token in doc]

def predict_intent(user_input):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, pattern_vectors)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    predicted_tag = intent_tags[best_match_index]

    if best_score < 0.2:
        return "unknown", best_score

    return predicted_tag, best_score

def get_response(tag):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

