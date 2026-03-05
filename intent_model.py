import json
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Load intents dataset
with open("intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

intent_patterns = []
intent_tags = []

# Extract patterns and tags from JSON
for intent in intents_data["intents"]:
    for pattern in intent["text"]:   # Kaggle dataset uses "text"
        intent_patterns.append(pattern.lower().strip())
        intent_tags.append(intent["intent"])  # Kaggle dataset uses "intent"

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(intent_patterns)

# POS tagging
def get_pos_tags(sentence):
    doc = nlp(sentence)
    return [(token.text, token.pos_) for token in doc]

# Intent prediction
def predict_intent(user_input):
    user_input = user_input.lower().strip()

    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, pattern_vectors)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    predicted_tag = intent_tags[best_match_index]

    # Threshold
    if best_score < 0.30:
        return "unknown", best_score

    return predicted_tag, best_score

# Get chatbot response
def get_response(tag):
    for intent in intents_data["intents"]:
        if intent["intent"] == tag:   # changed from tag → intent
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."
