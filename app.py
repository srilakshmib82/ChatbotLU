from flask import Flask, request, jsonify
from flask_cors import CORS
from intent_model import predict_intent, get_response, get_pos_tags
import os

app = Flask(__name__)
CORS(app)

# Home Route - index.html show
@app.route("/")
def home():
    return open("index.html", "r", encoding="utf-8").read()

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message not provided"}), 400

    user_input = data["message"]

    tag, score = predict_intent(user_input)
    response = get_response(tag)
    pos_tags = get_pos_tags(user_input)

    return jsonify({
        "intent": tag,
        "confidence": float(score),
        "response": response,
        "pos_tags": pos_tags
    })

# Favicon error avoid
@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
