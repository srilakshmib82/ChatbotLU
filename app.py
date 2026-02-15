from flask import Flask, request, jsonify
from flask_cors import CORS
from intent_model import predict_intent, get_response, get_pos_tags
import os

app = Flask(__name__)
CORS(app)

# Home Route - to show index.html
@app.route("/")
def home():
    return open("index.html", "r", encoding="utf-8").read()

# Chat Route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    tag, score = predict_intent(user_input)
    response = get_response(tag)

    pos_tags = get_pos_tags(user_input)

    return jsonify({
        "intent": tag,
        "confidence": float(score),
        "response": response,
        "pos_tags": pos_tags
    })

# Optional: favicon fix
@app.route("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
