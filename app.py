from flask import Flask, request, jsonify
from flask_cors import CORS
from intent_model import predict_intent, get_response, get_pos_tags

app = Flask(__name__)
CORS(app)

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

if __name__ == "__main__":
    app.run(debug=True)
