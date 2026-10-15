
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder="template")

BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / "faqs.json"

# Tune this threshold through testing.
SIMILARITY_THRESHOLD = 0.25


def load_faqs():
    with FAQ_FILE.open("r", encoding="utf-8") as file:
        faqs = json.load(file)

    if not isinstance(faqs, list) or not faqs:
        raise ValueError("faqs.json must contain a non-empty list.")

    for faq in faqs:
        if not isinstance(faq, dict):
            raise ValueError("Each FAQ must be a JSON object.")

        if not isinstance(faq.get("question"), str) or not faq["question"].strip():
            raise ValueError("Every FAQ needs a non-empty question.")

        if not isinstance(faq.get("answer"), str) or not faq["answer"].strip():
            raise ValueError("Every FAQ needs a non-empty answer.")

    return faqs


faqs = load_faqs()
faq_questions = [faq["question"] for faq in faqs]

# Convert FAQ questions into numerical TF-IDF vectors.
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    strip_accents="unicode"
)

faq_vectors = vectorizer.fit_transform(faq_questions)


def find_answer(user_question):
    # Transform the user's question using the same fitted vectorizer.
    user_vector = vectorizer.transform([user_question])

    # Compare the user vector with every FAQ question vector.
    scores = cosine_similarity(user_vector, faq_vectors).flatten()

    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answer": (
                "Sorry, I couldn't find a reliable answer. "
                "Please contact the relevant university department."
            ),
            "matched_question": None,
            "similarity": round(best_score, 3),
            "matched": False
        }

    return {
        "answer": faqs[best_index]["answer"],
        "matched_question": faqs[best_index]["question"],
        "similarity": round(best_score, 3),
        "matched": True
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please send a valid JSON object."}), 400

    question = data.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please enter a question."}), 400

    if len(question) > 500:
        return jsonify({"error": "Question must be 500 characters or fewer."}), 400

    result = find_answer(question.strip())
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
