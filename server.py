from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Load ML model + encoders
model = pickle.load(open("mood_model.pkl", "rb"))
flavor_encoder = pickle.load(open("flavor_encoder.pkl", "rb"))
mood_encoder = pickle.load(open("mood_encoder.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Mood ML API Running 🎉"

@app.route("/predict-mood", methods=["POST"])
def predict():
    data = request.json
    
    try:
        flavor = flavor_encoder.transform([data["flavor_type"]])[0]

        input_df = pd.DataFrame([{
            "flavor_type": flavor,
            "sweetness_level": data["sweetness_level"],
            "richness": data["richiness"],   # keep same spelling as frontend
            "fruity": data["fruity"],
            "nutty": data["nutty"],
            "price": data["price"]
        }])

        pred = model.predict(input_df)[0]
        mood = mood_encoder.inverse_transform([pred])[0]

        return jsonify({"mood": mood})

    except Exception as e:
        return jsonify({"error": str(e)})

app.run(host="0.0.0.0", port=5000)
