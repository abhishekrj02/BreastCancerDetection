import os
import io
import numpy as np
import cv2 as cv
from PIL import Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import tensorflow as tf
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# ─── Gemini configuration ───

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Model globals ───
model = None

CLASS_NAMES = [
    "Benign with Density 1",
    "Malignant with Density 1",
    "Benign with Density 2",
    "Malignant with Density 2",
    "Benign with Density 3",
    "Malignant with Density 3",
    "Benign with Density 4",
    "Malignant with Density 4",
]

CHAT_SYSTEM_PROMPT = (
    "You are a compassionate and knowledgeable breast health advisor AI assistant. "
    "You can answer questions about breast cancer, mammograms, breast health, explain "
    "medical terms in simple language, provide general health and lifestyle advice for "
    "breast cancer prevention, discuss screening guidelines, and offer emotional support.\n\n"
    "Important guidelines:\n"
    "- Always remind users that you are an AI and cannot replace professional medical advice\n"
    "- Be empathetic and supportive\n"
    "- If someone describes symptoms, advise them to see a healthcare provider\n"
    "- Provide evidence-based information\n"
    "- Keep responses concise but thorough"
)


def load_model():
    global model
    from model import download_model
    from weights import download_weights

    os.makedirs("model", exist_ok=True)
    os.makedirs("weight", exist_ok=True)

    if not os.path.exists("model/model.h5"):
        print("Building model architecture...")
        download_model()

    model = tf.keras.models.load_model("model/model.h5")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    if not os.path.exists("weight/modeldense1.h5"):
        print("Downloading weights...")
        download_weights()

    model.load_weights("weight/modeldense1.h5")
    print("Model loaded successfully.")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = cv.resize(img, (224, 224))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv.filter2D(img, -1, kernel)
    img = img / 255.0
    return img.reshape(1, 224, 224, 3)


# ─── Routes ───


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        img = preprocess_image(image_bytes)
        pred = model.predict(img)[0]

        detailed = {}
        for i, name in enumerate(CLASS_NAMES):
            detailed[name] = round(float(pred[i]) * 100, 2)

        benign = float(pred[0] + pred[2] + pred[4] + pred[6]) * 100
        malignant = float(pred[1] + pred[3] + pred[5] + pred[7]) * 100

        return jsonify(
            {
                "benign": round(benign, 2),
                "malignant": round(malignant, 2),
                "detailed": detailed,
                "diagnosis": "Malignant" if malignant > benign else "Benign",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-summary", methods=["POST"])
def get_summary():
    if not gemini_client:
        return jsonify({"error": "Gemini API key not configured. Set GEMINI_API_KEY in .env file."}), 500

    data = request.json
    benign = data.get("benign", 0)
    malignant = data.get("malignant", 0)
    diagnosis = data.get("diagnosis", "Unknown")
    detailed = data.get("detailed", {})

    detail_lines = "\n".join([f"- {k}: {v}%" for k, v in detailed.items()])

    prompt = f"""You are an experienced oncologist and breast health specialist. A mammogram analysis AI has produced the following results:

**Overall Prediction:**
- Benign probability: {benign}%
- Malignant probability: {malignant}%
- Primary Diagnosis: {diagnosis}

**Detailed Breakdown by Breast Density:**
{detail_lines}

Based on these results, please provide:

1. **Summary** - A clear, compassionate explanation of what these results mean in simple terms.
2. **Risk Assessment** - Based on the probability percentages, explain the level of concern.
3. **Recommended Next Steps** - What should the patient do next (further tests, biopsy, follow-up, etc.).
4. **Healthy Practices** - 5-7 breast health practices and lifestyle recommendations for prevention and well-being.
5. **Important Disclaimer** - Remind that this is an AI screening tool and not a definitive medical diagnosis.

Keep the tone professional yet empathetic, like a caring doctor would speak to their patient."""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return jsonify({"summary": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    if not gemini_client:
        return jsonify({"error": "Gemini API key not configured. Set GEMINI_API_KEY in .env file."}), 500

    data = request.json
    message = data.get("message", "")
    history = data.get("history", [])
    detection_results = data.get("detectionResults", None)

    # Build system instruction with detection context if available
    system_instruction = CHAT_SYSTEM_PROMPT
    if detection_results:
        benign = detection_results.get("benign", 0)
        malignant = detection_results.get("malignant", 0)
        diagnosis = detection_results.get("diagnosis", "Unknown")
        detailed = detection_results.get("detailed", {})
        detail_lines = "\n".join([f"  - {k}: {v}%" for k, v in detailed.items()])

        system_instruction += (
            f"\n\n--- CURRENT DETECTION RESULTS ---\n"
            f"The user has just analyzed a mammogram image. Here are the AI detection results:\n"
            f"- Overall Diagnosis: {diagnosis}\n"
            f"- Benign Probability: {benign}%\n"
            f"- Malignant Probability: {malignant}%\n"
            f"- Detailed Breakdown by Breast Density:\n{detail_lines}\n\n"
            f"Use these results to provide context-aware answers. If the user asks about their results, "
            f"refer to this data. Always remind them this is an AI screening tool and not a definitive diagnosis."
        )

    # Build contents list with system instruction and history
    contents = [{"role": "user", "parts": [{"text": CHAT_SYSTEM_PROMPT + "\n\nUser: " + message}]}]
    if history:
        contents = []
        for h in history:
            contents.append({"role": h["role"], "parts": [{"text": h["content"]}]})
        contents.append({"role": "user", "parts": [{"text": message}]})

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config={"system_instruction": system_instruction},
        )
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    app.run(debug=True, port=5000)
