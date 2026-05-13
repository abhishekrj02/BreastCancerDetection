import os
import io
import uuid
import json
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import cv2 as cv
from PIL import Image

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, send_file, abort
)
from flask_login import login_required, current_user
from dotenv import load_dotenv
import tensorflow as tf
from google import genai
from sqlalchemy import func

from extensions import db, login_manager, bcrypt, mail
from database import User, Prediction
from auth import auth
from gradcam import compute_gradcam, overlay_heatmap
from pdf_report import generate_report
from email_utils import send_prediction_email, send_high_risk_alert

load_dotenv()

# ─── App & Config ───
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "breastguard-dev-secret-2024")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///breastguard.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

# ─── Init Extensions ───
db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to access BreastGuard AI."
login_manager.login_message_category = "info"
mail.init_app(app)

app.register_blueprint(auth)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─── Gemini ───
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── ML Model ───
model = None

CLASS_NAMES = [
    "Benign with Density 1", "Malignant with Density 1",
    "Benign with Density 2", "Malignant with Density 2",
    "Benign with Density 3", "Malignant with Density 3",
    "Benign with Density 4", "Malignant with Density 4",
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


# ─── Role Decorators ───
def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ("doctor", "admin"):
            abort(403)
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated


# ─── Main Routes ───

@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/compare")
@login_required
def compare():
    return render_template("compare.html")


@app.route("/dashboard")
@login_required
def dashboard():
    preds = (
        Prediction.query
        .filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    total = len(preds)
    benign_count = sum(1 for p in preds if p.diagnosis == "Benign")
    malignant_count = total - benign_count
    return render_template(
        "dashboard.html",
        predictions=preds,
        total=total,
        benign_count=benign_count,
        malignant_count=malignant_count,
    )


@app.route("/doctor/dashboard")
@login_required
@doctor_required
def doctor_dashboard():
    search = request.args.get("search", "").strip()
    diag_filter = request.args.get("diagnosis", "")

    query = (
        Prediction.query
        .join(User, Prediction.user_id == User.id)
        .order_by(Prediction.created_at.desc())
    )
    if search:
        query = query.filter(User.username.ilike(f"%{search}%"))
    if diag_filter in ("Benign", "Malignant"):
        query = query.filter(Prediction.diagnosis == diag_filter)

    preds = query.all()
    return render_template("doctor_dashboard.html", predictions=preds, search=search, diag_filter=diag_filter)


@app.route("/admin/dashboard")
@login_required
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    total_preds = Prediction.query.count()
    malignant_count = Prediction.query.filter_by(diagnosis="Malignant").count()
    benign_count = Prediction.query.filter_by(diagnosis="Benign").count()
    doctors = User.query.filter_by(role="doctor").count()
    patients = User.query.filter_by(role="patient").count()

    # Predictions per day — last 7 days
    daily_labels = []
    daily_counts = []
    for i in range(6, -1, -1):
        day = datetime.utcnow().date() - timedelta(days=i)
        count = Prediction.query.filter(
            func.date(Prediction.created_at) == day
        ).count()
        daily_labels.append(day.strftime("%b %d"))
        daily_counts.append(count)

    users = User.query.order_by(User.created_at.desc()).all()
    recent_preds = (
        Prediction.query
        .join(User, Prediction.user_id == User.id)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_preds=total_preds,
        malignant_count=malignant_count,
        benign_count=benign_count,
        doctors=doctors,
        patients=patients,
        daily_labels=json.dumps(daily_labels),
        daily_counts=json.dumps(daily_counts),
        users=users,
        recent_preds=recent_preds,
    )


# ─── API Routes ───

@app.route("/predict", methods=["POST"])
@login_required
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

        detailed = {name: round(float(pred[i]) * 100, 2) for i, name in enumerate(CLASS_NAMES)}
        benign = float(pred[0] + pred[2] + pred[4] + pred[6]) * 100
        malignant = float(pred[1] + pred[3] + pred[5] + pred[7]) * 100
        diagnosis = "Malignant" if malignant > benign else "Benign"
        top_class = int(np.argmax(pred))

        # ── Grad-CAM ──
        heatmap_b64 = None
        heatmap_filename = None
        try:
            heatmap, _ = compute_gradcam(model, img, class_idx=top_class)
            if heatmap is not None:
                heatmap_b64 = overlay_heatmap(heatmap, image_bytes)
                # Save heatmap PNG to disk
                heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
                import base64
                heatmap_data = heatmap_b64.split(",", 1)[1]
                heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
                with open(heatmap_path, "wb") as hf:
                    hf.write(base64.b64decode(heatmap_data))
        except Exception as e:
            print(f"[Predict] Grad-CAM failed: {e}")

        # ── Save original image ──
        img_filename = f"scan_{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, img_filename)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img.save(img_path, "JPEG")

        # ── Save to DB ──
        record = Prediction(
            user_id=current_user.id,
            image_filename=img_filename,
            heatmap_filename=heatmap_filename,
            benign=round(benign, 2),
            malignant=round(malignant, 2),
            diagnosis=diagnosis,
            detailed=json.dumps(detailed),
        )
        db.session.add(record)
        db.session.commit()

        # ── Email notifications (background) ──
        send_prediction_email(app, current_user, record)
        if malignant > 50:
            send_high_risk_alert(app, current_user, record)

        return jsonify({
            "benign": round(benign, 2),
            "malignant": round(malignant, 2),
            "detailed": detailed,
            "diagnosis": diagnosis,
            "heatmap_b64": heatmap_b64,
            "prediction_id": record.id,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-summary", methods=["POST"])
@login_required
def get_summary():
    if not gemini_client:
        return jsonify({"error": "Gemini API key not configured."}), 500

    data = request.json
    benign = data.get("benign", 0)
    malignant = data.get("malignant", 0)
    diagnosis = data.get("diagnosis", "Unknown")
    detailed = data.get("detailed", {})
    prediction_id = data.get("prediction_id")

    detail_lines = "\n".join([f"- {k}: {v}%" for k, v in detailed.items()])
    prompt = f"""You are an experienced oncologist and breast health specialist. A mammogram analysis AI has produced the following results:

**Overall Prediction:**
- Benign probability: {benign}%
- Malignant probability: {malignant}%
- Primary Diagnosis: {diagnosis}

**Detailed Breakdown by Breast Density:**
{detail_lines}

Based on these results, please provide:
1. **Summary** - A clear, compassionate explanation of what these results mean.
2. **Risk Assessment** - The level of concern based on the probabilities.
3. **Recommended Next Steps** - What the patient should do next.
4. **Healthy Practices** - 5-7 breast health and lifestyle recommendations.
5. **Important Disclaimer** - Remind that this is an AI screening tool.

Keep the tone professional yet empathetic."""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        summary_text = response.text

        # Save summary to DB if prediction_id given
        if prediction_id:
            record = Prediction.query.get(prediction_id)
            if record and record.user_id == current_user.id:
                record.ai_summary = summary_text
                db.session.commit()

        return jsonify({"summary": summary_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    if not gemini_client:
        return jsonify({"error": "Gemini API key not configured."}), 500

    data = request.json
    message = data.get("message", "")
    history = data.get("history", [])
    detection_results = data.get("detectionResults", None)

    system_instruction = CHAT_SYSTEM_PROMPT
    if detection_results:
        benign = detection_results.get("benign", 0)
        malignant = detection_results.get("malignant", 0)
        diagnosis = detection_results.get("diagnosis", "Unknown")
        detailed = detection_results.get("detailed", {})
        detail_lines = "\n".join([f"  - {k}: {v}%" for k, v in detailed.items()])
        system_instruction += (
            f"\n\n--- CURRENT DETECTION RESULTS ---\n"
            f"Diagnosis: {diagnosis}\nBenign: {benign}%\nMalignant: {malignant}%\n"
            f"Breakdown:\n{detail_lines}\n\n"
            "Refer to these when the user asks about their results. "
            "Always remind them this is an AI tool, not a definitive diagnosis."
        )

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


@app.route("/download-report/<int:prediction_id>")
@login_required
def download_report(prediction_id):
    record = Prediction.query.get_or_404(prediction_id)
    # Patients see only their own; doctors/admins see all
    if current_user.role == "patient" and record.user_id != current_user.id:
        abort(403)

    patient = User.query.get(record.user_id)
    img_path = os.path.join(UPLOAD_FOLDER, record.image_filename) if record.image_filename else None
    heatmap_path = os.path.join(UPLOAD_FOLDER, record.heatmap_filename) if record.heatmap_filename else None

    buf = generate_report(record, patient, img_path, heatmap_path)
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"BreastGuard_Report_{patient.username}_{prediction_id}.pdf",
        mimetype="application/pdf",
    )


@app.route("/api/add-note/<int:prediction_id>", methods=["POST"])
@login_required
@doctor_required
def add_note(prediction_id):
    record = Prediction.query.get_or_404(prediction_id)
    data = request.json
    record.doctor_notes = data.get("notes", "")
    db.session.commit()
    return jsonify({"status": "saved"})


# ─── Error pages ───
@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("Loading model...")
    load_model()
    app.run(debug=True, port=5000)
