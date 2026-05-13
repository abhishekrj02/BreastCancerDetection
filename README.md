# BreastGuard AI — Breast Cancer Detection

An AI-powered full-stack web application for breast cancer detection from mammogram images using **DenseNet201** deep learning, **Grad-CAM** explainability, role-based authentication, clinical PDF reports, and **Gemini AI** health assistance.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Contributing](#contributing)
- [Training Your Own Model](#training-your-own-model)
- [Disclaimer](#disclaimer)

---

## Features

### AI / ML
- **Mammogram Classification** — 8-class prediction (Benign/Malignant × 4 breast density levels) using DenseNet201 transfer learning
- **Grad-CAM Heatmap** — Heatmap overlay showing which regions the model focused on for its prediction
- **AI Health Summary** — Gemini 2.5 Flash generates a clinical-style risk assessment and next steps
- **Context-Aware Chatbot** — Breast health AI assistant that automatically knows your scan results

### Authentication & Roles
- **Login / Register / Logout** — Secure session-based auth with bcrypt password hashing
- **Three roles:** Patient, Doctor, Admin — each with different access levels
- **Role-based route protection** — Unauthorized access returns 403

### Reports & History
- **PDF Clinical Report** — Downloadable A4 report with mammogram image, Grad-CAM heatmap, results table, and AI summary
- **Patient Dashboard** — Scan history with thumbnails, stats, and report downloads
- **Doctor Review Panel** — View all patient scans, add clinical notes, filter by patient or diagnosis
- **Admin Analytics Dashboard** — Platform-wide stats, Chart.js charts, full user and scan tables
- **Comparison View** — Upload two mammograms and analyze them side by side

### Notifications
- **Result Email** — Automatic email sent after every scan with diagnosis and probabilities
- **High-Risk Alert** — Urgent email when malignant probability exceeds 50%

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask, Flask-Login, Flask-SQLAlchemy, Flask-Bcrypt, Flask-Mail |
| Deep Learning | TensorFlow 2.21 / Keras — DenseNet201 transfer learning |
| Explainability | tf-keras-vis (Grad-CAM++) |
| AI Assistant | Google Gemini 2.5 Flash (`google-genai` SDK) |
| PDF Generation | ReportLab |
| Database | SQLite (via SQLAlchemy) |
| Image Processing | OpenCV, Pillow |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js (CDN) |
| Markdown Rendering | marked.js (CDN) |
| Icons / Fonts | Font Awesome 6.5, Google Fonts (Poppins) |

---

## Project Structure

```
BreastCancerDetection/
│
├── app.py                    # Main Flask app — all routes and business logic
├── extensions.py             # Flask extension instances (db, login_manager, bcrypt, mail)
├── database.py               # SQLAlchemy models: User, Prediction
├── auth.py                   # Auth Blueprint: /login, /register, /logout
├── gradcam.py                # Grad-CAM++ heatmap generation for DenseNet201
├── pdf_report.py             # Clinical PDF report generation with ReportLab
├── email_utils.py            # Async email notifications via Flask-Mail
│
├── model.py                  # Builds and saves DenseNet201 model architecture
├── weights.py                # Downloads pre-trained weights from Google Drive
├── train.py                  # Full training script with augmentation and callbacks
│
├── requirements.txt          # All Python dependencies
├── .env                      # Your API keys and secrets (never committed)
├── .gitignore
│
├── templates/
│   ├── index.html            # Main detection page
│   ├── login.html            # Login page
│   ├── register.html         # Registration page with role selector
│   ├── dashboard.html        # Patient scan history and stats
│   ├── doctor_dashboard.html # Doctor review panel with search/filter
│   ├── admin_dashboard.html  # Admin analytics with Chart.js
│   ├── compare.html          # Side-by-side scan comparison
│   ├── 403.html              # Access denied
│   └── 404.html              # Not found
│
├── static/
│   ├── css/style.css         # Full design system + all page styles
│   ├── js/main.js            # Detection page logic
│   ├── js/compare.js         # Comparison page logic
│   └── uploads/              # Saved scan images and heatmaps (auto-created, not in git)
│
├── model/                    # (auto-created) saved model architecture .h5
├── weight/                   # (auto-created) saved model weights .h5
├── instance/                 # (auto-created) SQLite database
└── Test_images/              # Sample mammograms for testing
```

---

## How It Works

### Prediction Flow

```
User uploads mammogram
       ↓
Preprocess: resize 224×224, Laplacian sharpening, normalize [0,1]
       ↓
DenseNet201 inference → 8 softmax probabilities
       ↓
Aggregate: Benign % (sum of 4 benign classes), Malignant % (sum of 4 malignant classes)
       ↓
Grad-CAM++: compute heatmap → overlay on original image (base64 PNG)
       ↓
Save image + heatmap to static/uploads/
Save prediction record to SQLite database
       ↓
Send email notification in background thread
       ↓
Return JSON: { benign, malignant, diagnosis, detailed, heatmap_b64, prediction_id }
```

### Model Architecture

```
Input (224 × 224 × 3)
      ↓
DenseNet201 (ImageNet pre-trained, last 5 layers unfrozen)
      ↓  Global Max Pooling
BatchNormalization
      ↓
Dense(2048, ReLU) + L1-L2 Regularization
      ↓
BatchNormalization
      ↓
Dense(8, Softmax)  →  8 classes: Benign/Malignant × Density 1–4
```

---

## Model Performance

Trained on **6,869 mammogram images** split 75/8/17 (train/val/test).

| Metric | Score |
|--------|-------|
| Test Accuracy | **88.5%** |
| Macro F1-Score | 82.3% |
| Weighted F1-Score | 87.9% |
| AUC-ROC (binary) | **0.934** |

### Per-Class Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Benign — Density 1 | 0.85 | 0.88 | 0.86 |
| Malignant — Density 1 | 0.94 | 0.93 | 0.93 |
| Benign — Density 2 | 0.79 | 0.75 | 0.77 |
| Malignant — Density 2 | 0.92 | 0.94 | 0.93 |
| Benign — Density 3 | 0.86 | 0.87 | 0.86 |
| Malignant — Density 3 | 0.81 | 0.83 | 0.82 |
| Benign — Density 4 | 0.80 | 0.78 | 0.79 |
| Malignant — Density 4 | 0.68 | 0.64 | 0.66 |

> Density 4 Malignant has the lowest score due to only 54 training samples (rarest class). Class weighting (13.25×) is applied during training to partially compensate.

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- Git
- A [Google Gemini API key](https://aistudio.google.com/apikey) (free)
- Gmail account with [App Password](https://myaccount.google.com/apppasswords) *(optional, for emails)*

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/BreastCancerDetection.git
cd BreastCancerDetection
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> First install takes a few minutes — TensorFlow alone is ~500MB.

### Step 4 — Create the `.env` File

Create a file named `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional — email notifications
MAIL_USERNAME=youremail@gmail.com
MAIL_PASSWORD=your_16_char_gmail_app_password

# Optional — change in production
SECRET_KEY=change-this-to-a-long-random-string
```

### Step 5 — Run the App

```bash
python app.py
```

On **first run**, the app automatically:
1. Creates the SQLite database (`instance/breastguard.db`)
2. Builds the DenseNet201 model → `model/model.h5`
3. Downloads pre-trained weights → `weight/modeldense1.h5`

Once you see `Model loaded successfully.`, open **http://localhost:5000**.

### Step 6 — Register and Use

Go to `http://localhost:5000`, you'll be redirected to login. Click **Create one** to register as a Patient or Doctor.

### Making Yourself Admin

```bash
sqlite3 instance/breastguard.db
UPDATE user SET role = 'admin' WHERE email = 'you@example.com';
.quit
```

Or use a GUI tool like **DB Browser for SQLite**.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `MAIL_USERNAME` | No | Gmail address for sending emails |
| `MAIL_PASSWORD` | No | Gmail App Password (not your account password) |
| `SECRET_KEY` | Recommended | Flask session secret — always change in production |

---

## Usage Guide

### Patient
1. Register with Account Type = **Patient**
2. Upload a mammogram on the main page and click **Analyze Image**
3. View diagnosis, probability rings, Grad-CAM heatmap, and per-density breakdown
4. Click **Get AI Health Summary** for a Gemini-generated clinical summary
5. Click **Download PDF Report** to save a clinical report
6. Open the **Chat Assistant** (bottom-right) — it knows your scan results
7. Visit `/dashboard` to see all past scans and download previous reports
8. Visit `/compare` to analyze two scans side by side

### Doctor
1. Register with Account Type = **Doctor**
2. Visit `/doctor/dashboard` — see all patient scans in a searchable table
3. Add clinical notes inline — click the notes field, type, click Save
4. Download PDF reports for any patient
5. Filter by patient name or diagnosis type

### Admin
1. Set your role to `admin` in the database (see above)
2. Visit `/admin/dashboard` for platform-wide stats, charts, and user/scan tables

---

## API Reference

All routes require login. JSON responses.

| Method | Endpoint | Role | Description |
|--------|----------|------|-------------|
| `POST` | `/predict` | Any | Upload image → prediction + heatmap + save to DB |
| `POST` | `/get-summary` | Any | Gemini AI health summary |
| `POST` | `/chat` | Any | Gemini chatbot message |
| `GET` | `/download-report/<id>` | Owner / Doctor | Download PDF report |
| `POST` | `/api/add-note/<id>` | Doctor / Admin | Save clinical note |

**POST `/predict`** — `multipart/form-data` with `image` field

```json
{
  "benign": 72.4,
  "malignant": 27.6,
  "diagnosis": "Benign",
  "detailed": { "Benign with Density 1": 45.2, "...": "..." },
  "heatmap_b64": "data:image/png;base64,...",
  "prediction_id": 42
}
```

**POST `/get-summary`** — send prediction JSON + optional `prediction_id`

```json
{ "summary": "# AI Health Summary\n\n..." }
```

**POST `/api/add-note/<id>`** — `{ "notes": "Clinical note text" }` → `{ "status": "saved" }`

---

## Database Schema

### `user`

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| username | String(80) | Unique |
| email | String(120) | Unique, used for login |
| password_hash | String(255) | bcrypt |
| role | String(20) | `patient` / `doctor` / `admin` |
| created_at | DateTime | |

### `prediction`

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| user_id | Integer FK | → user.id |
| image_filename | String | Stored in `static/uploads/` |
| heatmap_filename | String | Stored in `static/uploads/` |
| benign | Float | Aggregated benign % |
| malignant | Float | Aggregated malignant % |
| diagnosis | String | `Benign` or `Malignant` |
| detailed | Text | JSON string of 8-class probabilities |
| ai_summary | Text | Gemini summary (saved after generation) |
| doctor_notes | Text | Doctor's clinical notes |
| created_at | DateTime | |

---

## Contributing

### Getting Started

1. **Fork** the repo on GitHub
2. **Clone your fork** and set up the project following [Setup & Installation](#setup--installation)
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Branch Naming

| Prefix | Use for |
|--------|---------|
| `feature/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code cleanup, no behavior change |

### Making Changes

- **New routes** → `app.py`
- **New DB models** → `database.py`
- **New templates** → `templates/` — copy the navbar pattern from any existing template
- **New CSS** → add at the bottom of `static/css/style.css` using existing CSS variables (`--pink`, `--blue`, `--card`, etc.)
- **New JS** → create `static/js/<page>.js` and link in the template

**Protecting a new route:**
```python
@app.route('/new-page')
@login_required          # requires any logged-in user
def new_page():
    ...

@app.route('/doctor-only')
@login_required
@doctor_required         # requires doctor or admin role
def doctor_only():
    ...
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add two-factor authentication
fix: handle null heatmap when model not loaded
docs: update API reference
refactor: extract email HTML to separate templates
```

### Submitting a Pull Request

1. Test locally — run `python app.py` and verify everything works
2. Commit your changes (add specific files, not `git add -A`)
3. Push to your fork and open a PR against `main`
4. Describe what you changed and why; include screenshots for UI changes

### What Not to Commit

Never commit: `.env`, `instance/`, `model/`, `weight/`, `static/uploads/`, `venv/`

---

## Training Your Own Model

1. Download the dataset: [Google Drive](https://drive.google.com/file/d/12umDKmXJ8--ZmuiTrchSQRCs8SmRl12h/view)
2. Organize into 8 subfolders per class under `train/` and `test/`
3. Update `TRAIN_DIR` and `TEST_DIR` in `train.py`
4. Run:
   ```bash
   python train.py
   ```

The script includes data augmentation, inverse-frequency class weighting, early stopping, `ReduceLROnPlateau`, `ModelCheckpoint`, and TensorBoard logging. Best weights are saved to `weight/modeldense1.h5`.

---

## Disclaimer

BreastGuard AI is built for **educational purposes** as a B.Tech final year project. It is **not a certified medical device** and must not be used as a substitute for professional medical diagnosis. Always consult a qualified radiologist or oncologist for clinical decisions.
