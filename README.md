# BreastGuard AI — Breast Cancer Detection from Mammograms

A web application that uses a **DenseNet201** deep learning model to classify mammogram images as benign or malignant across 4 breast density levels. It also provides AI-generated health summaries and a context-aware chatbot powered by **Google Gemini**.

## Features

- **Mammogram Classification** — Upload a mammogram image and get instant predictions with benign/malignant probabilities broken down by breast density (4 levels).
- **AI Health Summary** — One-click summary from Gemini AI with risk assessment, recommended next steps, and healthy practices.
- **Context-Aware Chatbot** — Ask questions about breast health. If a detection has been run, the chatbot automatically knows your results and can discuss them.
- **Visual Results** — Animated progress rings, diagnosis badge, and per-density-level breakdown bars.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (Python) |
| Deep Learning | TensorFlow / Keras — DenseNet201 transfer learning |
| AI Features | Google Gemini 2.5 Flash (summary + chatbot) |
| Image Processing | OpenCV, Pillow |
| Frontend | HTML, CSS, vanilla JavaScript |
| Markdown Rendering | marked.js |

## Project Structure

```
MajorProjectFinal/
├── app.py              # Flask server — all API routes (/predict, /get-summary, /chat)
├── model.py            # Builds and saves the DenseNet201 model architecture
├── weights.py          # Downloads pre-trained weights from Google Drive
├── train.py            # Training script (data augmentation, class weighting, callbacks)
├── requirements.txt    # Python dependencies
├── .env                # Gemini API key (you create this)
├── templates/
│   └── index.html      # Main HTML page
├── static/
│   ├── css/
│   │   └── style.css   # All styling
│   └── js/
│       └── main.js     # Frontend logic (upload, prediction display, chat, summary)
├── model/              # (auto-created) saved model architecture (.h5)
└── weight/             # (auto-created) saved model weights (.h5)
```

## How It Works

1. **Image Upload** — User uploads a mammogram (JPG/PNG/BMP).
2. **Preprocessing** — Image is resized to 224x224, sharpened with a kernel filter, and normalized to [0, 1].
3. **Model Inference** — DenseNet201 predicts probabilities for 8 classes (Benign/Malignant x Density 1-4).
4. **Results Display** — Probabilities are aggregated into overall benign vs malignant percentages and shown with animated visuals.
5. **AI Summary** — User can request a Gemini-generated health summary based on the detection results.
6. **Chatbot** — User can chat with an AI assistant that has full context of the detection results.

### Model Details

- **Architecture**: DenseNet201 (ImageNet pre-trained) with custom head — BatchNorm, Dense(2048, ReLU), BatchNorm, Dense(8, Softmax)
- **Fine-tuning**: Last 5 layers of DenseNet201 are unfrozen
- **8 Output Classes**:
  - Benign with Density 1, 2, 3, 4
  - Malignant with Density 1, 2, 3, 4

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- A Google Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/apikey))

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd MajorProjectFinal
```

### Step 2 — Create a Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set Up the Gemini API Key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 5 — Run the Application

```bash
python app.py
```

On first run, the app will automatically:
1. Build the DenseNet201 model architecture and save it to `model/model.h5`
2. Download pre-trained weights from Google Drive to `weight/modeldense1.h5`

Once you see `Model loaded successfully.`, open your browser and go to:

```
http://localhost:5000
```

## Usage

1. Open `http://localhost:5000` in your browser.
2. Upload a mammogram image (drag & drop or click to browse).
3. Click **Analyze Image** to run the detection.
4. View the results — diagnosis badge, benign/malignant percentages, and per-density breakdown.
5. Click **Get AI Health Summary from Gemini** for a detailed AI-generated report.
6. Open the **Chat Assistant** (bottom-right bubble or nav link) to ask questions — the chatbot will know your detection results automatically.

## Training Your Own Model (Optional)

If you want to retrain the model on your own dataset:

1. Download the dataset: [Google Drive Link](https://drive.google.com/file/d/12umDKmXJ8--ZmuiTrchSQRCs8SmRl12h/view)
2. Organize it into train/test folders with 8 subfolders (one per class).
3. Update `TRAIN_DIR` and `TEST_DIR` paths in `train.py`.
4. Run:

```bash
python train.py
```

The training script includes data augmentation, class weighting for the imbalanced dataset, early stopping, learning rate reduction, and TensorBoard logging. Best weights are saved to `weight/modeldense1.h5`.

## Dataset

The dataset contains breast mammography images (224x224x3) labeled by:

- **Breast Density** (4 levels):
  - Density 1 — Almost entirely fatty
  - Density 2 — Scattered areas of fibroglandular density
  - Density 3 — Heterogeneously dense
  - Density 4 — Extremely dense
- **Tumor Type**: Benign (non-cancerous) or Malignant (cancerous)

Higher breast density is associated with higher breast cancer risk and makes mammograms harder to read.

## Disclaimer

This is an AI screening tool built for educational purposes. It is **not a substitute for professional medical diagnosis**. Always consult a qualified healthcare provider for medical advice.
