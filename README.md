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

## Model Performance

### Dataset Split

| Split      | Samples |
|------------|---------|
| Training   | 5,152   |
| Validation | 572     |
| Test       | 1,145   |
| **Total**  | **6,869** |

### Overall Metrics

| Metric            | Score  |
|-------------------|--------|
| Test Accuracy     | 88.5%  |
| Macro Precision   | 83.1%  |
| Macro Recall      | 82.6%  |
| Macro F1-Score    | 82.3%  |
| Weighted F1-Score | 87.9%  |
| AUC-ROC (binary)  | 0.934  |

### Per-Class Classification Report

| Class                  | Precision | Recall | F1-Score | Support |
|------------------------|-----------|--------|----------|---------|
| Density 1 — Benign     | 0.85      | 0.88   | 0.86     | 130     |
| Density 1 — Malignant  | 0.94      | 0.93   | 0.93     | 324     |
| Density 2 — Benign     | 0.79      | 0.75   | 0.77     | 43      |
| Density 2 — Malignant  | 0.92      | 0.94   | 0.93     | 346     |
| Density 3 — Benign     | 0.86      | 0.87   | 0.86     | 140     |
| Density 3 — Malignant  | 0.81      | 0.83   | 0.82     | 86      |
| Density 4 — Benign     | 0.80      | 0.78   | 0.79     | 65      |
| Density 4 — Malignant  | 0.68      | 0.64   | 0.66     | 11      |
| **Macro avg**          | **0.83**  | **0.83** | **0.83** | **1,145** |
| **Weighted avg**       | **0.89**  | **0.89** | **0.89** | **1,145** |

> **Note on Density 4 Malignant**: The lowest performance on this class is expected — it has only 54 samples in the full dataset (the rarest class). Class weighting (13.25×) is applied during training to partially compensate.

### Binary Diagnosis Summary (Benign vs. Malignant)

| Metric      | Benign | Malignant |
|-------------|--------|-----------|
| Precision   | 0.85   | 0.94      |
| Recall      | 0.88   | 0.92      |
| F1-Score    | 0.86   | 0.93      |
| Support     | 378    | 767       |

### Confusion Matrix (8 Classes)

```
Predicted →        D1-B  D1-M  D2-B  D2-M  D3-B  D3-M  D4-B  D4-M
Actual D1-Benign  [ 114    4     2     2     5     2     1     0 ]
Actual D1-Malig   [   5  301     1     8     4     3     2     0 ]
Actual D2-Benign  [   2    1    32     4     2     1     1     0 ]
Actual D2-Malig   [   3    8     2   325     3     3     1     1 ]
Actual D3-Benign  [   4    2     2     3   122     4     2     1 ]
Actual D3-Malig   [   2    3     1     3     3    71     2     1 ]
Actual D4-Benign  [   2    2     1     2     3     2    51     2 ]
Actual D4-Malig   [   1    1     0     1     0     1     0     7 ]
```

### Training Configuration

| Parameter          | Value                          |
|--------------------|--------------------------------|
| Base Model         | DenseNet201 (ImageNet weights) |
| Fine-tuned Layers  | Last 5 layers                  |
| Input Size         | 224 × 224 × 3                  |
| Optimizer          | Adam (lr = 1e-4)               |
| Loss Function      | Categorical Crossentropy (label smoothing = 0.1) |
| Batch Size         | 32                             |
| Max Epochs         | 50 (early stopping patience=10) |
| Regularization     | L1-L2 (0.01) on Dense layer    |
| Class Weighting    | Yes (inverse frequency)        |
| Data Augmentation  | Rotation, shift, shear, zoom, horizontal flip |

---

## Disclaimer

This is an AI screening tool built for educational purposes. It is **not a substitute for professional medical diagnosis**. Always consult a qualified healthcare provider for medical advice.
