// ─── Compare Page Logic ───

const panels = {
    a: {
        file: null, results: null, predId: null,
        uploadArea: document.getElementById("upload-a"),
        fileInput:  document.getElementById("file-a"),
        placeholder:document.getElementById("placeholder-a"),
        preview:    document.getElementById("preview-a"),
        img:        document.getElementById("img-a"),
        analyzeBtn: document.getElementById("analyze-a"),
        loader:     document.getElementById("loader-a"),
        resultsDiv: document.getElementById("results-a"),
        diagBadge:  document.getElementById("diag-badge-a"),
        benignVal:  document.getElementById("benign-a"),
        malignantVal: document.getElementById("malignant-a"),
        heatmapWrap: document.getElementById("heatmap-a-wrap"),
        heatmapImg: document.getElementById("heatmap-a"),
    },
    b: {
        file: null, results: null, predId: null,
        uploadArea: document.getElementById("upload-b"),
        fileInput:  document.getElementById("file-b"),
        placeholder:document.getElementById("placeholder-b"),
        preview:    document.getElementById("preview-b"),
        img:        document.getElementById("img-b"),
        analyzeBtn: document.getElementById("analyze-b"),
        loader:     document.getElementById("loader-b"),
        resultsDiv: document.getElementById("results-b"),
        diagBadge:  document.getElementById("diag-badge-b"),
        benignVal:  document.getElementById("benign-b"),
        malignantVal: document.getElementById("malignant-b"),
        heatmapWrap: document.getElementById("heatmap-b-wrap"),
        heatmapImg: document.getElementById("heatmap-b"),
    }
};

// ─── Wire up events for each panel ───

for (const [key, p] of Object.entries(panels)) {
    p.uploadArea.addEventListener("click", () => p.fileInput.click());
    p.uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        p.uploadArea.classList.add("dragover");
    });
    p.uploadArea.addEventListener("dragleave", () => p.uploadArea.classList.remove("dragover"));
    p.uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        p.uploadArea.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) handleFile(key, file);
    });
    p.fileInput.addEventListener("change", () => {
        if (p.fileInput.files[0]) handleFile(key, p.fileInput.files[0]);
    });
    p.analyzeBtn.addEventListener("click", () => analyzePanel(key));
}

function handleFile(key, file) {
    const p = panels[key];
    p.file = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        p.img.src = e.target.result;
        p.placeholder.classList.add("hidden");
        p.preview.classList.remove("hidden");
        p.analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearPanel(key) {
    const p = panels[key];
    p.file = null;
    p.results = null;
    p.predId = null;
    p.fileInput.value = "";
    p.img.src = "";
    p.placeholder.classList.remove("hidden");
    p.preview.classList.add("hidden");
    p.analyzeBtn.disabled = true;
    p.resultsDiv.classList.add("hidden");
    p.heatmapWrap.classList.add("hidden");
    document.getElementById("compare-summary").classList.add("hidden");
}

async function analyzePanel(key) {
    const p = panels[key];
    if (!p.file) return;

    p.analyzeBtn.disabled = true;
    p.loader.classList.remove("hidden");
    p.resultsDiv.classList.add("hidden");

    const formData = new FormData();
    formData.append("image", p.file);

    try {
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            alert("Error analyzing Scan " + key.toUpperCase() + ": " + data.error);
            return;
        }

        p.results = data;
        p.predId = data.prediction_id;
        displayPanelResults(key, data);
        checkShowComparison();

    } catch (err) {
        alert("Failed to analyze Scan " + key.toUpperCase() + ". Please try again.");
        console.error(err);
    } finally {
        p.loader.classList.add("hidden");
        p.analyzeBtn.disabled = false;
    }
}

function displayPanelResults(key, data) {
    const p = panels[key];
    p.resultsDiv.classList.remove("hidden");

    // Diagnosis badge
    p.diagBadge.className = "compare-diag-badge " + data.diagnosis.toLowerCase();
    p.diagBadge.innerHTML = `<i class="fas fa-stethoscope"></i> ${data.diagnosis}`;

    // Percentages with animation
    animateVal(p.benignVal, data.benign);
    animateVal(p.malignantVal, data.malignant);

    // Heatmap
    if (data.heatmap_b64) {
        p.heatmapImg.src = data.heatmap_b64;
        p.heatmapWrap.classList.remove("hidden");
    }
}

function animateVal(el, target) {
    let current = 0;
    const step = target / 40;
    const interval = setInterval(() => {
        current += step;
        if (current >= target) { current = target; clearInterval(interval); }
        el.textContent = current.toFixed(1);
    }, 25);
}

function checkShowComparison() {
    const a = panels.a.results;
    const b = panels.b.results;
    if (!a || !b) return;

    const summaryDiv = document.getElementById("compare-summary");
    const summaryText = document.getElementById("summary-text");
    summaryDiv.classList.remove("hidden");

    const diffBenign = Math.abs(a.benign - b.benign).toFixed(1);
    const diffMalignant = Math.abs(a.malignant - b.malignant).toFixed(1);

    let html = `<div class="compare-summary-grid">
        <div class="csumm-item">
            <span class="csumm-label">Scan A — ${a.diagnosis}</span>
            <span class="csumm-val benign-text">Benign: ${a.benign}%</span>
            <span class="csumm-val malignant-text">Malignant: ${a.malignant}%</span>
        </div>
        <div class="csumm-vs"><i class="fas fa-arrows-alt-h"></i></div>
        <div class="csumm-item">
            <span class="csumm-label">Scan B — ${b.diagnosis}</span>
            <span class="csumm-val benign-text">Benign: ${b.benign}%</span>
            <span class="csumm-val malignant-text">Malignant: ${b.malignant}%</span>
        </div>
    </div>`;

    if (a.diagnosis !== b.diagnosis) {
        html += `<div class="compare-highlight different">
            <i class="fas fa-exclamation-triangle"></i>
            <strong>Different diagnoses!</strong> Scan A is ${a.diagnosis} while Scan B is ${b.diagnosis}.
            Consult a doctor for a professional evaluation.
        </div>`;
    } else {
        html += `<div class="compare-highlight same">
            <i class="fas fa-check-circle"></i>
            Both scans show <strong>${a.diagnosis}</strong> results.
            Benign difference: ${diffBenign}% | Malignant difference: ${diffMalignant}%
        </div>`;
    }

    summaryText.innerHTML = html;
    summaryDiv.scrollIntoView({ behavior: "smooth" });
}

function downloadBoth() {
    if (panels.a.predId) window.open("/download-report/" + panels.a.predId, "_blank");
    if (panels.b.predId) window.open("/download-report/" + panels.b.predId, "_blank");
}
