// ─── DOM Elements ───
const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const uploadPreview = document.getElementById("upload-preview");
const previewImg = document.getElementById("preview-img");
const removeBtn = document.getElementById("remove-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const loader = document.getElementById("loader");
const resultsSection = document.getElementById("results-section");
const summarySection = document.getElementById("summary-section");
const summaryBtn = document.getElementById("summary-btn");
const summaryLoader = document.getElementById("summary-loader");
const summaryContent = document.getElementById("summary-content");
const chatToggle = document.getElementById("chat-toggle");
const chatPanel = document.getElementById("chat-panel");
const chatClose = document.getElementById("chat-close");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const navChatBtn = document.getElementById("nav-chat-btn");

let selectedFile = null;
let lastResults = null;
let chatHistory = [];

// ─── File Upload ───

uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
        handleFile(file);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) {
        handleFile(fileInput.files[0]);
    }
});

removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadPlaceholder.classList.add("hidden");
        uploadPreview.classList.remove("hidden");
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    uploadPlaceholder.classList.remove("hidden");
    uploadPreview.classList.add("hidden");
    analyzeBtn.disabled = true;
    resultsSection.classList.add("hidden");
    summarySection.classList.add("hidden");
}

// ─── Prediction ───

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loader.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    summarySection.classList.add("hidden");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        lastResults = data;
        displayResults(data);
    } catch (err) {
        alert("Failed to analyze image. Please try again.");
        console.error(err);
    } finally {
        loader.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    resultsSection.classList.remove("hidden");

    // Diagnosis badge
    const badge = document.getElementById("diagnosis-badge");
    const diagText = document.getElementById("diagnosis-text");
    badge.className = "diagnosis-badge " + data.diagnosis.toLowerCase();
    diagText.textContent = data.diagnosis;

    // Animate progress rings
    animateRing("benign-ring", "benign-pct", data.benign);
    animateRing("malignant-ring", "malignant-pct", data.malignant);

    // Detail bars
    const barsContainer = document.getElementById("detail-bars");
    barsContainer.innerHTML = "";

    for (const [name, value] of Object.entries(data.detailed)) {
        const isBenign = name.toLowerCase().includes("benign");
        const item = document.createElement("div");
        item.className = "detail-bar-item";
        item.innerHTML = `
            <div class="detail-bar-label">
                <span>${name}</span>
                <span>${value}%</span>
            </div>
            <div class="detail-bar-track">
                <div class="detail-bar-fill ${isBenign ? "benign-bar" : "malignant-bar"}"
                     style="width: 0"></div>
            </div>
        `;
        barsContainer.appendChild(item);

        // Animate bar fill
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                item.querySelector(".detail-bar-fill").style.width = value + "%";
            });
        });
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function animateRing(ringId, pctId, value) {
    const ring = document.getElementById(ringId);
    const pctEl = document.getElementById(pctId);
    const circumference = 326.73;
    const offset = circumference - (value / 100) * circumference;

    ring.style.strokeDashoffset = circumference;
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            ring.style.strokeDashoffset = offset;
        });
    });

    // Animate number
    let current = 0;
    const target = Math.round(value * 10) / 10;
    const step = target / 40;
    const interval = setInterval(() => {
        current += step;
        if (current >= target) {
            current = target;
            clearInterval(interval);
        }
        pctEl.textContent = current.toFixed(1);
    }, 25);
}

// ─── Gemini Summary ───

summaryBtn.addEventListener("click", async () => {
    if (!lastResults) return;

    summarySection.classList.remove("hidden");
    summaryLoader.classList.remove("hidden");
    summaryContent.innerHTML = "";
    summarySection.scrollIntoView({ behavior: "smooth" });

    try {
        const res = await fetch("/get-summary", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(lastResults),
        });

        const data = await res.json();

        if (data.error) {
            summaryContent.innerHTML = `<p style="color:var(--red);">Error: ${data.error}</p>`;
        } else {
            summaryContent.innerHTML = marked.parse(data.summary);
        }
    } catch (err) {
        summaryContent.innerHTML =
            '<p style="color:var(--red);">Failed to get AI summary. Please try again.</p>';
        console.error(err);
    } finally {
        summaryLoader.classList.add("hidden");
    }
});

// ─── Chatbot ───

chatToggle.addEventListener("click", toggleChat);
chatClose.addEventListener("click", toggleChat);
navChatBtn.addEventListener("click", (e) => {
    e.preventDefault();
    if (chatPanel.classList.contains("hidden")) {
        toggleChat();
    }
});

function toggleChat() {
    chatPanel.classList.toggle("hidden");
}

chatSend.addEventListener("click", sendChatMessage);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    appendChatMsg("user", message);
    chatInput.value = "";

    // Add to history
    chatHistory.push({ role: "user", content: message });

    // Show typing indicator
    const typingEl = appendTyping();

    try {
        const payload = { message, history: chatHistory.slice(0, -1) };
        if (lastResults) {
            payload.detectionResults = lastResults;
        }

        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();
        typingEl.remove();

        if (data.error) {
            appendChatMsg("bot", "Sorry, I encountered an error: " + data.error);
        } else {
            appendChatMsg("bot", data.response, true);
            chatHistory.push({ role: "model", content: data.response });
        }
    } catch (err) {
        typingEl.remove();
        appendChatMsg("bot", "Sorry, something went wrong. Please try again.");
        console.error(err);
    }
}

function appendChatMsg(role, content, isMarkdown = false) {
    const msg = document.createElement("div");
    msg.className = "chat-msg " + role;

    const bubble = document.createElement("div");
    bubble.className = "chat-msg-content";

    if (isMarkdown) {
        bubble.innerHTML = marked.parse(content);
    } else {
        bubble.textContent = content;
    }

    msg.appendChild(bubble);
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function appendTyping() {
    const typing = document.createElement("div");
    typing.className = "chat-typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    chatMessages.appendChild(typing);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typing;
}
