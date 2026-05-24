/**
 * app.js – MediPredict frontend logic
 * Handles symptom selection, chip display, search filtering, and API calls.
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
const state = { selected: new Set() };

// ── DOM references (populated on DOMContentLoaded) ────────────────────────────
let dom = {};

document.addEventListener("DOMContentLoaded", () => {
  dom = {
    predictPanel:  document.getElementById("predictPanel"),
    emptyState:    document.getElementById("emptyState"),
    selectedChips: document.getElementById("selectedChips"),
    chipContainer: document.getElementById("chipContainer"),
    selectedCount: document.getElementById("selectedCount"),
    errorBox:      document.getElementById("errorBox"),
    resultsBox:    document.getElementById("resultsBox"),
    predictBtn:    document.getElementById("predictBtn"),
    btnText:       document.getElementById("btnText"),
    btnLoader:     document.getElementById("btnLoader"),
    consensusFill: document.getElementById("consensusFill"),
    consensusPct:  document.getElementById("consensusPct"),
    consensusDisease: document.getElementById("consensusDisease"),
    modelResults:  document.getElementById("modelResults"),
    consensusBox:  document.getElementById("consensusBox"),
  };
});

// ── Symptom checkbox change handler ──────────────────────────────────────────
function updateChips() {
  const checkboxes = document.querySelectorAll(".symptom-grid input[type='checkbox']");

  checkboxes.forEach(cb => {
    const label = cb.closest(".symptom-item");
    if (cb.checked) {
      state.selected.add(cb.value);
      label.classList.add("checked");
    } else {
      state.selected.delete(cb.value);
      label.classList.remove("checked");
    }
  });

  renderChips();
  togglePredictPanel();
  hideResults();
}

// ── Render selected chips ─────────────────────────────────────────────────────
function renderChips() {
  const container = dom.chipContainer;
  if (!container) return;

  container.innerHTML = "";
  state.selected.forEach(key => {
    const label = getSymptomLabel(key);
    const chip  = document.createElement("div");
    chip.className = "chip";
    chip.dataset.key = key;
    chip.innerHTML  = `<span>${label}</span><span class="chip-remove" onclick="removeSymptom('${key}')">×</span>`;
    container.appendChild(chip);
  });

  // Show/hide the chips area
  dom.selectedChips.style.display = state.selected.size > 0 ? "block" : "none";
  dom.selectedCount.textContent   = state.selected.size;
}

// ── Remove a symptom via chip ─────────────────────────────────────────────────
function removeSymptom(key) {
  state.selected.delete(key);

  // Uncheck the corresponding checkbox
  const cb = document.querySelector(`.symptom-grid input[value="${key}"]`);
  if (cb) {
    cb.checked = false;
    cb.closest(".symptom-item").classList.remove("checked");
  }

  renderChips();
  togglePredictPanel();
  hideResults();
}

// ── Clear all selected symptoms ───────────────────────────────────────────────
function clearAllSymptoms() {
  state.selected.clear();

  document.querySelectorAll(".symptom-grid input[type='checkbox']").forEach(cb => {
    cb.checked = false;
    cb.closest(".symptom-item").classList.remove("checked");
  });

  renderChips();
  togglePredictPanel();
  hideResults();
}

// ── Toggle predict panel visibility ──────────────────────────────────────────
function togglePredictPanel() {
  const hasSymptoms = state.selected.size > 0;

  if (dom.predictPanel) dom.predictPanel.style.display = hasSymptoms ? "block" : "none";
  if (dom.emptyState)   dom.emptyState.style.display   = hasSymptoms ? "none"  : "block";
}

// ── Filter symptoms grid by search query ─────────────────────────────────────
function filterSymptoms(query) {
  const q     = query.trim().toLowerCase();
  const items = document.querySelectorAll(".symptom-item");

  items.forEach(item => {
    const label = item.dataset.label || "";
    item.classList.toggle("symptom-hidden", q !== "" && !label.includes(q));
  });
}

// ── Get display label for a symptom key ──────────────────────────────────────
function getSymptomLabel(key) {
  const item = document.querySelector(`.symptom-item[data-key="${key}"] .symptom-text`);
  return item ? item.textContent : key.replace(/_/g, " ");
}

// ── Hide results & error ──────────────────────────────────────────────────────
function hideResults() {
  if (dom.resultsBox) dom.resultsBox.style.display = "none";
  if (dom.errorBox)   dom.errorBox.style.display   = "none";
}

// ── Run prediction (main API call) ────────────────────────────────────────────
async function runPrediction() {
  if (state.selected.size === 0) {
    showError("Please select at least one symptom.");
    return;
  }

  // Loading state
  setLoading(true);
  hideResults();

  try {
    const response = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ symptoms: [...state.selected] }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || "An unexpected error occurred.");
      return;
    }

    renderResults(data);

  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    setLoading(false);
  }
}

// ── Render prediction results ─────────────────────────────────────────────────
function renderResults(data) {
  // Consensus
  dom.consensusDisease.textContent = data.consensus_disease;

  // Animate confidence bar (rAF ensures transition fires)
  dom.consensusFill.style.width = "0%";
  dom.consensusPct.textContent  = "0%";

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const pct = data.consensus_confidence;
      dom.consensusFill.style.width = pct + "%";
      animateNumber(dom.consensusPct, 0, pct, 900, "%");
    });
  });

  // Per-model results
  dom.modelResults.innerHTML = "";
  const modelOrder = ["Random Forest", "Decision Tree", "Naive Bayes"];
  const delays     = [0, 100, 200];

  modelOrder.forEach((name, i) => {
    const result = data.results[name];
    if (!result) return;

    const item = document.createElement("div");
    item.className = "model-result-item";
    item.style.animationDelay = delays[i] + "ms";

    item.innerHTML = `
      <div class="model-result-header">
        <span class="model-result-name">
          ${modelIcon(name)} ${name}
        </span>
        <span class="model-result-pct" id="pct_${i}">0%</span>
      </div>
      <div class="model-result-disease">${result.disease}</div>
      <div class="mini-bar-track">
        <div class="mini-bar-fill" id="fill_${i}" style="width:0%"></div>
      </div>
    `;

    dom.modelResults.appendChild(item);

    // Animate bars
    setTimeout(() => {
      const fill = document.getElementById(`fill_${i}`);
      const pct  = document.getElementById(`pct_${i}`);
      fill.style.width = result.confidence + "%";
      animateNumber(pct, 0, result.confidence, 900, "%");
    }, 50 + delays[i]);
  });

  // Show results
  dom.resultsBox.style.display = "block";
  dom.errorBox.style.display   = "none";

  // Smooth scroll to results on mobile
  if (window.innerWidth < 1200) {
    setTimeout(() => dom.resultsBox.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);
  }
}

// ── Animate a number counter ──────────────────────────────────────────────────
function animateNumber(el, from, to, duration, suffix = "") {
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const value    = from + (to - from) * eased;
    el.textContent = value.toFixed(1) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Model icon helper ─────────────────────────────────────────────────────────
function modelIcon(name) {
  const icons = {
    "Random Forest": "bi-tree-fill",
    "Decision Tree": "bi-diagram-3-fill",
    "Naive Bayes":   "bi-calculator-fill",
  };
  return `<i class="bi ${icons[name] || 'bi-cpu-fill'}"></i>`;
}

// ── Loading state ─────────────────────────────────────────────────────────────
function setLoading(on) {
  dom.predictBtn.disabled     = on;
  dom.btnText.style.display   = on ? "none"   : "flex";
  dom.btnLoader.style.display = on ? "inline" : "none";
}

// ── Error display ─────────────────────────────────────────────────────────────
function showError(msg) {
  dom.errorBox.innerHTML     = `<i class="bi bi-exclamation-circle-fill"></i> ${msg}`;
  dom.errorBox.style.display = "flex";
  dom.resultsBox.style.display = "none";
}
