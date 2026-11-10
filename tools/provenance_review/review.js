const state = {
  pairs: [],
  decisions: {},
  filtered: [],
  currentIndex: 0,
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function normalizePriority(p) {
  if (p.startsWith("Priority 1")) return "Priority 1";
  if (p.startsWith("Priority 2")) return "Priority 2";
  if (p.startsWith("Priority 3")) return "Priority 3";
  return p;
}

function isReviewed(pair) {
  return !!state.decisions[pair.pair_id];
}

function labelCombo(pair) {
  const a = pair.label_a.toLowerCase();
  const b = pair.label_b.toLowerCase();
  const aNorm = a.includes("tuberculosis") ? "tb" : "normal";
  const bNorm = b.includes("tuberculosis") ? "tb" : "normal";
  if (aNorm !== bNorm) return "cross";
  if (aNorm === "tb") return "tb-tb";
  return "normal-normal";
}

function cohortMatch(pair, cohort) {
  if (!cohort) return true;
  return pair.image_a.cohort === cohort || pair.image_b.cohort === cohort;
}

function applyFilters() {
  const priority = document.getElementById("filterPriority").value;
  const status = document.getElementById("filterStatus").value;
  const cohort = document.getElementById("filterCohort").value;
  const labels = document.getElementById("filterLabels").value;

  state.filtered = state.pairs.filter(pair => {
    if (priority && normalizePriority(pair.priority) !== priority) return false;
    if (cohort && !cohortMatch(pair, cohort)) return false;
    if (labels && labelCombo(pair) !== labels) return false;
    const reviewed = isReviewed(pair);
    if (status === "ai_uncertain") return pair.ai_assistance?.recommendation === "UNRESOLVED";
    if (status === "unreviewed" && reviewed) return false;
    if (status === "reviewed" && !reviewed) return false;
    if (status && status !== "unreviewed" && status !== "reviewed") {
      const d = state.decisions[pair.pair_id];
      if (!d || d.decision !== status) return false;
    }
    return true;
  });

  if (state.filtered.length === 0) {
    state.currentIndex = 0;
  } else if (state.currentIndex >= state.filtered.length) {
    state.currentIndex = state.filtered.length - 1;
  }
  updateDashboard();
  renderCurrent();
}

function updateDashboard() {
  document.getElementById("statTotal").textContent = state.pairs.length;
  document.getElementById("statP1").textContent = state.pairs.filter(p => normalizePriority(p.priority) === "Priority 1").length;
  document.getElementById("statP2").textContent = state.pairs.filter(p => normalizePriority(p.priority) === "Priority 2").length;
  document.getElementById("statP3").textContent = state.pairs.filter(p => normalizePriority(p.priority) === "Priority 3").length;

  const decisions = Object.values(state.decisions);
  document.getElementById("statReviewed").textContent = decisions.length;
  document.getElementById("statRemaining").textContent = state.pairs.length - decisions.length;
  document.getElementById("statConfirmed").textContent = decisions.filter(d => d.decision === "CONFIRMED_DUPLICATE").length;
  document.getElementById("statProbable").textContent = decisions.filter(d => d.decision === "PROBABLE_DUPLICATE").length;
  document.getElementById("statSimilar").textContent = decisions.filter(d => d.decision === "SIMILAR_BUT_DISTINCT").length;
  document.getElementById("statUnresolved").textContent = decisions.filter(d => d.decision === "UNRESOLVED").length;
  document.getElementById("statConflicts").textContent = decisions.filter(d => d.label_conflict_status === "potential_label_conflict").length;
}

function imageUrl(cohort, filename) {
  return `/image?cohort=${encodeURIComponent(cohort)}&file=${encodeURIComponent(filename)}`;
}

function renderCurrent() {
  if (state.filtered.length === 0) {
    document.getElementById("pairPosition").textContent = "No pairs in current filter.";
    return;
  }
  const pair = state.filtered[state.currentIndex];
  document.getElementById("pairPosition").textContent = `${state.currentIndex + 1} / ${state.filtered.length} (global pair ${pair.pair_id})`;

  document.getElementById("mPairId").textContent = pair.pair_id;
  document.getElementById("mPriority").textContent = pair.priority;
  document.getElementById("mPhash").textContent = pair.phash_distance;
  document.getElementById("mNcc").textContent = pair.ncc_512;
  document.getElementById("mSsim").textContent = pair.ssim_512;
  document.getElementById("mEcc").textContent = pair.ecc;
  document.getElementById("mStatus").textContent = pair.existing_status;
  const assistance = pair.ai_assistance;
  document.getElementById("aiAssistance").textContent = assistance
    ? `${assistance.recommendation} (AI suggestion only). ${assistance.basis} ${assistance.cross_label_note} Inspection: ${assistance.inspection}.`
    : "No current AI screening notes are available for this pair.";

  const contact = document.getElementById("mContact");
  if (pair.contact_sheet_name) {
    contact.href = `/contact?pair=${encodeURIComponent(pair.pair_id)}`;
    contact.textContent = pair.contact_sheet_name;
  } else {
    contact.href = "#";
    contact.textContent = "none";
  }

  const imgA = document.getElementById("imgA");
  const imgB = document.getElementById("imgB");
  const imgDiff = document.getElementById("imgDiff");
  imgA.src = imageUrl(pair.image_a.cohort, pair.image_a.filename);
  imgB.src = imageUrl(pair.image_b.cohort, pair.image_b.filename);
  imgDiff.src = `/diff?pair=${encodeURIComponent(pair.pair_id)}`;

  document.getElementById("fullA").href = imageUrl(pair.image_a.cohort, pair.image_a.filename);
  document.getElementById("fullB").href = imageUrl(pair.image_b.cohort, pair.image_b.filename);

  document.getElementById("metaA").textContent =
    `File: ${pair.image_a.filename}\nCohort: ${pair.image_a.cohort}\nLabel: ${pair.label_a}\nDimensions: ${pair.image_a.dimensions}\nSHA-256: ${pair.image_a.sha256}`;
  document.getElementById("metaB").textContent =
    `File: ${pair.image_b.filename}\nCohort: ${pair.image_b.cohort}\nLabel: ${pair.label_b}\nDimensions: ${pair.image_b.dimensions}\nSHA-256: ${pair.image_b.sha256}`;

  const d = state.decisions[pair.pair_id] || {};
  document.getElementById("labelConflict").value = d.label_conflict_status || "none";
  document.getElementById("confidence").value = d.reviewer_confidence || "Medium";
  document.getElementById("reviewerId").value = d.reviewer_identifier || "";
  document.getElementById("notes").value = d.reviewer_notes || "";
  document.getElementById("saveStatus").textContent = d.decision ? `Saved: ${d.decision}` : "Not reviewed";
}

async function loadAll() {
  const [pairs, decisions] = await Promise.all([
    api("/api/pairs"),
    api("/api/decisions"),
  ]);
  state.pairs = pairs;
  state.decisions = decisions || {};
  applyFilters();
}

async function saveDecision() {
  const pair = state.filtered[state.currentIndex];
  if (!pair) return;
  const decisionBtn = document.querySelector(".decision-buttons button.selected");
  const decision = decisionBtn ? decisionBtn.dataset.decision : "";
  if (!decision) {
    document.getElementById("saveStatus").textContent = "Choose a primary decision first.";
    return;
  }
  const body = {
    pair_id: pair.pair_id,
    decision,
    label_conflict_status: document.getElementById("labelConflict").value,
    reviewer_confidence: document.getElementById("confidence").value,
    reviewer_notes: document.getElementById("notes").value,
    reviewer_identifier: document.getElementById("reviewerId").value,
  };
  await api("/api/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.decisions[pair.pair_id] = {
    pair_id: pair.pair_id,
    decision,
    label_conflict_status: body.label_conflict_status,
    reviewer_confidence: body.reviewer_confidence,
    reviewer_notes: body.reviewer_notes,
    review_timestamp: new Date().toISOString(),
    reviewer_identifier: body.reviewer_identifier,
  };
  document.getElementById("saveStatus").textContent = `Saved: ${decision}`;
  updateDashboard();
}

function selectDecision(btn) {
  document.querySelectorAll(".decision-buttons button").forEach(b => b.classList.remove("selected"));
  btn.classList.add("selected");
}

function go(delta) {
  if (state.filtered.length === 0) return;
  state.currentIndex = Math.max(0, Math.min(state.filtered.length - 1, state.currentIndex + delta));
  renderCurrent();
}

function returnToUnresolved() {
  document.getElementById("filterStatus").value = "UNRESOLVED";
  applyFilters();
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadAll();

  document.getElementById("filterPriority").addEventListener("change", applyFilters);
  document.getElementById("filterStatus").addEventListener("change", applyFilters);
  document.getElementById("filterCohort").addEventListener("change", applyFilters);
  document.getElementById("filterLabels").addEventListener("change", applyFilters);

  document.getElementById("btnPrev").addEventListener("click", () => go(-1));
  document.getElementById("btnNext").addEventListener("click", () => go(1));
  document.getElementById("btnSkip").addEventListener("click", () => go(1));
  document.getElementById("btnReturnUnresolved").addEventListener("click", returnToUnresolved);
  document.getElementById("btnSave").addEventListener("click", () => saveDecision().catch(error => {
    document.getElementById("saveStatus").textContent = `Not saved: ${error.message}`;
  }));

  document.querySelectorAll(".decision-buttons button").forEach(btn => {
    btn.addEventListener("click", () => selectDecision(btn));
  });
});
