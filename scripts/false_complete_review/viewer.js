"use strict";

const HUMAN_FIELDS = ["human_label", "human_failure_type", "human_notes"];
const EVIDENCE_FIELDS = [
  "task_incomplete",
  "failure_event_detected",
  "next_phase_entry",
  "terminal_like",
  "valid_recovery_attempt",
  "failure_types",
  "classification_reason",
  "false_complete_evidence_score",
  "decision_policy",
  "rule_version",
];

const elements = {
  progress: document.querySelector("#progress"),
  search: document.querySelector("#search"),
  classificationFilter: document.querySelector("#classification-filter"),
  priorityFilter: document.querySelector("#priority-filter"),
  recommendedOnly: document.querySelector("#recommended-only"),
  unreviewedOnly: document.querySelector("#unreviewed-only"),
  summary: document.querySelector("#summary"),
  previous: document.querySelector("#previous"),
  next: document.querySelector("#next"),
  episodeKey: document.querySelector("#episode-key"),
  taskDescription: document.querySelector("#task-description"),
  identityChips: document.querySelector("#identity-chips"),
  autoLabel: document.querySelector("#auto-label"),
  autoConfidence: document.querySelector("#auto-confidence"),
  videoGrid: document.querySelector("#video-grid"),
  evidence: document.querySelector("#evidence"),
  playAll: document.querySelector("#play-all"),
  pauseAll: document.querySelector("#pause-all"),
  form: document.querySelector("#annotation-form"),
  humanLabel: document.querySelector("#human-label"),
  humanFailureType: document.querySelector("#human-failure-type"),
  humanNotes: document.querySelector("#human-notes"),
  status: document.querySelector("#status"),
  exportButton: document.querySelector("#export"),
  importInput: document.querySelector("#import"),
};

const state = {
  manifest: null,
  visible: [],
  visiblePosition: 0,
  results: {},
};

function truthy(value) {
  return ["true", "1", "yes"].includes(String(value ?? "").trim().toLowerCase());
}

function episodeKey(row, rowNumber = 0) {
  if (String(row.review_id ?? "").trim()) return String(row.review_id).trim();
  if (String(row.suite ?? "").trim() && String(row.episode_index ?? "").trim()) {
    return `${String(row.suite).trim()}:${String(row.episode_index).trim()}`;
  }
  if (String(row.assignment_id ?? "").trim()) return String(row.assignment_id).trim();
  throw new Error(`CSV ${rowNumber}행에서 episode identity를 만들 수 없습니다.`);
}

function blankHuman() {
  return Object.fromEntries(HUMAN_FIELDS.map((field) => [field, ""]));
}

function humanFor(episode) {
  return state.results[episode.key] ?? Object.fromEntries(
    HUMAN_FIELDS.map((field) => [field, String(episode.row[field] ?? "").trim()])
  );
}

function reviewed(episode) {
  return Boolean(humanFor(episode).human_label);
}

function storageKey() {
  return `re_vla_rollout_viewer_${state.manifest.dataset_id}`;
}

function persist() {
  localStorage.setItem(storageKey(), JSON.stringify({ version: 1, results: state.results }));
}

function restore() {
  const raw = localStorage.getItem(storageKey());
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
    if (saved.version !== 1 || typeof saved.results !== "object" || saved.results === null) return;
    const valid = new Set(state.manifest.episodes.map((episode) => episode.key));
    for (const [key, result] of Object.entries(saved.results)) {
      if (!valid.has(key) || typeof result !== "object" || result === null) continue;
      const normalized = Object.fromEntries(HUMAN_FIELDS.map((field) => [field, String(result[field] ?? "").trim()]));
      if (normalized.human_label) state.results[key] = normalized;
    }
  } catch (error) {
    console.warn("저장된 review 상태를 무시했습니다.", error);
  }
}

function setStatus(message, kind = "") {
  elements.status.textContent = message;
  elements.status.className = kind;
}

function addOptions(select, values) {
  for (const value of [...new Set(values.filter(Boolean))].sort()) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
}

function renderSummary() {
  const total = state.manifest.episodes.length;
  const reviewedCount = state.manifest.episodes.filter(reviewed).length;
  const autoCandidates = state.manifest.episodes.filter((episode) => episode.row.classification === "false_complete").length;
  elements.summary.replaceChildren();
  const lines = [
    ["전체", total],
    ["현재 필터", state.visible.length],
    ["Human 완료", reviewedCount],
    ["Auto FC 후보", autoCandidates],
  ];
  for (const [label, value] of lines) {
    const line = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = String(value);
    line.append(`${label} `, strong);
    elements.summary.append(line);
  }
}

function applyFilters({ preserveKey = true } = {}) {
  const previousKey = preserveKey && state.visible.length
    ? state.manifest.episodes[state.visible[state.visiblePosition]].key
    : null;
  const query = elements.search.value.trim().toLowerCase();
  state.visible = state.manifest.episodes
    .map((episode, index) => [episode, index])
    .filter(([episode]) => {
      const row = episode.row;
      if (query && !JSON.stringify(row).toLowerCase().includes(query)) return false;
      if (elements.classificationFilter.value && row.classification !== elements.classificationFilter.value) return false;
      if (elements.priorityFilter.value && row.review_priority !== elements.priorityFilter.value) return false;
      if (elements.recommendedOnly.checked && !truthy(row.review_recommended)) return false;
      if (elements.unreviewedOnly.checked && reviewed(episode)) return false;
      return true;
    })
    .map(([, index]) => index);
  const preserved = previousKey
    ? state.visible.findIndex((index) => state.manifest.episodes[index].key === previousKey)
    : -1;
  state.visiblePosition = preserved >= 0 ? preserved : 0;
  renderSummary();
  renderCurrent();
}

function currentEpisode() {
  return state.visible.length ? state.manifest.episodes[state.visible[state.visiblePosition]] : null;
}

function chip(text) {
  const span = document.createElement("span");
  span.className = "chip";
  span.textContent = text;
  return span;
}

function renderVideos(episode) {
  elements.videoGrid.replaceChildren();
  for (const media of episode.videos) {
    const panel = document.createElement("article");
    panel.className = "video-panel";
    const label = document.createElement("p");
    label.textContent = media.label;
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.src = media.url;
    panel.append(label, video);
    elements.videoGrid.append(panel);
  }
}

function renderEvidence(row) {
  elements.evidence.replaceChildren();
  const fields = EVIDENCE_FIELDS.filter((field) => String(row[field] ?? "") !== "");
  if (!fields.length) {
    const item = document.createElement("div");
    item.textContent = "표시할 자동 evidence 열이 없습니다.";
    elements.evidence.append(item);
    return;
  }
  for (const field of fields) {
    const item = document.createElement("div");
    const term = document.createElement("dt");
    const value = document.createElement("dd");
    term.textContent = field;
    value.textContent = String(row[field]);
    item.append(term, value);
    elements.evidence.append(item);
  }
}

function renderCurrent() {
  const episode = currentEpisode();
  if (!episode) {
    elements.progress.textContent = "필터 결과 0개";
    elements.episodeKey.textContent = "—";
    elements.taskDescription.textContent = "조건에 맞는 episode가 없습니다.";
    elements.identityChips.replaceChildren();
    elements.videoGrid.replaceChildren();
    elements.evidence.replaceChildren();
    elements.previous.disabled = true;
    elements.next.disabled = true;
    return;
  }
  const row = episode.row;
  const human = humanFor(episode);
  elements.progress.textContent = `${state.visiblePosition + 1} / ${state.visible.length} · 전체 ${state.manifest.episodes.length}`;
  elements.episodeKey.textContent = episode.key;
  elements.taskDescription.textContent = row.task_description || `Episode ${row.episode_index || episode.key}`;
  elements.identityChips.replaceChildren();
  for (const [label, value] of [["suite", row.suite], ["task", row.task_id], ["episode", row.episode_index], ["seed", row.seed], ["priority", row.review_priority]]) {
    if (String(value ?? "") !== "") elements.identityChips.append(chip(`${label}: ${value}`));
  }
  elements.autoLabel.textContent = row.classification || "not provided";
  elements.autoConfidence.textContent = [row.confidence, row.false_complete_evidence_score].filter((value) => String(value ?? "") !== "").join(" · ") || "—";
  elements.humanLabel.value = human.human_label;
  elements.humanFailureType.value = human.human_failure_type;
  elements.humanNotes.value = human.human_notes;
  elements.previous.disabled = state.visiblePosition === 0;
  elements.next.disabled = state.visiblePosition === state.visible.length - 1;
  renderVideos(episode);
  renderEvidence(row);
  setStatus(reviewed(episode) ? "저장된 human review를 불러왔습니다." : "아직 저장하지 않았습니다.", reviewed(episode) ? "saved" : "");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function move(delta) {
  if (!state.visible.length) return;
  state.visiblePosition = Math.max(0, Math.min(state.visiblePosition + delta, state.visible.length - 1));
  renderCurrent();
}

function saveCurrent() {
  const episode = currentEpisode();
  if (!episode) return;
  const label = elements.humanLabel.value;
  if (!label) throw new Error("Human label을 선택하세요.");
  state.results[episode.key] = {
    human_label: label,
    human_failure_type: elements.humanFailureType.value.trim(),
    human_notes: elements.humanNotes.value.trim(),
  };
  persist();
  renderSummary();
  setStatus("브라우저에 저장했습니다.", "saved");
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportRows() {
  return state.manifest.episodes.map((episode) => ({ ...episode.row, ...humanFor(episode) }));
}

function toCsv(rows, fields) {
  return [fields, ...rows.map((row) => fields.map((field) => row[field] ?? ""))]
    .map((row) => row.map(csvCell).join(","))
    .join("\r\n") + "\r\n";
}

function downloadCsv() {
  const blob = new Blob(["\uFEFF", toCsv(exportRows(), state.manifest.fields)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.manifest.source_csv_name.replace(/\.csv$/i, "")}.human_review.csv`;
  link.click();
  URL.revokeObjectURL(url);
  setStatus("Human review CSV를 내려받았습니다.", "saved");
}

function parseCsv(text) {
  const source = text.replace(/^\uFEFF/, "");
  const table = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (quoted) {
      if (char === '"' && source[index + 1] === '"') { cell += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { row.push(cell); cell = ""; }
    else if (char === "\n") { row.push(cell.replace(/\r$/, "")); table.push(row); row = []; cell = ""; }
    else cell += char;
  }
  if (quoted) throw new Error("CSV 따옴표가 닫히지 않았습니다.");
  if (cell || row.length) { row.push(cell.replace(/\r$/, "")); table.push(row); }
  if (table.length < 2) throw new Error("CSV가 비어 있습니다.");
  const headers = table[0];
  return table.slice(1).filter((values) => values.some(Boolean)).map((values, index) => {
    if (values.length !== headers.length) throw new Error(`CSV ${index + 2}행의 열 수가 다릅니다.`);
    return Object.fromEntries(headers.map((header, column) => [header, values[column]]));
  });
}

function importRows(rows) {
  if (rows.length !== state.manifest.episodes.length) throw new Error("원본과 CSV episode 수가 다릅니다.");
  const imported = new Map(rows.map((row, index) => [episodeKey(row, index + 2), row]));
  if (imported.size !== state.manifest.episodes.length) throw new Error("CSV episode identity가 중복됩니다.");
  const results = {};
  for (const episode of state.manifest.episodes) {
    const row = imported.get(episode.key);
    if (!row) throw new Error(`CSV에 episode가 없습니다: ${episode.key}`);
    const label = String(row.human_label ?? "").trim();
    if (label) {
      results[episode.key] = {
        human_label: label,
        human_failure_type: String(row.human_failure_type ?? "").trim(),
        human_notes: String(row.human_notes ?? "").trim(),
      };
    }
  }
  state.results = results;
  persist();
  applyFilters();
  setStatus(`CSV를 불러왔습니다. Human 완료 ${Object.keys(results).length}개.`, "saved");
}

async function initialize() {
  const response = await fetch("episodes.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`episodes.json 로딩 실패 (${response.status})`);
  state.manifest = await response.json();
  if (state.manifest.schema_version !== "re-vla-local-rollout-viewer-v1") throw new Error("지원하지 않는 viewer schema입니다.");
  restore();
  addOptions(elements.classificationFilter, state.manifest.episodes.map((episode) => episode.row.classification));
  addOptions(elements.priorityFilter, state.manifest.episodes.map((episode) => episode.row.review_priority));
  const recommended = state.manifest.episodes.filter((episode) => truthy(episode.row.review_recommended)).length;
  elements.recommendedOnly.checked = recommended > 0 && recommended < state.manifest.episodes.length;
  applyFilters({ preserveKey: false });
}

for (const element of [elements.search, elements.classificationFilter, elements.priorityFilter, elements.recommendedOnly, elements.unreviewedOnly]) {
  element.addEventListener("input", () => applyFilters());
}
elements.previous.addEventListener("click", () => move(-1));
elements.next.addEventListener("click", () => move(1));
elements.playAll.addEventListener("click", () => {
  const videos = [...elements.videoGrid.querySelectorAll("video")];
  const anchor = videos[0]?.currentTime ?? 0;
  videos.forEach((video) => { video.currentTime = anchor; video.play().catch(() => {}); });
});
elements.pauseAll.addEventListener("click", () => elements.videoGrid.querySelectorAll("video").forEach((video) => video.pause()));
elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  try { saveCurrent(); if (state.visiblePosition < state.visible.length - 1) move(1); }
  catch (error) { setStatus(error.message, "error"); }
});
elements.exportButton.addEventListener("click", downloadCsv);
elements.importInput.addEventListener("change", async () => {
  try {
    const file = elements.importInput.files?.[0];
    if (file) importRows(parseCsv(await file.text()));
  } catch (error) { setStatus(error.message, "error"); }
  finally { elements.importInput.value = ""; }
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, textarea")) return;
  if (event.key === "ArrowLeft") move(-1);
  if (event.key === "ArrowRight") move(1);
});

initialize().catch((error) => {
  elements.progress.textContent = "로딩 실패";
  setStatus(error.message, "error");
  console.error(error);
});
