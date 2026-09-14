const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("fileInput");
const sourcePlayer = $("sourcePlayer");
const sourceAudio = $("sourceAudio");
const sourceFilename = $("sourceFilename");
const styleGrid = $("styleGrid");
const runBtn = $("runBtn");
const progressBox = $("progress");
const progressPulse = $("progressPulse");
const progressElapsed = $("progressElapsed");
const progressStageLabel = $("progressStageLabel");
const resultsEmpty = $("resultsEmpty");
const styleResults = $("styleResults");
const toast = $("toast");
const statusDot = $("statusDot");
const statusText = $("statusText");
const transcribeBtn = $("transcribeBtn");
const lyricsPanel = $("lyricsPanel");
const lyricsText = $("lyricsText");
const stylesPanel = $("stylesPanel");
const lyricsProgress = $("lyricsProgress");
const lyricsPulse = $("lyricsPulse");
const lyricsElapsed = $("lyricsElapsed");
const lyricsStageLabel = $("lyricsStageLabel");

let selectedFile = null;
let sessionId = null;
let toastTimer = null;
const styleRowById = {};

function unlock(panel) {
  panel.classList.remove("is-locked");
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 6000);
}

function setBusy(btn, busyLabel, isBusy, idleLabel) {
  btn.disabled = isBusy;
  btn.textContent = isBusy ? busyLabel : idleLabel;
}

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---- Source selection ----

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  sourceAudio.src = URL.createObjectURL(file);
  sourceFilename.textContent = file.name;
  sourcePlayer.hidden = false;
  transcribeBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ---- Style picker ----

async function loadStyles() {
  try {
    const res = await fetch("/api/styles");
    const data = await res.json();
    styleGrid.innerHTML = "";
    data.styles.forEach((style) => {
      const label = document.createElement("label");
      label.className = "style-option";
      label.dataset.styleId = style.id;
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = style.id;
      input.addEventListener("change", () => {
        label.classList.toggle("is-checked", input.checked);
      });
      label.appendChild(input);
      label.appendChild(document.createTextNode(style.label));
      styleGrid.appendChild(label);
    });
  } catch {
    styleGrid.innerHTML = '<span class="hint">Could not load style presets. Is the app server running?</span>';
  }
}
loadStyles();

function selectedStyleIds() {
  return Array.from(styleGrid.querySelectorAll("input:checked")).map((el) => el.value);
}

// ---- Shared SSE progress watcher (single jobs and batches alike) ----

function watchEvents(eventsUrl, onProgress, onReconnecting) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(eventsUrl);

    source.addEventListener("open", () => {
      if (onReconnecting) onReconnecting(false);
    });

    source.addEventListener("progress", (evt) => {
      onProgress(JSON.parse(evt.data));
    });

    source.addEventListener("final", (evt) => {
      const data = JSON.parse(evt.data);
      source.close();
      if (data.status === "error") {
        reject(new Error(data.error || "Cover run failed."));
      } else {
        resolve();
      }
    });

    // A dropped connection (a backgrounded tab throttling, a brief network
    // blip) is transient: the browser retries on its own as long as we
    // don't close() here. The batch keeps running server-side regardless of
    // whether anyone's watching, so just surface "reconnecting" and let it
    // recover instead of giving up on the first hiccup.
    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) {
        reject(new Error("Lost connection to the progress stream."));
      } else if (onReconnecting) {
        onReconnecting(true);
      }
    };
  });
}

function renderStyleResult(style) {
  let row = styleRowById[style.id];
  if (!row) {
    row = document.createElement("div");
    row.className = "style-result";
    row.innerHTML = `
      <span class="style-result-label">${style.label}</span>
      <span class="style-result-status"></span>
    `;
    styleResults.appendChild(row);
    styleRowById[style.id] = row;
  }

  row.className = `style-result is-${style.status}`;
  const statusEl = row.querySelector(".style-result-status");
  const existingAudio = row.querySelector("audio");
  const existingLink = row.querySelector("a");

  if (style.status === "queued") {
    statusEl.textContent = "queued";
  } else if (style.status === "running") {
    statusEl.textContent = "generating…";
  } else if (style.status === "error") {
    statusEl.textContent = style.error || "failed";
  } else if (style.status === "done" && style.filename) {
    statusEl.textContent = "";
    if (!existingAudio) {
      const audioEl = document.createElement("audio");
      audioEl.controls = true;
      audioEl.src = `/completed/${style.filename}`;
      row.appendChild(audioEl);
    }
    if (!existingLink) {
      const link = document.createElement("a");
      link.href = `/completed/${style.filename}`;
      link.textContent = style.filename;
      link.target = "_blank";
      row.appendChild(link);
    }
  }
}

// ---- Transcribe lyrics (review/edit before generating) ----

transcribeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setBusy(transcribeBtn, "Transcribing…", true, "Transcribe lyrics");
  lyricsProgress.hidden = false;
  lyricsPulse.classList.remove("is-active");
  lyricsElapsed.textContent = "0:00";
  lyricsStageLabel.textContent = "Transcribing lyrics…";

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const startRes = await fetch("/api/covers/prepare", { method: "POST", body: form });
    if (!startRes.ok) {
      const detail = await startRes.text();
      throw new Error(detail || `Could not start (${startRes.status})`);
    }
    const { session_id, job_id } = await startRes.json();
    sessionId = session_id;

    await watchEvents(
      `/api/jobs/${job_id}/events`,
      (data) => {
        lyricsElapsed.textContent = formatElapsed(data.elapsed_s);
        lyricsPulse.classList.toggle("is-active", !!data.engine_active);
      },
      (isReconnecting) => {
        if (isReconnecting) lyricsStageLabel.textContent = "Reconnecting… still transcribing";
      }
    );

    const res = await fetch(`/api/jobs/${job_id}/result`);
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || `Transcription failed (${res.status})`);
    }
    const data = await res.json();
    lyricsText.value = data.lyrics || "";
    unlock(lyricsPanel);
    unlock(stylesPanel);
    runBtn.disabled = false;
  } catch (err) {
    showToast(err.message || "Lyrics transcription failed.");
  } finally {
    setBusy(transcribeBtn, "Transcribing…", false, "Transcribe lyrics");
    lyricsPulse.classList.remove("is-active");
  }
});

// ---- Generate (uses the session from the transcribe step + edited lyrics) ----

runBtn.addEventListener("click", async () => {
  if (!sessionId) { showToast("Transcribe lyrics first."); return; }
  const styleIds = selectedStyleIds();
  if (styleIds.length === 0) { showToast("Pick at least one style."); return; }

  setBusy(runBtn, "Running…", true, "Make covers");
  progressBox.hidden = false;
  progressPulse.classList.remove("is-active");
  progressElapsed.textContent = "0:00";
  progressStageLabel.textContent = "Starting…";
  resultsEmpty.hidden = true;
  styleResults.innerHTML = "";
  Object.keys(styleRowById).forEach((k) => delete styleRowById[k]);

  const form = new FormData();
  form.append("lyrics", lyricsText.value);
  form.append("styles", JSON.stringify(styleIds));
  form.append("cot", $("cotSelect").value);
  form.append("seed", $("seedInput").value || "831001");
  form.append("num_inference_steps", $("stepsInput").value || "16");
  form.append("flip_key", $("flipKeyInput").checked ? "true" : "false");
  form.append("vocal", $("vocalSelect").value);

  try {
    const startRes = await fetch(`/api/covers/${sessionId}/generate`, { method: "POST", body: form });
    if (!startRes.ok) {
      const detail = await startRes.text();
      throw new Error(detail || `Could not start (${startRes.status})`);
    }
    const { batch_id } = await startRes.json();
    sessionId = null; // the session is consumed server-side once generation starts

    await watchEvents(
      `/api/covers/${batch_id}/events`,
      (data) => {
        progressElapsed.textContent = formatElapsed(data.elapsed_s);
        progressPulse.classList.toggle("is-active", !!data.engine_active);
        progressStageLabel.textContent = data.stage;
        data.styles.forEach(renderStyleResult);
      },
      (isReconnecting) => {
        if (isReconnecting) {
          progressStageLabel.textContent += " (reconnecting… the run itself is still going)";
        }
      }
    );

    showToast("Cover run complete.");
  } catch (err) {
    showToast(err.message || "Cover run failed.");
  } finally {
    setBusy(runBtn, "Running…", false, "Make covers");
    progressPulse.classList.remove("is-active");
  }
});

// ---- Engine status ----

async function pollHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ok) {
      statusDot.className = "status-dot ok";
      statusText.textContent = data.yue2_loaded ? "engine ready" : "engine ready · model loads on first use";
    } else {
      statusDot.className = "status-dot bad";
      statusText.textContent = "engine unreachable";
    }
  } catch {
    statusDot.className = "status-dot bad";
    statusText.textContent = "engine unreachable";
  }
}

pollHealth();
setInterval(pollHealth, 15000);
