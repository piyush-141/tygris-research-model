/**
 * Tiger Vision AI - Clean & Minimal Console Logic
 */

let currentEventData = null;
let activeTrackIndex = 0;
let currentPreset = "multi_tiger";
let uploadedBase64Frames = [];
let mapInstance = null;
let mapMarker = null;

const CAMERA_COORDS = {
  "PTR-CORE-EP-01": [21.685, 79.310],
  "PTR-CORE-CP-01": [21.655, 79.320],
  "PTR-CORE-WP-01": [21.640, 79.280],
  "PTR-BUFF-01": [21.710, 79.360]
};

const SKELETON_CONNECTIONS = [
  ["nose", "left_eye"], ["nose", "right_eye"], ["nose", "neck"],
  ["neck", "left_shoulder"], ["neck", "right_shoulder"],
  ["left_shoulder", "left_front_paw"], ["right_shoulder", "right_front_paw"],
  ["neck", "spine_mid"], ["spine_mid", "spine_base"],
  ["spine_base", "left_hip"], ["spine_base", "right_hip"],
  ["left_hip", "left_hind_paw"], ["right_hip", "right_hind_paw"],
  ["spine_base", "tail_base"], ["tail_base", "tail_tip"]
];

document.addEventListener("DOMContentLoaded", () => {
  setupDropZone();
  initMap();
  loadPreset("multi_tiger");
});

function setupDropZone() {
  const dropZone = document.getElementById("drop-zone");
  if (!dropZone) return;

  ["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    }, false);
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    }, false);
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      loadFiles(files);
    }
  }, false);
}

function triggerFileInput() {
  document.getElementById("file-input").click();
}

function handleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    loadFiles(files);
  }
}

function loadFiles(files) {
  uploadedBase64Frames = [];
  const fileArray = Array.from(files).slice(0, 8);
  let loaded = 0;

  fileArray.forEach(file => {
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedBase64Frames.push(e.target.result);
      loaded++;
      if (loaded === fileArray.length) {
        renderThumbs();
        currentPreset = "custom_upload";
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        runAnalysis();
      }
    };
    reader.readAsDataURL(file);
  });
}

function renderThumbs() {
  const row = document.getElementById("drop-thumbs-row");
  const container = document.getElementById("drop-thumbs-container");
  container.innerHTML = "";

  if (uploadedBase64Frames.length > 0) {
    row.style.display = "flex";
    uploadedBase64Frames.forEach((b64, idx) => {
      const img = document.createElement("img");
      img.className = "drop-thumb";
      img.src = b64;
      container.appendChild(img);
    });
  } else {
    row.style.display = "none";
  }
}

function clearUpload(event) {
  if (event) {
    event.stopPropagation();
    event.preventDefault();
  }
  uploadedBase64Frames = [];
  document.getElementById("file-input").value = "";
  renderThumbs();
  loadPreset("multi_tiger");
}

function loadPreset(presetName) {
  currentPreset = presetName;
  uploadedBase64Frames = [];
  renderThumbs();

  document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
  event && event.target && event.target.closest(".preset-btn") && event.target.closest(".preset-btn").classList.add("active");

  clearResults();
}

function clearResults() {
  currentEventData = null;
  const primaryAlert = document.getElementById("primary-alert");
  if (primaryAlert) primaryAlert.style.display = "none";
  
  const tracksContainer = document.getElementById("tiger-tabs-container");
  if (tracksContainer) tracksContainer.style.display = "none";
  
  const resultHero = document.getElementById("result-hero");
  if (resultHero) resultHero.style.display = "none";

  const visualGrid = document.getElementById("visual-grid");
  if (visualGrid) visualGrid.style.display = "none";
  
  const detailsCard = document.getElementById("details-card");
  if (detailsCard) detailsCard.style.display = "none";
  
  const banner = document.getElementById("status-banner");
  if (banner) banner.style.display = "none";
}

async function runAnalysis() {
  const btn = document.getElementById("btn-analyze");
  const origText = btn.innerHTML;
  btn.innerHTML = `<span>⏳</span> Analyzing...`;
  btn.disabled = true;

  const cam = document.getElementById("camera-select").value;
  const coords = CAMERA_COORDS[cam] || [21.685, 79.310];

  const payload = {
    event_type: currentPreset,
    base64_frames: uploadedBase64Frames,
    camera_id: cam,
    latitude: coords[0],
    longitude: coords[1],
    timestamp: new Date().toISOString()
  };

  const banner = document.getElementById("status-banner");

  try {
    const res = await fetch("/api/process_event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.error || `HTTP Error ${res.status}`);
    }

    const data = await res.json();
    currentEventData = data;
    if (banner) banner.style.display = "none";
    renderResults(data);
  } catch (err) {
    console.error("Analysis failed:", err);
    if (banner) {
      banner.style.display = "block";
      banner.style.background = "rgba(244, 63, 94, 0.15)";
      banner.style.border = "1px solid rgba(244, 63, 94, 0.4)";
      banner.style.color = "#f43f5e";
      banner.innerHTML = `⚠️ <b>Notice:</b> ${err.message}`;
    }
  } finally {
    btn.innerHTML = origText;
    btn.disabled = false;
  }
}

function renderResults(data) {
  if (!data) return;

  // Render Tiger Tabs if multiple tigers detected
  const tabsContainer = document.getElementById("tiger-tabs-container");
  tabsContainer.innerHTML = "";

  if (data.tigers && data.tigers.length > 1) {
    tabsContainer.style.display = "flex";
    data.tigers.forEach((t, idx) => {
      const btn = document.createElement("button");
      btn.className = `tiger-tab-btn ${idx === activeTrackIndex ? "active" : ""}`;
      const name = t.tiger_id ? `Tiger #${t.tiger_id}` : "Unknown Individual";
      btn.innerHTML = `🐅 ${t.track_id} (${name})`;
      btn.onclick = () => selectTrack(idx);
      tabsContainer.appendChild(btn);
    });
  } else {
    tabsContainer.style.display = "none";
  }

  // Render active tiger or fallback
  if (data.tigers && data.tigers.length > 0) {
    const idx = Math.min(activeTrackIndex, data.tigers.length - 1);
    renderTiger(data.tigers[idx]);
  } else {
    renderNonTarget(data);
  }

  // Show result panels
  document.getElementById("result-hero").style.display = "flex";
  document.getElementById("visual-grid").style.display = "grid";
  document.getElementById("details-card").style.display = "block";

  // Storage Telemetry
  if (data.telemetry) {
    const tel = data.telemetry;
    document.getElementById("storage-saved-headline").textContent = 
      `⚡ ${tel.storage_reduction_pct.toFixed(1)}% Storage Saved vs Full Video Recording`;
    document.getElementById("tel-raw").textContent = `${tel.raw_video_frames} frames`;
    document.getElementById("tel-screened").textContent = `${tel.sampled_screening_frames} frames`;
    document.getElementById("tel-retained").textContent = `${tel.retained_final_frames} images`;
    document.getElementById("tel-time").textContent = `${(tel.processing_time_sec * 1000).toFixed(1)} ms`;
  }

  // GIS Map
  if (mapInstance && data.latitude && data.longitude) {
    if (mapMarker) mapInstance.removeLayer(mapMarker);
    mapMarker = L.circleMarker([data.latitude, data.longitude], {
      radius: 9,
      color: data.tiger_detected ? "#10b981" : "#f59e0b",
      fillColor: data.tiger_detected ? "#10b981" : "#f59e0b",
      fillOpacity: 0.9
    }).addTo(mapInstance);
    mapInstance.setView([data.latitude, data.longitude], 12);
  }
}

function selectTrack(idx) {
  activeTrackIndex = idx;
  if (!currentEventData || !currentEventData.tigers) return;

  const tabs = document.querySelectorAll(".tiger-tab-btn");
  tabs.forEach((t, i) => t.classList.toggle("active", i === idx));

  if (idx < currentEventData.tigers.length) {
    renderTiger(currentEventData.tigers[idx]);
  }
}

function renderTiger(tiger) {
  const hero = document.getElementById("result-hero");
  const tag = document.getElementById("result-status-tag");
  const name = document.getElementById("result-name");
  const meta = document.getElementById("result-meta");
  const score = document.getElementById("result-score-val");

  if (tiger.status === "KNOWN") {
    hero.className = "result-hero";
    tag.textContent = "✓ KNOWN INDIVIDUAL CONFIRMED";
    tag.style.color = "var(--emerald)";
    name.textContent = `Identified as Tiger #${tiger.tiger_id}`;
    meta.textContent = `${tiger.track_id} · Best Frame Selected (Quality: ${tiger.quality_score.toFixed(1)}/100) · Pose: ${(tiger.pose || 'Walking').toUpperCase()} · Logged to Pench DB`;
    score.textContent = `${(tiger.confidence * 100).toFixed(1)}%`;
    score.style.color = "var(--emerald)";
  } else {
    hero.className = "result-hero unknown";
    tag.textContent = "⚠️ UNKNOWN DISCOVERY";
    tag.style.color = "var(--rose)";
    name.textContent = "New Individual Tiger Detected";
    meta.textContent = `No matching identity in catalog. Flagged for registration.`;
    score.textContent = `${(tiger.confidence * 100).toFixed(1)}%`;
    score.style.color = "var(--amber)";
  }

  // Step 1: Input Frame
  setVisualStep(1, tiger.segmented_crop_b64, "BEST FRAME", `Laplacian Sharpness: ${tiger.quality_score.toFixed(1)}/100`);

  // Step 2: Pose
  setVisualStep(2, tiger.segmented_crop_b64, (tiger.pose || "WALKING").toUpperCase(), `15 Anatomical Keypoints · Conf: ${(tiger.pose_confidence * 100).toFixed(1)}%`);
  drawPoseSkeleton(tiger.keypoints_data);

  // Step 3: Segmentation
  setVisualStep(3, tiger.mask_b64, "DDRNET-39", "Isolated Tiger Silhouette · Background Removed");

  // Step 4: Stripes
  setVisualStep(4, tiger.stripe_ridge_b64, `${(tiger.stripe_match_score * 100).toFixed(1)}% MATCH`, `Flank Density: ${(tiger.stripe_density * 100).toFixed(1)}% · Ridge Branches Extracted`);

  // Matches Table
  const tbody = document.getElementById("matches-table-body");
  tbody.innerHTML = "";
  if (tiger.nearest_neighbors && tiger.nearest_neighbors.length > 0) {
    tiger.nearest_neighbors.forEach((m, idx) => {
      const isWinner = (tiger.tiger_id && String(m.tiger_id) === String(tiger.tiger_id));
      const tr = document.createElement("tr");
      if (isWinner) tr.className = "winner-row";
      tr.innerHTML = `
        <td><strong>#${idx + 1}</strong></td>
        <td><strong>Tiger #${m.tiger_id}</strong> ${isWinner ? '<span style="font-size:0.75rem; color:var(--emerald);">[WINNER]</span>' : ''}</td>
        <td>${m.side || 'Flank'} Profile</td>
        <td style="font-family: var(--font-mono);">${m.distance.toFixed(4)}</td>
        <td><strong style="color: var(--cyan);">+${m.weight.toFixed(2)} pts</strong></td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function renderNonTarget(data) {
  const hero = document.getElementById("result-hero");
  hero.className = "result-hero unknown";
  document.getElementById("result-status-tag").textContent = "NO TIGER DETECTED";
  document.getElementById("result-status-tag").style.color = "var(--text-dim)";
  document.getElementById("result-name").textContent = data.animal_detected ? "Non-Target Wildlife (Herbivore)" : "Empty Frame / False Trigger";
  document.getElementById("result-meta").textContent = "Pass 1 Screening discarded this event early to save GPU and storage.";
  document.getElementById("result-score-val").textContent = "0.0%";
  document.getElementById("result-score-val").style.color = "var(--text-dim)";

  for (let i = 1; i <= 4; i++) {
    clearVisualStep(i);
  }
  document.getElementById("matches-table-body").innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-dim);">No tiger matches in this event.</td></tr>`;
}

function setVisualStep(stepNum, imgSrc, badgeText, footerText) {
  const img = document.getElementById(`img-step-${stepNum}`);
  const ph = document.getElementById(`ph-step-${stepNum}`);
  const badge = document.getElementById(`badge-step-${stepNum}`);
  const lbl = document.getElementById(`lbl-step-${stepNum}`);

  if (imgSrc) {
    img.src = imgSrc;
    img.style.display = "block";
    if (ph) ph.style.display = "none";
  } else {
    img.style.display = "none";
    if (ph) ph.style.display = "block";
  }

  if (badge && badgeText) badge.textContent = badgeText;
  if (lbl && footerText) lbl.textContent = footerText;
}

function clearVisualStep(stepNum) {
  const img = document.getElementById(`img-step-${stepNum}`);
  const ph = document.getElementById(`ph-step-${stepNum}`);
  if (img) img.style.display = "none";
  if (ph) ph.style.display = "block";
}

function drawPoseSkeleton(keypointsData) {
  const canvas = document.getElementById("pose-canvas");
  if (!canvas || !keypointsData || !keypointsData.keypoints) return;

  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const kps = keypointsData.keypoints;

  // Skeleton Lines
  ctx.strokeStyle = "rgba(6, 182, 212, 0.8)";
  ctx.lineWidth = 2.5;

  for (const [u, v] of SKELETON_CONNECTIONS) {
    if (kps[u] && kps[v]) {
      ctx.beginPath();
      ctx.moveTo(kps[u].rel_x * canvas.width, kps[u].rel_y * canvas.height);
      ctx.lineTo(kps[v].rel_x * canvas.width, kps[v].rel_y * canvas.height);
      ctx.stroke();
    }
  }

  // Keypoints Nodes
  for (const [_, pt] of Object.entries(kps)) {
    const px = pt.rel_x * canvas.width;
    const py = pt.rel_y * canvas.height;

    ctx.beginPath();
    ctx.arc(px, py, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "#10b981";
    ctx.shadowColor = "#10b981";
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.beginPath();
    ctx.arc(px, py, 1.5, 0, 2 * Math.PI);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }
}

function switchTab(tabId) {
  document.querySelectorAll(".nav-tab-link").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

  event && event.target && event.target.classList.add("active");
  const target = document.getElementById(tabId);
  if (target) target.classList.add("active");

  if (tabId === "tab-map" && mapInstance) {
    setTimeout(() => { mapInstance.invalidateSize(); }, 150);
  }
}

function initMap() {
  const mapEl = document.getElementById("gis-map");
  if (!mapEl || mapInstance) return;

  mapInstance = L.map("gis-map").setView([21.665, 79.315], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "© OpenStreetMap"
  }).addTo(mapInstance);

  for (const [camId, coords] of Object.entries(CAMERA_COORDS)) {
    L.circleMarker(coords, {
      radius: 6,
      color: "#06b6d4",
      fillColor: "#06b6d4",
      fillOpacity: 0.8
    }).addTo(mapInstance).bindPopup(`<b>Station:</b> ${camId}`);
  }
}
