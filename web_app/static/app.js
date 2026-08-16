/* ==========================================================================
   Tiger Re-ID Console — Simplified Visual Pipeline Controller
   ========================================================================== */

let currentQuery = {
  image_path: null,
  image_base64: null,
  result: null
};

let leafletMap = null;
let mapLayers = {
  cameras: null,
  sightings: null,
  trajectories: null,
  mcp_polygon: null
};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSampleCarousel();
  initFileUpload();
  initRunButton();
  initGISMap();
  initEnrollmentModal();
  loadSightings();
  loadAblations();
});

/* Tabs */
function initTabs() {
  const btns = document.querySelectorAll(".tab-btn");
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      document.getElementById(tabId).classList.add("active");

      if (tabId === "tab-gis" && leafletMap) {
        setTimeout(() => leafletMap.invalidateSize(), 200);
      }
    });
  });
}

/* Sample Carousel */
async function initSampleCarousel() {
  const container = document.getElementById("sample-carousel");
  try {
    const res = await fetch("/api/sample_images");
    const data = await res.json();
    container.innerHTML = "";

    data.samples.forEach((s, idx) => {
      const thumb = document.createElement("div");
      thumb.className = `sample-thumb ${idx === 0 ? "active" : ""}`;
      thumb.title = `Click to load: ${s.filename}`;
      thumb.innerHTML = `<img src="/api/image_raw?path=${encodeURIComponent(s.path)}" alt="${s.filename}" />`;

      thumb.addEventListener("click", () => {
        document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
        thumb.classList.add("active");
        selectImage(s.path, null);
      });

      container.appendChild(thumb);
    });

    if (data.samples.length > 0) {
      selectImage(data.samples[0].path, null);
    }
  } catch (e) {
    container.innerHTML = `<div class="text-sm text-muted">Failed to load samples</div>`;
  }
}

/* File Upload */
function initFileUpload() {
  const fileInput = document.getElementById("file-input");
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
        selectImage(null, evt.target.result);
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });
}

function selectImage(path, b64) {
  currentQuery.image_path = path;
  currentQuery.image_base64 = b64;
  currentQuery.result = null;

  // Update Stage 1 Preview
  const img1 = document.getElementById("img-stage-1");
  const p1 = document.getElementById("placeholder-stage-1");
  img1.src = b64 || `/api/image_raw?path=${encodeURIComponent(path)}`;
  img1.style.display = "block";
  p1.style.display = "none";

  // Reset Stages 2-5 to pending
  resetDownstreamStages();
}

function resetDownstreamStages() {
  ["img-stage-2", "img-stage-3"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });
  ["placeholder-stage-2", "placeholder-stage-3"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "block";
  });

  document.getElementById("info-qa").textContent = "Awaiting run";
  document.getElementById("info-bbox").textContent = "Pending";
  document.getElementById("branch-a-pred").textContent = "Top Guess: --";
  document.getElementById("branch-a-fill").style.width = "0%";
  document.getElementById("branch-a-conf").textContent = "Confidence: 0.0%";
  document.getElementById("sparkline-preview").innerHTML = "";
  document.getElementById("branch-b-dist").textContent = "Nearest Database Match: --";
  document.getElementById("final-tiger-id").textContent = "--";
  document.getElementById("final-status-badge").textContent = "Awaiting Query";
  document.getElementById("top-matches-list").innerHTML = `<div class="knn-item-placeholder">Click 'Run AI Pipeline' above</div>`;

  const banner = document.getElementById("decision-callout");
  banner.className = "decision-callout banner-neutral";
  document.getElementById("decision-badge").textContent = "READY TO ANALYZE";
  document.getElementById("decision-title").textContent = "Click 'Run AI Pipeline' above";
  document.getElementById("decision-explanation").textContent = "Watch how the raw jungle photo gets segmented, background-removed, and fingerprint-matched.";
  document.getElementById("enrollment-action-box").classList.add("hidden");
}

/* Run Pipeline Button */
function initRunButton() {
  const btn = document.getElementById("run-pipeline-btn");
  btn.addEventListener("click", async () => {
    if (!currentQuery.image_path && !currentQuery.image_base64) {
      alert("Please select a tiger image first!");
      return;
    }

    btn.disabled = true;
    btn.innerHTML = `<span>⏳ Processing Stage 1 → 5...</span>`;

    try {
      const payload = {
        image_path: currentQuery.image_path,
        image_base64: currentQuery.image_base64,
        camera_id: "PTR-CORE-EP-01",
        timestamp: new Date().toISOString().slice(0, 19)
      };

      const res = await fetch("/api/inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      currentQuery.result = data;

      renderPipelineStages(data);
      loadSightings();
    } catch (e) {
      alert("Inference execution failed: " + e.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span>⚡ Run AI Pipeline</span>`;
    }
  });
}

/* Render 5 Stages */
function renderPipelineStages(data) {
  // 1. Stage 2: Segmentation Mask
  const img2 = document.getElementById("img-stage-2");
  const p2 = document.getElementById("placeholder-stage-2");
  img2.src = data.mask_b64;
  img2.style.display = "block";
  p2.style.display = "none";
  document.getElementById("info-qa").textContent = data.qa_status;
  document.getElementById("info-qa").style.color = (data.qa_status === "PASSED") ? "var(--accent-emerald)" : "var(--accent-amber)";

  // 2. Stage 3: Isolated Tiger Crop
  const img3 = document.getElementById("img-stage-3");
  const p3 = document.getElementById("placeholder-stage-3");
  img3.src = data.segmented_crop_b64;
  img3.style.display = "block";
  p3.style.display = "none";
  if (data.bbox) {
    document.getElementById("info-bbox").textContent = `[${data.bbox.join(", ")}]`;
  }

  // 3. Stage 4: Dual-Branch Features
  document.getElementById("branch-a-pred").textContent = `Top Guess: ${data.classifier_prediction}`;
  const confPct = (data.classifier_confidence * 100).toFixed(1);
  document.getElementById("branch-a-fill").style.width = `${Math.min(100, confPct)}%`;
  document.getElementById("branch-a-conf").textContent = `Confidence: ${confPct}% (Weight = 1.0 if ≥ 95%)`;

  const sparkline = document.getElementById("sparkline-preview");
  sparkline.innerHTML = "";
  if (data.embedding_preview) {
    data.embedding_preview.forEach(val => {
      const h = Math.max(2, Math.min(18, Math.abs(val) * 45));
      const bar = document.createElement("div");
      bar.className = "spark-bar";
      bar.style.height = `${h}px`;
      bar.title = `Dim value: ${val.toFixed(4)}`;
      sparkline.appendChild(bar);
    });
  }

  const topMatch = (data.nearest_neighbors && data.nearest_neighbors.length > 0) ? data.nearest_neighbors[0] : null;
  if (topMatch) {
    document.getElementById("branch-b-dist").textContent = `Nearest Match: ${topMatch.tiger_id} (${topMatch.side}) · d = ${topMatch.distance.toFixed(4)}`;
  }

  // 4. Stage 5: Weighted Fusion & Matches
  document.getElementById("final-tiger-id").textContent = data.recognized ? `Tiger #${data.tiger_id}` : "UNKNOWN TIGER";
  document.getElementById("final-status-badge").textContent = data.recognized ? "✓ VERIFIED KNOWN TIGER" : "⚠️ UNKNOWN DISCOVERY";
  document.getElementById("final-status-badge").style.color = data.recognized ? "var(--accent-emerald)" : "var(--accent-rose)";

  const matchesList = document.getElementById("top-matches-list");
  matchesList.innerHTML = "";
  if (data.nearest_neighbors) {
    data.nearest_neighbors.slice(0, 3).forEach((m, idx) => {
      const item = document.createElement("div");
      item.className = "knn-item";
      item.innerHTML = `
        <strong>#${idx+1} Tiger ${m.tiger_id} (${m.side})</strong>
        <span style="color: var(--accent-cyan);">d = ${m.distance.toFixed(4)}</span>
      `;
      matchesList.appendChild(item);
    });
  }

  // 5. Decision Callout Banner
  const banner = document.getElementById("decision-callout");
  const badge = document.getElementById("decision-badge");
  const title = document.getElementById("decision-title");
  const desc = document.getElementById("decision-explanation");
  const enrollBox = document.getElementById("enrollment-action-box");

  if (data.recognized && data.status === "KNOWN") {
    banner.className = "decision-callout banner-known";
    badge.textContent = "KNOWN TIGER CONFIRMED";
    title.textContent = `Identified as Tiger #${data.tiger_id}`;
    desc.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}% · Sighting logged to Pench Database.`;
    enrollBox.classList.add("hidden");
  } else {
    banner.className = "decision-callout banner-unknown";
    badge.textContent = "UNKNOWN TIGER DETECTED";
    title.textContent = "New Individual Detected in Camera Trap";
    desc.textContent = `Both branches rejected match (Classification prob ${confPct}% < 95% or Distance ${data.nearest_distance} > 0.40).`;
    enrollBox.classList.remove("hidden");
  }
}

/* Enrollment Modal */
function initEnrollmentModal() {
  const modal = document.getElementById("enroll-modal");
  document.getElementById("open-enroll-modal-btn").addEventListener("click", () => modal.classList.remove("hidden"));
  document.getElementById("close-modal-btn").addEventListener("click", () => modal.classList.add("hidden"));
  document.getElementById("cancel-enroll-btn").addEventListener("click", () => modal.classList.add("hidden"));

  document.getElementById("confirm-enroll-btn").addEventListener("click", async () => {
    const tid = document.getElementById("enroll-tiger-id").value.trim();
    if (!tid) {
      alert("Please enter a Tiger ID");
      return;
    }

    try {
      const res = await fetch("/api/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tiger_id: tid,
          side: document.getElementById("enroll-side").value,
          camera_id: "PTR-CORE-EP-01",
          timestamp: new Date().toISOString().slice(0, 19),
          verifier_notes: document.getElementById("enroll-notes").value
        })
      });
      const data = await res.json();
      alert(`Tiger ${tid} successfully enrolled into Gallery!`);
      modal.classList.add("hidden");
    } catch (e) {
      alert("Enrollment failed: " + e.message);
    }
  });
}

/* GIS Map */
function initGISMap() {
  leafletMap = L.map("gis-map").setView([21.660, 79.315], 12);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> | Pench Tiger Reserve',
    maxZoom: 18
  }).addTo(leafletMap);

  mapLayers.cameras = L.layerGroup().addTo(leafletMap);
  mapLayers.sightings = L.layerGroup().addTo(leafletMap);
  mapLayers.trajectories = L.layerGroup().addTo(leafletMap);
  mapLayers.mcp_polygon = L.layerGroup().addTo(leafletMap);

  document.getElementById("mcp-tiger-selector").addEventListener("change", refreshGISData);
  refreshGISData();
}

async function refreshGISData() {
  if (!leafletMap) return;

  mapLayers.cameras.clearLayers();
  mapLayers.sightings.clearLayers();
  mapLayers.trajectories.clearLayers();
  mapLayers.mcp_polygon.clearLayers();

  try {
    const statusRes = await fetch("/api/status");
    const statusData = await statusRes.json();
    if (statusData.camera_stations) {
      statusData.camera_stations.forEach(cam => {
        const marker = L.circleMarker([cam.lat, cam.lon], {
          radius: 7,
          fillColor: "#38bdf8",
          color: "#fff",
          weight: 1.5,
          fillOpacity: 0.9
        }).bindPopup(`<strong>${cam.camera_id}</strong><br>${cam.name}`);
        mapLayers.cameras.addLayer(marker);
      });
    }
  } catch (e) {}

  try {
    const sightRes = await fetch("/api/sightings");
    const sightData = await sightRes.json();
    const selectedTiger = document.getElementById("mcp-tiger-selector").value;

    const filtered = (selectedTiger === "ALL") 
      ? sightData.sightings 
      : sightData.sightings.filter(s => s.tiger_id === selectedTiger);

    const pts = [];
    filtered.forEach(s => {
      const marker = L.circleMarker([s.latitude, s.longitude], {
        radius: 9,
        fillColor: "#f59e0b",
        color: "#fff",
        weight: 2,
        fillOpacity: 0.95
      }).bindPopup(`<strong>Tiger: ${s.tiger_id}</strong><br>Station: ${s.camera_id}<br>Time: ${s.timestamp}`);
      mapLayers.sightings.addLayer(marker);
      pts.push([s.latitude, s.longitude]);
    });

    if (pts.length > 1) {
      const poly = L.polyline(pts, {
        color: "#f59e0b",
        weight: 2.5,
        dashArray: "5, 8",
        opacity: 0.75
      });
      mapLayers.trajectories.addLayer(poly);
    }

    const targetId = (selectedTiger === "ALL") ? "TIG_007" : selectedTiger;
    const mcpRes = await fetch(`/api/mcp?tiger_id=${targetId}`);
    const mcpData = await mcpRes.json();

    if (mcpData.polygon_points && mcpData.polygon_points.length >= 3) {
      const polyLayer = L.polygon(mcpData.polygon_points, {
        color: "#10b981",
        fillColor: "#10b981",
        fillOpacity: 0.25,
        weight: 2
      }).bindPopup(`<strong>100% MCP Home Range: ${mcpData.tiger_id}</strong><br>Area: ${mcpData.mcp_area_km2} km²`);
      mapLayers.mcp_polygon.addLayer(polyLayer);

      document.getElementById("mcp-area-display").textContent = `${mcpData.mcp_area_km2} km²`;
      const vertList = document.getElementById("mcp-vertices-list");
      vertList.innerHTML = mcpData.polygon_points.map((pt, i) => `<li>V${i+1}: [${pt[0].toFixed(4)}, ${pt[1].toFixed(4)}]</li>`).join("");
    }
  } catch (e) {}
}

/* Sightings */
async function loadSightings() {
  try {
    const res = await fetch("/api/sightings");
    const data = await res.json();
    const tbody = document.querySelector("#table-sightings tbody");
    tbody.innerHTML = data.sightings.map(s => `
      <tr>
        <td><code>${s.event_id}</code></td>
        <td><strong style="color: var(--accent-amber);">${s.tiger_id}</strong></td>
        <td>${s.camera_id}</td>
        <td>${s.latitude.toFixed(4)}</td>
        <td>${s.longitude.toFixed(4)}</td>
        <td>${s.timestamp}</td>
        <td><span class="badge badge-primary">${s.side || 'Unknown'}</span></td>
        <td>${(s.confidence * 100).toFixed(1)}%</td>
      </tr>
    `).join("");
  } catch (e) {}
}

/* Ablations */
async function loadAblations() {
  try {
    const res = await fetch("/api/ablations");
    const data = await res.json();

    const tbody = document.querySelector("#table-bg-ablation tbody");
    tbody.innerHTML = data.background_ablation.map(r => `
      <tr>
        <td><strong>${r["Variant / Setting"]}</strong></td>
        <td>${(r["Paper Reported Accuracy"] * 100).toFixed(2)}%</td>
        <td>${typeof r["Dataset Accuracy"] === 'number' ? (r["Dataset Accuracy"] * 100).toFixed(2) + '%' : r["Dataset Accuracy"]}</td>
        <td>${(r["Paper Reported Precision"] * 100).toFixed(2)}%</td>
        <td>${typeof r["Dataset Precision"] === 'number' ? (r["Dataset Precision"] * 100).toFixed(2) + '%' : r["Dataset Precision"]}</td>
        <td>${(r["Paper Reported Micro-F1"] * 100).toFixed(2)}%</td>
        <td>${typeof r["Dataset Micro-F1"] === 'number' ? (r["Dataset Micro-F1"] * 100).toFixed(2) + '%' : r["Dataset Micro-F1"]}</td>
      </tr>
    `).join("");
  } catch (e) {}
}
