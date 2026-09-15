/* ==========================================================================
   Indian Railways Maintenance AI Dashboard — Application Logic
   ========================================================================== */

const API_BASE = ""; // relative path for seamless backend serving

let chartSystemDist = null;
let chartZonePie    = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  fetchHealthStatus();
  loadOperationsHub();
  loadScheduleData();
  loadAnalyticsData();
  initFormHandler();
  initCopilotHandler();

  document.getElementById("btn-refresh-data").addEventListener("click", () => {
    loadOperationsHub();
    loadScheduleData();
    loadAnalyticsData();
  });

  document.getElementById("btn-trigger-pipeline").addEventListener("click", triggerPipelineRegen);
});

/* ── Navigation Tabs ────────────────────────────────────────────────────── */
function initTabs() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabContents = document.querySelectorAll(".tab-content");

  const titles = {
    "dashboard-tab":       ["Operations Hub & Asset Intelligence", "Real-time predictive risk scoring across TMS, SMMS, TDMS & COA"],
    "field-predictor-tab": ["Field Inspection Risk Assessor", "Evaluate failure probability & maintenance duration for physical assets"],
    "schedule-tab":        ["CP-SAT Optimized Maintenance Schedule", "Constraint programming allocations matched against approved COA windows"],
    "analytics-tab":       ["System Analytics & Risk Distribution", "Statistical breakdown across subsystems and railway zones"],
    "copilot-tab":         ["Gemini AI Maintenance Copilot", "Ask natural language questions about risk priorities and schedule windows"]
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");

      navItems.forEach(i => i.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      item.classList.add("active");
      document.getElementById(tabId).classList.add("active");

      if (titles[tabId]) {
        document.getElementById("page-title").textContent = titles[tabId][0];
        document.getElementById("page-desc").textContent  = titles[tabId][1];
      }
    });
  });
}

/* ── Health Status ──────────────────────────────────────────────────────── */
async function fetchHealthStatus() {
  const statusText = document.getElementById("sys-status-text");
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === "healthy" && data.models_ready) {
      statusText.textContent = "All Systems Operational";
      statusText.style.color = "var(--risk-low)";
    } else {
      statusText.textContent = "Pipeline Models Pending";
      statusText.style.color = "var(--risk-medium)";
    }
  } catch (err) {
    statusText.textContent = "API Disconnected";
    statusText.style.color = "var(--risk-critical)";
  }
}

/* ── Operations Hub ─────────────────────────────────────────────────────── */
async function loadOperationsHub() {
  const tbody = document.getElementById("top-risk-table-body");
  tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 2rem;">Loading high-risk assets...</td></tr>`;

  try {
    const res  = await fetch(`${API_BASE}/assets/risk?min_risk=0.5&top_n=10`);
    const data = await res.json();

    document.getElementById("kpi-critical-count").textContent = data.total.toLocaleString();
    
    if (!data.assets || data.assets.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 2rem;">No high risk assets found.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.assets.map(asset => {
      const probPct   = (asset.risk_probability * 100).toFixed(1);
      const riskBand  = probPct >= 85 ? "critical" : probPct >= 65 ? "high" : "medium";
      const sysBadge  = asset.source_system.toLowerCase();

      return `
        <tr>
          <td class="mono" style="font-weight: 600;">${asset.asset_id}</td>
          <td><span class="badge badge-${sysBadge}">${asset.source_system}</span></td>
          <td style="font-weight: 500;">${asset.zone}</td>
          <td class="mono">${asset.track_section || 'N/A'}</td>
          <td>${asset.component_wear_pct ? asset.component_wear_pct.toFixed(1) + '%' : 'N/A'}</td>
          <td>${asset.fault_count || 0}</td>
          <td style="font-weight: 700; color: ${probPct >= 85 ? 'var(--risk-critical)' : 'var(--risk-high)'}">${probPct}%</td>
          <td>${asset.est_maintenance_hrs ? asset.est_maintenance_hrs.toFixed(1) + ' hrs' : 'N/A'}</td>
          <td><span class="badge badge-${riskBand}">${riskBand.toUpperCase()}</span></td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: var(--risk-critical); padding: 2rem;">Failed to load assets: ${err.message}</td></tr>`;
  }
}

/* ── Maintenance Schedule ───────────────────────────────────────────────── */
async function loadScheduleData() {
  const tbody = document.getElementById("schedule-table-body");
  tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 2rem;">Loading optimized schedule...</td></tr>`;

  try {
    const res  = await fetch(`${API_BASE}/schedule/optimized`);
    if (!res.ok) throw new Error("No schedule generated yet.");
    const data = await res.json();

    document.getElementById("kpi-scheduled-count").textContent = data.total_tasks.toLocaleString();

    window._fullSchedule = data.schedule; // Cache for local filtering
    renderScheduleRows(data.schedule);

    // Setup Filter Events
    document.getElementById("sched-filter-system").onchange = filterSchedule;
    document.getElementById("sched-filter-zone").onchange   = filterSchedule;

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: var(--text-muted); padding: 2rem;">${err.message}</td></tr>`;
  }
}

function filterSchedule() {
  const sys  = document.getElementById("sched-filter-system").value;
  const zone = document.getElementById("sched-filter-zone").value;

  let filtered = window._fullSchedule || [];
  if (sys)  filtered = filtered.filter(s => s.source_system === sys);
  if (zone) filtered = filtered.filter(s => s.zone === zone);

  renderScheduleRows(filtered);
}

function renderScheduleRows(schedule) {
  const tbody = document.getElementById("schedule-table-body");
  if (!schedule || schedule.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 2rem;">No matching tasks found.</td></tr>`;
    return;
  }

  tbody.innerHTML = schedule.map(task => {
    const probPct  = (task.risk_probability * 100).toFixed(1);
    const sysBadge = (task.source_system || "TMS").toLowerCase();

    return `
      <tr>
        <td class="mono" style="font-weight: 600; color: var(--primary);">${task.asset_id}</td>
        <td><span class="badge badge-${sysBadge}">${task.source_system}</span></td>
        <td class="mono">${task.track_section}</td>
        <td style="font-weight: 700; color: var(--risk-critical)">${probPct}%</td>
        <td>${task.est_duration_hrs ? task.est_duration_hrs.toFixed(1) + ' hrs' : 'N/A'}</td>
        <td class="mono">${task.block_date}</td>
        <td class="mono">${task.block_start_time} - ${task.block_end_time}</td>
        <td><span class="badge badge-low">${task.block_type || 'engineering'}</span></td>
      </tr>
    `;
  }).join("");
}

/* ── Analytics & Charts ─────────────────────────────────────────────────── */
async function loadAnalyticsData() {
  try {
    const res  = await fetch(`${API_BASE}/eda/summary`);
    const data = await res.json();

    renderCharts(data);
  } catch (err) {
    console.error("Failed to load analytics:", err);
  }
}

function renderCharts(edaData) {
  const ctxDist = document.getElementById("system-dist-chart").getContext("2d");
  const ctxZone = document.getElementById("zone-pie-chart").getContext("2d");

  if (chartSystemDist) chartSystemDist.destroy();
  if (chartZonePie)    chartZonePie.destroy();

  // Subsystem distribution chart
  chartSystemDist = new Chart(ctxDist, {
    type: 'bar',
    data: {
      labels: ['TMS (Track)', 'SMMS (Signal)', 'TDMS (Traction)'],
      datasets: [
        {
          label: 'Total Records',
          data: [edaData.TMS?.total_records || 5000, edaData.SMMS?.total_records || 3000, edaData.TDMS?.total_records || 3000],
          backgroundColor: 'rgba(59, 130, 246, 0.4)',
          borderColor: '#3b82f6',
          borderWidth: 1
        },
        {
          label: 'High-Risk Records',
          data: [edaData.TMS?.high_risk_count || 1534, edaData.SMMS?.high_risk_count || 429, edaData.TDMS?.high_risk_count || 740],
          backgroundColor: 'rgba(239, 68, 68, 0.6)',
          borderColor: '#ef4444',
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });

  // Zone pie chart
  chartZonePie = new Chart(ctxZone, {
    type: 'doughnut',
    data: {
      labels: ['NR (Northern)', 'SR (Southern)', 'WR (Western)', 'ER (Eastern)', 'CR (Central)', 'SCR (South Central)'],
      datasets: [{
        data: [25, 20, 18, 15, 12, 10],
        backgroundColor: ['#00f2fe', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', font: { size: 11 } } } }
    }
  });
}

/* ── Field Predictor Form ───────────────────────────────────────────────── */
function initFormHandler() {
  const form = document.getElementById("risk-assessor-form");
  const box  = document.getElementById("prediction-result-box");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      source_system:        document.getElementById("form-system").value,
      zone:                 document.getElementById("form-zone").value,
      asset_age_years:      parseFloat(document.getElementById("form-age").value),
      fault_count:          parseInt(document.getElementById("form-faults").value),
      last_inspection_days: parseInt(document.getElementById("form-days").value),
      component_wear_pct:   parseFloat(document.getElementById("form-wear").value),
      traffic_load_mgt:     parseFloat(document.getElementById("form-traffic").value),
    };

    try {
      const res  = await fetch(`${API_BASE}/predict/risk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      const pct = (data.risk_probability * 100).toFixed(1);
      const bandEl = document.getElementById("res-risk-band");
      bandEl.textContent = data.risk_band;
      bandEl.className   = `badge badge-${data.risk_band.toLowerCase()}`;

      document.getElementById("res-risk-pct").textContent = `${pct}%`;
      document.getElementById("res-duration").textContent = `${data.estimated_maintenance_hrs} Hours`;
      document.getElementById("res-recommendation").textContent = data.recommendation;

      box.classList.add("active");
    } catch (err) {
      alert("Failed to score asset: " + err.message);
    }
  });
}

/* ── Gemini AI Copilot ──────────────────────────────────────────────────── */
function initCopilotHandler() {
  const form  = document.getElementById("copilot-form");
  const input = document.getElementById("copilot-input");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;
    askCopilot(query);
    input.value = "";
  });
}

async function askCopilot(query) {
  const chatList = document.getElementById("chat-messages-list");

  // Append User message
  const userDiv = document.createElement("div");
  userDiv.className = "chat-bubble user";
  userDiv.textContent = query;
  chatList.appendChild(userDiv);

  // Append Bot thinking placeholder
  const botDiv = document.createElement("div");
  botDiv.className = "chat-bubble bot";
  botDiv.textContent = "Analyzing schedule context with Gemini AI...";
  chatList.appendChild(botDiv);
  chatList.scrollTop = chatList.scrollHeight;

  try {
    const res = await fetch(`${API_BASE}/copilot/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, include_schedule_context: true })
    });
    
    const data = await res.json();
    if (data.answer) {
      botDiv.innerHTML = typeof marked !== 'undefined' ? marked.parse(data.answer) : data.answer;
    } else {
      botDiv.textContent = data.detail || "Unable to get response from Gemini.";
    }
  } catch (err) {
    botDiv.textContent = "Copilot Error: " + err.message;
  }
  chatList.scrollTop = chatList.scrollHeight;
}

/* ── Background Pipeline Trigger ────────────────────────────────────────── */
async function triggerPipelineRegen() {
  if (!confirm("Are you sure you want to trigger background data re-synthesis and CP-SAT re-optimization?")) return;

  try {
    const res  = await fetch(`${API_BASE}/schedule/regenerate`, { method: "POST" });
    const data = await res.json();
    alert("Pipeline Started! " + data.message);
  } catch (err) {
    alert("Failed to start pipeline: " + err.message);
  }
}
