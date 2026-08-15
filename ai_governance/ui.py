"""PM-facing web dashboard served at /ui.

Single-page app (vanilla JS + Chart.js CDN, no build step) that polls
/dashboard every 5s. Includes:
  - Safety status banner
  - KPI cards (total, resolution, response, drift)
  - Side-by-side: normal metrics vs governance signals
  - Drift trend chart (sparkline from /trends)
  - Category breakdown table with drift bars
  - Alert timeline
"""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Governance Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js"></script>
<style>
  :root {
    --safe: #16a34a; --warn: #ca8a04; --critical: #dc2626; --rollback: #7c3aed;
    --bg: #0f172a; --surface: #1e293b; --border: #334155; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #38bdf8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: var(--bg); color: var(--text); font-size: 13px; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-bottom: 1px solid var(--border); background: var(--surface); }
  header h1 { font-size: 15px; font-weight: 600; color: var(--accent); letter-spacing: 0.05em; }
  #status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  #last-update { color: var(--muted); font-size: 11px; }

  #safety-banner {
    padding: 14px 20px; font-size: 14px; font-weight: 700; letter-spacing: 0.04em;
    text-align: center; border-bottom: 1px solid var(--border); text-transform: uppercase;
  }
  .banner-safe    { background: #052e16; color: #4ade80; border-left: 4px solid var(--safe); }
  .banner-unsafe  { background: #450a0a; color: #f87171; border-left: 4px solid var(--critical); }

  main { padding: 16px 20px; display: flex; flex-direction: column; gap: 16px; }

  .section-title { font-size: 11px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }

  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px;
  }
  .card-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  .card-value { font-size: 22px; font-weight: 700; }
  .card-sub { font-size: 10px; color: var(--muted); margin-top: 2px; }

  .col-header { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; }
  .col-header-title { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }

  .kv-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--border); }
  .kv-row:last-child { border-bottom: none; }
  .kv-key { color: var(--muted); }
  .kv-val { font-weight: 600; }

  table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }
  th { background: #0f172a; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  td { padding: 7px 10px; border-bottom: 1px solid var(--border); color: var(--text); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.03); }

  .badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
  .badge-ok       { background: #052e16; color: #4ade80; }
  .badge-warn     { background: #422006; color: #fbbf24; }
  .badge-critical { background: #450a0a; color: #f87171; }
  .badge-rollback { background: #2e1065; color: #c4b5fd; }

  .alert-row { display: flex; gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--border); align-items: flex-start; }
  .alert-row:last-child { border-bottom: none; }
  .alert-time { color: var(--muted); font-size: 11px; min-width: 60px; }
  .alert-msg { flex: 1; }
  .alert-rule { font-weight: 600; }
  .alert-text { color: var(--muted); font-size: 11px; }
  #alerts-list.empty { color: var(--muted); text-align: center; padding: 20px; }

  .drift-bar-container { display: flex; align-items: center; gap: 8px; }
  .drift-bar { flex: 1; height: 5px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .drift-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s; }

  .chart-container { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }
  .chart-container canvas { width: 100% !important; height: 200px !important; }

  .forecast-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; }
  .forecast-item { padding: 6px 0; border-bottom: 1px solid var(--border); display: flex; gap: 10px; align-items: center; }
  .forecast-item:last-child { border-bottom: none; }
  .forecast-eta { font-weight: 700; min-width: 60px; }
  .forecast-detail { flex: 1; font-size: 12px; color: var(--muted); }

  .accuracy-bar { display: flex; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 4px; }
  .accuracy-correct { background: var(--safe); }
  .accuracy-incorrect { background: var(--critical); }

  footer { padding: 10px 20px; text-align: center; color: var(--muted); font-size: 10px; border-top: 1px solid var(--border); }
  .error-banner { background: #450a0a; color: #f87171; padding: 12px 20px; text-align: center; display: none; }
</style>
</head>
<body>

<header>
  <h1>AI GOVERNANCE DASHBOARD</h1>
  <span id="agent-label" style="color:var(--muted);font-size:11px">loading...</span>
  <span id="last-update"></span>
</header>

<div id="safety-banner" class="banner-safe">loading...</div>
<div id="error-banner" class="error-banner">Cannot reach governance API - retrying...</div>

<main>
  <!-- top KPI row -->
  <div>
    <div class="section-title">Key Metrics</div>
    <div class="grid-4">
      <div class="card">
        <div class="card-label">Total Decisions</div>
        <div class="card-value" id="kpi-total">-</div>
        <div class="card-sub" id="kpi-24h">- last 24 h</div>
      </div>
      <div class="card">
        <div class="card-label">Resolution Rate</div>
        <div class="card-value" id="kpi-resolution">-</div>
        <div class="card-sub">recent window</div>
      </div>
      <div class="card">
        <div class="card-label">Avg Response</div>
        <div class="card-value" id="kpi-response">-</div>
        <div class="card-sub">ms</div>
      </div>
      <div class="card">
        <div class="card-label">Overall Drift</div>
        <div class="card-value" id="kpi-drift">-</div>
        <div class="card-sub" id="kpi-violations">- violations</div>
      </div>
    </div>
  </div>

  <!-- side-by-side normal vs governance -->
  <div>
    <div class="section-title">Side-by-Side: Normal Metrics vs Governance Signals</div>
    <div class="grid-2">
      <div class="col-header">
        <div class="col-header-title">Normal Performance</div>
        <div class="kv-row"><span class="kv-key">Resolution rate</span><span class="kv-val" id="nm-resolution">-</span></div>
        <div class="kv-row"><span class="kv-key">Escalation rate</span><span class="kv-val" id="nm-escalation">-</span></div>
        <div class="kv-row"><span class="kv-key">Denial rate</span><span class="kv-val" id="nm-denial">-</span></div>
        <div class="kv-row"><span class="kv-key">Avg response time</span><span class="kv-val" id="nm-response">-</span></div>
        <div class="kv-row"><span class="kv-key">Decisions last hour</span><span class="kv-val" id="nm-hour">-</span></div>
        <div class="kv-row"><span class="kv-key">Decisions last 24 h</span><span class="kv-val" id="nm-day">-</span></div>
      </div>
      <div class="col-header">
        <div class="col-header-title">Governance Signals</div>
        <div class="kv-row"><span class="kv-key">Overall drift score</span><span class="kv-val" id="gv-drift">-</span></div>
        <div class="kv-row"><span class="kv-key">Active violations</span><span class="kv-val" id="gv-violations">-</span></div>
        <div class="kv-row"><span class="kv-key">Rollback events</span><span class="kv-val" id="gv-rollbacks">-</span></div>
        <div class="kv-row"><span class="kv-key">High-risk decisions</span><span class="kv-val" id="gv-highrisk">-</span></div>
        <div class="kv-row"><span class="kv-key">High-risk escalation %</span><span class="kv-val" id="gv-hrescalation">-</span></div>
        <div class="kv-row"><span class="kv-key">High-risk accuracy</span><span class="kv-val" id="gv-accuracy">-</span></div>
      </div>
    </div>
  </div>

  <!-- drift trend chart + forecast -->
  <div>
    <div class="section-title">Drift Trend &amp; Forecast</div>
    <div class="grid-2">
      <div class="chart-container">
        <canvas id="drift-chart"></canvas>
      </div>
      <div class="forecast-card">
        <div class="col-header-title">Threshold Breach Forecast</div>
        <div id="forecast-list" style="color:var(--muted);text-align:center;padding:20px">No forecasts available</div>
      </div>
    </div>
  </div>

  <!-- category breakdown -->
  <div>
    <div class="section-title">Category Breakdown</div>
    <table>
      <thead>
        <tr>
          <th>Category</th><th>Recent</th><th>Resolution%</th><th>Escalation%</th>
          <th>High-Risk</th><th>Drift</th><th>Status</th>
        </tr>
      </thead>
      <tbody id="cat-tbody"><tr><td colspan="7" style="text-align:center;color:var(--muted)">no data</td></tr></tbody>
    </table>
  </div>

  <!-- recent alerts -->
  <div>
    <div class="section-title">Recent Alerts</div>
    <div class="card" style="padding:10px 16px">
      <div id="alerts-list" class="empty">No alerts</div>
    </div>
  </div>
</main>

<footer>Auto-refreshes every 5 s | <a href="/docs" style="color:var(--accent)">API Docs</a> | <a href="/metrics" style="color:var(--accent)">Metrics</a> | <a href="/audit" style="color:var(--accent)">Audit Log</a> | <a href="/admin/report/text" style="color:var(--accent)">Compliance Report</a></footer>

<script>
const pct = v => (v * 100).toFixed(1) + '%';
const fmt = v => v == null ? 'N/A' : (typeof v === 'number' ? (v < 1 ? pct(v) : v.toFixed(1)) : v);

function severityClass(s) {
  const map = { ok: 'badge-ok', warn: 'badge-warn', critical: 'badge-critical', rollback: 'badge-rollback' };
  return 'badge ' + (map[s] || 'badge-ok');
}

function driftColor(score) {
  if (score < 0.2) return '#4ade80';
  if (score < 0.4) return '#fbbf24';
  if (score < 0.7) return '#f97316';
  return '#f87171';
}

function set(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

function renderDrift(score) {
  const pctVal = Math.round(score * 100);
  return `<div class="drift-bar-container">
    <div class="drift-bar"><div class="drift-bar-fill" style="width:${pctVal}%;background:${driftColor(score)}"></div></div>
    <span style="font-size:11px;color:${driftColor(score)};min-width:34px">${pctVal}%</span>
  </div>`;
}

// Chart.js drift history
let driftChart = null;
const driftHistory = [];
const MAX_HISTORY = 60;

function updateDriftChart(driftScore) {
  const now = new Date().toLocaleTimeString();
  driftHistory.push({ t: now, v: driftScore });
  if (driftHistory.length > MAX_HISTORY) driftHistory.shift();

  const ctx = document.getElementById('drift-chart');
  if (!ctx) return;

  if (driftChart) {
    driftChart.data.labels = driftHistory.map(p => p.t);
    driftChart.data.datasets[0].data = driftHistory.map(p => p.v * 100);
    driftChart.update('none');
    return;
  }

  driftChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: driftHistory.map(p => p.t),
      datasets: [{
        label: 'Drift Score %',
        data: driftHistory.map(p => p.v * 100),
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        annotation: {
          annotations: {
            warnLine: { type: 'line', yMin: 30, yMax: 30, borderColor: '#fbbf24', borderWidth: 1, borderDash: [4,4] },
            critLine: { type: 'line', yMin: 60, yMax: 60, borderColor: '#f87171', borderWidth: 1, borderDash: [4,4] },
          }
        }
      },
      scales: {
        x: { display: true, ticks: { color: '#64748b', maxTicksLimit: 8, font: { size: 9 } }, grid: { color: '#1e293b' } },
        y: { min: 0, max: 100, ticks: { color: '#64748b', font: { size: 9 }, callback: v => v + '%' }, grid: { color: '#1e293b' } },
      }
    }
  });
}

async function loadForecast() {
  try {
    const r = await fetch('/forecast');
    if (!r.ok) return;
    const data = await r.json();
    const el = document.getElementById('forecast-list');
    if (!data.forecasts || data.forecasts.length === 0) {
      el.innerHTML = '<div style="color:var(--muted);text-align:center;padding:20px">No threshold breaches forecasted within ' + data.horizon_hours + 'h</div>';
      return;
    }
    el.innerHTML = data.forecasts.slice(0, 5).map(f => {
      const color = f.severity === 'imminent' ? '#f87171' : f.severity === 'near' ? '#fbbf24' : '#38bdf8';
      const eta = f.hours_to_breach != null ? f.hours_to_breach.toFixed(1) + 'h' : '?';
      return `<div class="forecast-item">
        <span class="forecast-eta" style="color:${color}">${eta}</span>
        <span class="forecast-detail">${f.category}/${f.metric} - ${(f.current_delta / f.threshold_delta * 100).toFixed(0)}% of threshold</span>
        <span class="${severityClass(f.severity === 'imminent' ? 'critical' : f.severity === 'near' ? 'warn' : 'ok')}">${f.severity}</span>
      </div>`;
    }).join('');
  } catch(e) {}
}

async function refresh() {
  try {
    const r = await fetch('/dashboard');
    if (!r.ok) throw new Error(r.statusText);
    const d = await r.json();
    document.getElementById('error-banner').style.display = 'none';

    set('agent-label', d.agent_id);
    set('last-update', 'Updated ' + new Date().toLocaleTimeString());

    const banner = document.getElementById('safety-banner');
    banner.textContent = d.is_safe ? 'Agent is safe to continue running' : 'Agent is NOT safe - ' + (d.safety_reason || 'threshold breached');
    banner.className = d.is_safe ? 'banner-safe' : 'banner-unsafe';

    const nm = d.normal_metrics, gv = d.governance_signals;

    set('kpi-total', nm.total_decisions.toLocaleString());
    set('kpi-24h', (nm.decisions_last_24h || 0) + ' last 24 h');
    set('kpi-resolution', pct(nm.resolution_rate));
    set('kpi-response', nm.avg_response_time_ms.toFixed(0) + ' ms');
    set('kpi-drift', (gv.overall_drift_score * 100).toFixed(1) + '%');
    set('kpi-violations', gv.active_violations + ' violation' + (gv.active_violations !== 1 ? 's' : ''));

    set('nm-resolution', pct(nm.resolution_rate));
    set('nm-escalation', pct(nm.escalation_rate));
    set('nm-denial', pct(nm.denial_rate));
    set('nm-response', nm.avg_response_time_ms.toFixed(1) + ' ms');
    set('nm-hour', nm.decisions_last_hour);
    set('nm-day', nm.decisions_last_24h);

    set('gv-drift', (gv.overall_drift_score * 100).toFixed(2) + '%');
    set('gv-violations', gv.active_violations);
    set('gv-rollbacks', gv.rollback_events);
    set('gv-highrisk', gv.high_risk_decisions_recent);
    set('gv-hrescalation', pct(gv.high_risk_escalation_rate));
    set('gv-accuracy', gv.high_risk_accuracy != null ? pct(gv.high_risk_accuracy) : 'N/A (no labels)');

    updateDriftChart(gv.overall_drift_score);

    const tbody = document.getElementById('cat-tbody');
    if (!d.category_breakdown || d.category_breakdown.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted)">no categories detected yet</td></tr>';
    } else {
      tbody.innerHTML = d.category_breakdown.map(row => `
        <tr>
          <td style="font-weight:600">${row.category}</td>
          <td>${row.total_recent}</td>
          <td>${pct(row.resolution_rate)}</td>
          <td>${pct(row.escalation_rate)}</td>
          <td>${row.high_risk_count}</td>
          <td>${renderDrift(row.drift_score)}</td>
          <td><span class="${severityClass(row.severity)}">${row.severity}</span></td>
        </tr>`).join('');
    }

    const alertsList = document.getElementById('alerts-list');
    if (!d.recent_alerts || d.recent_alerts.length === 0) {
      alertsList.className = 'empty';
      alertsList.textContent = 'No alerts';
    } else {
      alertsList.className = '';
      alertsList.innerHTML = d.recent_alerts.map(a => `
        <div class="alert-row">
          <span class="alert-time">${a.time}</span>
          <div class="alert-msg">
            <div class="alert-rule">${a.rule} <span class="${severityClass(a.severity)}">${a.severity}</span></div>
            <div class="alert-text">${a.message}</div>
          </div>
        </div>`).join('');
    }
  } catch(e) {
    document.getElementById('error-banner').style.display = 'block';
  }
}

refresh();
loadForecast();
setInterval(refresh, 5000);
setInterval(loadForecast, 30000);
</script>
</body>
</html>"""
