/* ════════════════════════════════════════════════════════
   EduPredict — Frontend Script
   Handles: tabs, predict, dataset CRUD, insights chart
════════════════════════════════════════════════════════ */

// ── TAB NAVIGATION ────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    switchTab(tab);
  });
});

function switchTab(tab) {
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById(`nav-${tab}`).classList.add('active');
  document.getElementById(`tab-${tab}`).classList.add('active');
  if (tab === 'dataset')  loadDataset();
  if (tab === 'insights') loadInsights();
}

// ── SLIDER LIVE LABELS ────────────────────────────────
function bindSliders() {
  const map = [
    { id: 'study_hours',          valId: 'val-study',      suffix: 'h' },
    { id: 'attendance',           valId: 'val-attendance',  suffix: '%' },
    { id: 'previous_marks',       valId: 'val-marks',       suffix: '' },
    { id: 'assignments_completed',valId: 'val-assign',      suffix: '' },
  ];
  map.forEach(({ id, valId, suffix }) => {
    const slider = document.getElementById(id);
    const label  = document.getElementById(valId);
    function update() {
      label.textContent = slider.value + suffix;
      updateSliderFill(slider);
    }
    slider.addEventListener('input', update);
    update();
  });
}

function updateSliderFill(slider) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background =
    `linear-gradient(to right, var(--primary) ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
}

// ── PREDICT ───────────────────────────────────────────
async function runPrediction() {
  const btn = document.getElementById('predict-btn');
  btn.classList.add('loading');
  btn.querySelector('span:last-child').textContent = 'Predicting…';

  const payload = {
    study_hours:           parseFloat(document.getElementById('study_hours').value),
    attendance:            parseFloat(document.getElementById('attendance').value),
    previous_marks:        parseFloat(document.getElementById('previous_marks').value),
    assignments_completed: parseFloat(document.getElementById('assignments_completed').value),
  };

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      showResult(data);
    } else {
      showToast('❌ Prediction failed: ' + data.error);
    }
  } catch (e) {
    showToast('❌ Could not connect to server.');
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('span:last-child').textContent = 'Predict Score';
  }
}

function showResult(data) {
  document.getElementById('result-placeholder').classList.add('hidden');
  const content = document.getElementById('result-content');
  content.classList.remove('hidden');

  const score = data.predicted_score;

  // Animated counter
  animateCounter('score-display', 0, score, 1000);

  // Ring fill  (circumference = 2π × 80 ≈ 502.65)
  const C = 502.65;
  const offset = C - (score / 100) * C;
  setTimeout(() => {
    document.getElementById('ring-fill').style.strokeDashoffset = offset;
  }, 80);

  // Grade badge
  const badge = document.getElementById('grade-badge');
  badge.textContent = data.grade;
  badge.className = `grade-badge ${data.grade_class}`;

  // Performance bar
  setTimeout(() => {
    document.getElementById('perf-bar').style.width = score + '%';
  }, 100);

  // Stats pills
  document.getElementById('stat-mae').textContent = data.mae;
  document.getElementById('stat-r2').textContent  = data.r2;
  const acc = Math.round((1 - data.mae / 100) * 100);
  document.getElementById('stat-acc').textContent = acc + '%';

  // Feedback
  const fb = document.getElementById('feedback-box');
  fb.className = 'feedback-box visible';
  if (score >= 80) {
    fb.className += ' feedback-good';
    fb.textContent = '🌟 Excellent performance predicted! Keep up the strong study habits and consistent attendance.';
  } else if (score >= 60) {
    fb.className += ' feedback-mid';
    fb.textContent = '📈 Decent score! Try increasing study hours and completing more assignments to push into the A range.';
  } else {
    fb.className += ' feedback-low';
    fb.textContent = '⚠️ Below average prediction. Focus on boosting attendance, study time, and assignment completion rate.';
  }
}

function animateCounter(id, from, to, duration) {
  const el = document.getElementById(id);
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (from + (to - from) * eased).toFixed(1);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── DATASET ───────────────────────────────────────────
async function loadDataset() {
  try {
    const res  = await fetch('/api/dataset');
    const data = await res.json();
    if (data.success) renderTable(data.data);
  } catch (e) {
    document.getElementById('table-body').innerHTML =
      '<tr><td colspan="7" class="loading-row">Failed to load data.</td></tr>';
  }
}

function renderTable(rows) {
  document.getElementById('record-count').textContent = `${rows.length} student records`;
  const tbody = document.getElementById('table-body');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="loading-row">No records yet.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((r, i) => `
    <tr>
      <td style="color:var(--muted)">${i + 1}</td>
      <td>${r.study_hours}h</td>
      <td>${r.attendance}%</td>
      <td>${r.previous_marks}</td>
      <td>${r.assignments_completed}/10</td>
      <td class="score-cell">${r.final_score}</td>
      <td><button class="delete-btn" onclick="deleteStudent(${i})">🗑 Delete</button></td>
    </tr>
  `).join('');
}

async function addStudent() {
  const payload = {
    study_hours:           parseFloat(document.getElementById('add-study').value),
    attendance:            parseFloat(document.getElementById('add-attendance').value),
    previous_marks:        parseFloat(document.getElementById('add-marks').value),
    assignments_completed: parseFloat(document.getElementById('add-assignments').value),
    final_score:           parseFloat(document.getElementById('add-score').value),
  };

  // Validate
  for (const [k, v] of Object.entries(payload)) {
    if (isNaN(v)) { showFeedback('add-feedback', 'error', '⚠️ Please fill in all fields correctly.'); return; }
  }

  try {
    const res  = await fetch('/api/add-student', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      showFeedback('add-feedback', 'success', '✅ ' + data.message);
      loadDataset();
      showToast('✅ Student added and model retrained!');
    } else {
      showFeedback('add-feedback', 'error', '❌ ' + data.error);
    }
  } catch (e) {
    showFeedback('add-feedback', 'error', '❌ Failed to connect to server.');
  }
}

async function deleteStudent(index) {
  if (!confirm(`Delete student record #${index + 1}?`)) return;
  try {
    const res  = await fetch(`/api/delete-student/${index}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      loadDataset();
      showToast('🗑 Record deleted and model retrained!');
    } else {
      showToast('❌ ' + data.error);
    }
  } catch (e) {
    showToast('❌ Failed to connect to server.');
  }
}

function showFeedback(id, type, msg) {
  const el = document.getElementById(id);
  el.className = `add-feedback show ${type}`;
  el.textContent = msg;
  setTimeout(() => { el.className = 'add-feedback'; }, 4000);
}

// ── INSIGHTS ──────────────────────────────────────────
async function loadInsights() {
  try {
    const res  = await fetch('/api/feature-importance');
    const data = await res.json();
    if (data.success) renderInsights(data);
  } catch (e) {
    document.getElementById('chart-bars').innerHTML =
      '<div class="loading-placeholder">Failed to load insights.</div>';
  }
}

const FEATURE_LABELS = {
  study_hours:           '📚 Study Hours',
  attendance:            '🏫 Attendance',
  previous_marks:        '📋 Previous Marks',
  assignments_completed: '✅ Assignments Done',
};

function renderInsights(data) {
  // Big stats
  document.getElementById('ins-r2').textContent       = data.r2;
  document.getElementById('ins-mae').textContent      = data.mae;
  document.getElementById('ins-students').textContent = data.total_students;

  // Chart bars
  const container = document.getElementById('chart-bars');
  const maxImp = Math.max(...data.importance);

  container.innerHTML = data.features.map((feat, i) => {
    const pct = ((data.importance[i] / maxImp) * 100).toFixed(1);
    const label = FEATURE_LABELS[feat] || feat;
    return `
      <div class="chart-bar-row">
        <div class="chart-bar-label">${label}</div>
        <div class="chart-bar-track">
          <div class="chart-bar-fill" data-width="${pct}">
            <span class="chart-bar-pct">${(data.importance[i] * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Animate bars after render
  setTimeout(() => {
    container.querySelectorAll('.chart-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 80);
}

// ── TOAST ──────────────────────────────────────────────
function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── SVG GRADIENT for ring ─────────────────────────────
function injectRingGradient() {
  const svg = document.querySelector('.score-ring');
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `
    <linearGradient id="ring-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#7c6aff"/>
      <stop offset="100%" stop-color="#22d3ee"/>
    </linearGradient>
  `;
  svg.insertBefore(defs, svg.firstChild);
}

// ── INIT ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  bindSliders();
  injectRingGradient();
  // Pre-warm: load insights data for stats even on predict tab
});
