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
  if (tab === 'mlops')    loadMLOps();
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
      showToast('Prediction failed: ' + data.error);
    }
  } catch (e) {
    showToast('Could not connect to server.');
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('span:last-child').textContent = 'Predict Score';
  }
}

let lastPredictionPayload = null;

function showResult(data) {
  document.getElementById('result-placeholder').classList.add('hidden');
  const content = document.getElementById('result-content');
  content.classList.remove('hidden');

  const score = data.predicted_score;

  // Save prediction details for AI Advisor
  lastPredictionPayload = {
    study_hours:           parseFloat(document.getElementById('study_hours').value),
    attendance:            parseFloat(document.getElementById('attendance').value),
    previous_marks:        parseFloat(document.getElementById('previous_marks').value),
    assignments_completed: parseFloat(document.getElementById('assignments_completed').value),
    predicted_score:       score
  };

  // Reset AI Advisor UI
  const contentArea = document.getElementById('ai-content-area');
  const generateBtn = document.getElementById('ai-generate-btn');
  if (contentArea) contentArea.classList.add('hidden');
  if (generateBtn) generateBtn.classList.remove('hidden');
  const responseText = document.getElementById('ai-response-text');
  if (responseText) responseText.innerHTML = '';

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



  // Feedback
  const fb = document.getElementById('feedback-box');
  fb.className = 'feedback-box visible';
  if (score >= 80) {
    fb.className += ' feedback-good';
    fb.textContent = 'Excellent performance predicted! Keep up the strong study habits and consistent attendance.';
  } else if (score >= 60) {
    fb.className += ' feedback-mid';
    fb.textContent = 'Decent score! Try increasing study hours and completing more assignments to push into the A range.';
  } else {
    fb.className += ' feedback-low';
    fb.textContent = 'Below average prediction. Focus on boosting attendance, study time, and assignment completion rate.';
  }

  // SHAP Explanations
  const shapList = document.getElementById('shap-list');
  if (shapList && data.explanations) {
    // Sort features by absolute impact (highest impact first)
    const sorted = [...data.explanations].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
    
    shapList.innerHTML = sorted.map(item => {
      const label = FEATURE_LABELS[item.feature] || item.feature;
      const impact = item.impact;
      
      let cssClass = 'neutral';
      let signText = '';
      if (impact > 0) {
        cssClass = 'positive';
        signText = '+';
      } else if (impact < 0) {
        cssClass = 'negative';
      }
      
      return `
        <div class="shap-item">
          <span class="shap-feat-name">${label}</span>
          <span class="shap-impact-val ${cssClass}">${signText}${impact.toFixed(2)}</span>
        </div>
      `;
    }).join('');
  }

  // Trigger GSAP Sparkle Burst around the score ring
  const scoreCenter = document.querySelector('.score-center');
  if (scoreCenter) {
    setTimeout(() => {
      spawnSparkles(scoreCenter, 30);
    }, 450);
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
      showFeedback('add-feedback', 'success', data.message);
      loadDataset();
      showToast('Student added and model retrained!');
    } else {
      showFeedback('add-feedback', 'error', data.error);
    }
  } catch (e) {
    showFeedback('add-feedback', 'error', 'Failed to connect to server.');
  }
}

async function deleteStudent(index) {
  if (!confirm(`Delete student record #${index + 1}?`)) return;
  try {
    const res  = await fetch(`/api/delete-student/${index}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      loadDataset();
      showToast('Record deleted and model retrained!');
    } else {
      showToast(data.error);
    }
  } catch (e) {
    showToast('Failed to connect to server.');
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
  study_hours:            'Study Hours',
  attendance:             'Attendance',
  previous_marks:         'Previous Marks',
  assignments_completed:  'Assignments Done',
  study_hours_attendance: 'Study-Attendance Interaction',
  study_hours_log:        'Study Efficiency Curve',
  assignment_marks_ratio: 'Assignment/Marks Ratio'
};

function renderInsights(data) {
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
      <stop offset="0%"   stop-color="#5a8a00"/>
      <stop offset="100%" stop-color="#c97a00"/>
    </linearGradient>
  `;
  svg.insertBefore(defs, svg.firstChild);
}

// ── MLOPS SYSTEM INTEGRATION ──────────────────────────
async function fetchActiveModelVersion() {
  try {
    const res = await fetch('/api/mlops/history');
    const data = await res.json();
    if (data.success && data.active_version) {
      document.getElementById('header-model-ver').textContent = `Model: ${data.active_version}`;
      return data.active_version;
    }
  } catch (e) {}
  document.getElementById('header-model-ver').textContent = 'Model: Unknown';
}

async function loadMLOps() {
  const tbody = document.getElementById('mlops-history-body');
  tbody.innerHTML = '<tr><td colspan="7" class="loading-row">Loading registry runs…</td></tr>';
  
  try {
    const res  = await fetch('/api/mlops/history');
    const data = await res.json();
    if (data.success) {
      renderMLOpsDashboard(data);
    } else {
      showToast('Failed to load MLOps: ' + data.error);
    }
  } catch (e) {
    showToast('Failed to connect to registry API.');
  }
}

function renderMLOpsDashboard(data) {
  const activeVer = data.active_version;
  const history = data.history || [];
  
  const activeRun = history.find(run => run.version === activeVer) || {};
  
  document.getElementById('mlops-active-ver').textContent = activeVer || 'None';
  document.getElementById('mlops-active-mae').textContent = activeRun.mae !== undefined ? activeRun.mae : '—';
  document.getElementById('mlops-active-size').textContent = activeRun.data_size !== undefined ? activeRun.data_size : '—';
  document.getElementById('mlops-active-date').textContent = activeRun.created_at || '—';
  
  if (activeVer) {
    document.getElementById('header-model-ver').textContent = `Model: ${activeVer}`;
  }
  
  document.getElementById('mlops-runs-count').textContent = `${history.length} model version${history.length !== 1 ? 's' : ''} registered`;
  
  const tbody = document.getElementById('mlops-history-body');
  if (!history.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="loading-row">No runs registered yet. Retrain the model to create one.</td></tr>';
    return;
  }
  
  const rowsHtml = [...history].reverse().map(run => {
    const isActive = run.version === activeVer;
    const statusBadge = isActive 
      ? '<span class="mlops-badge active-badge">Serving</span>' 
      : '<span class="mlops-badge idle-badge">Idle</span>';
      
    const actionBtn = isActive 
      ? '<button class="rollback-btn disabled" disabled>Active</button>' 
      : `<button class="rollback-btn" onclick="rollbackToVersion('${run.version}')">Activate</button>`;
      
    return `
      <tr class="${isActive ? 'row-active' : ''}">
        <td class="ver-cell font-bold">${run.version}</td>
        <td>${run.created_at}</td>
        <td>${run.data_size} students</td>
        <td>${run.mae}</td>
        <td>${statusBadge}</td>
        <td>${actionBtn}</td>
      </tr>
    `;
  }).join('');
  
  tbody.innerHTML = rowsHtml;
}

async function rollbackToVersion(version) {
  if (!confirm(`Are you sure you want to activate/rollback to model version ${version}?`)) return;
  try {
    const res = await fetch('/api/mlops/rollback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Successfully activated model version ${version}!`);
      loadMLOps();
    } else {
      showToast('Rollback failed: ' + data.error);
    }
  } catch (e) {
    showToast('Failed to connect to rollback API.');
  }
}

async function triggerManualRetrain() {
  const btn = document.querySelector('.mlops-btn');
  btn.style.opacity = '0.7';
  btn.style.pointerEvents = 'none';
  btn.querySelector('span').textContent = 'Pipeline running...';
  
  try {
    const res = await fetch('/api/retrain', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast(`Model retrained successfully! Version ${data.active_version} created.`);
      loadMLOps();
    } else {
      showToast('Retrain failed: ' + data.error);
    }
  } catch (e) {
    showToast('Connection error during retraining.');
  } finally {
    btn.style.opacity = '1';
    btn.style.pointerEvents = 'auto';
    btn.querySelector('span').textContent = 'Force Pipeline Retraining';
  }
}

// ── AI STUDY PLAN GENERATOR (LLM) ─────────────────────
async function generateAISuggestions() {
  if (!lastPredictionPayload) {
    showToast('No active prediction data found.');
    return;
  }

  const generateBtn = document.getElementById('ai-generate-btn');
  const contentArea = document.getElementById('ai-content-area');
  const spinner = document.getElementById('ai-loading-spinner');
  const responseText = document.getElementById('ai-response-text');

  // Configure UI state
  generateBtn.classList.add('hidden');
  contentArea.classList.remove('hidden');
  spinner.classList.remove('hidden');
  responseText.innerHTML = '';

  try {
    const res = await fetch('/api/generate-advice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(lastPredictionPayload)
    });
    const data = await res.json();
    if (data.success) {
      responseText.innerHTML = parseMarkdown(data.advice);
      if (data.is_mock) {
        showToast('Displaying simulated coaching suggestions.');
      } else {
        showToast('AI Coaching Plan generated successfully!');
      }
      
      // Trigger GSAP sparkles on AI Header icon
      const aiIcon = document.querySelector('.ai-icon');
      if (aiIcon) {
        spawnSparkles(aiIcon, 25);
      }
    } else {
      showToast('Advisor model failed: ' + data.error);
      generateBtn.classList.remove('hidden');
      contentArea.classList.add('hidden');
    }
  } catch (e) {
    showToast('Failed to reach the AI study adviser.');
    generateBtn.classList.remove('hidden');
    contentArea.classList.add('hidden');
  } finally {
    spinner.classList.add('hidden');
  }
}

function parseMarkdown(md) {
  if (!md) return '';
  let html = md;
  
  // Basic escaping for security
  html = html
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Re-allow blockquotes since we escaped '>'
  html = html.replace(/^&gt;\s*(.*)$/gm, '<blockquote>$1</blockquote>');
  
  // Headings
  html = html.replace(/^####\s+(.*)$/gm, '<h4>$1</h4>');
  html = html.replace(/^###\s+(.*)$/gm, '<h3>$1</h3>');
  html = html.replace(/^##\s+(.*)$/gm, '<h2>$1</h2>');
  html = html.replace(/^#\s+(.*)$/gm, '<h1>$1</h1>');

  // Bold Text
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Lists
  // Parse lines to detect list blocks
  let lines = html.split('\n');
  let inUl = false;
  let inOl = false;
  
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();
    
    // Unordered list items: starts with * or -
    if (/^[\*\-]\s+(.*)$/.test(line)) {
      lines[i] = lines[i].replace(/^[\*\-]\s+(.*)$/, '<li>$1</li>');
      if (!inUl) {
        lines[i] = '<ul>' + lines[i];
        inUl = true;
      }
      if (inOl) {
        lines[i] = '</ol>' + lines[i];
        inOl = false;
      }
    } 
    // Ordered list items: starts with digits + dot
    else if (/^\d+\.\s+(.*)$/.test(line)) {
      lines[i] = lines[i].replace(/^\d+\.\s+(.*)$/, '<li>$1</li>');
      if (!inOl) {
        lines[i] = '<ol>' + lines[i];
        inOl = true;
      }
      if (inUl) {
        lines[i] = '</ul>' + lines[i];
        inUl = false;
      }
    } 
    // Regular line
    else {
      if (inUl) {
        lines[i] = '</ul>' + lines[i];
        inUl = false;
      }
      if (inOl) {
        lines[i] = '</ol>' + lines[i];
        inOl = false;
      }
    }
  }
  
  // Close any open lists at the end
  if (inUl) lines[lines.length - 1] += '</ul>';
  if (inOl) lines[lines.length - 1] += '</ol>';
  
  html = lines.join('\n');

  // Add spacers for paragraphs
  html = html.replace(/\n\n/g, '<div class="ai-p-spacing"></div>');
  html = html.replace(/\n/g, '<br/>');

  return html;
}

function spawnSparkles(targetElement, count = 25) {
  if (!window.gsap) return;
  
  const rect = targetElement.getBoundingClientRect();
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
  const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
  const centerX = rect.left + rect.width / 2 + scrollLeft;
  const centerY = rect.top + rect.height / 2 + scrollTop;
  
  const colors = ['#FFCD7F', '#C5FF7F', '#ffffff', '#ffb703', '#5a8a00', '#c97a00'];
  
  for (let i = 0; i < count; i++) {
    const sparkle = document.createElement('div');
    const isStar = Math.random() > 0.4;
    sparkle.className = `gsap-sparkle ${isStar ? 'star' : ''}`;
    
    const size = Math.floor(Math.random() * 8) + 6;
    sparkle.style.width = `${size}px`;
    sparkle.style.height = `${size}px`;
    sparkle.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
    
    sparkle.style.left = `${centerX - size/2}px`;
    sparkle.style.top = `${centerY - size/2}px`;
    
    document.body.appendChild(sparkle);
    
    const angle = Math.random() * Math.PI * 2;
    const distance = Math.floor(Math.random() * 90) + 40;
    const destX = Math.cos(angle) * distance;
    const destY = Math.sin(angle) * distance;
    
    gsap.to(sparkle, {
      x: destX,
      y: destY,
      rotation: Math.random() * 720 - 360,
      opacity: 0,
      scale: 0.1,
      duration: Math.random() * 0.8 + 0.6,
      ease: "power2.out",
      onComplete: () => {
        sparkle.remove();
      }
    });
  }
}

// ── INIT ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  bindSliders();
  injectRingGradient();
  fetchActiveModelVersion();
});
