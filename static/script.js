/* ════════════════════════════════════════════════════════
   EduPredict — Frontend Script
   Handles: tabs, predict, dataset CRUD, insights chart
════════════════════════════════════════════════════════ */

// Premium Request Cache Manager (Stale-While-Revalidate pattern)
const requestCache = new Map();

async function cachedFetch(url, options = {}) {
  const cacheKey = url + (options.body ? `_${options.body}` : '');
  
  if (requestCache.has(cacheKey)) {
    const cachedData = requestCache.get(cacheKey);
    fetch(url, options)
      .then(res => res.json())
      .then(data => {
        requestCache.set(cacheKey, data);
      }).catch(err => console.warn('Background refetch failed:', err));
      
    return cachedData;
  }
  
  const res = await fetch(url, options);
  const data = await res.json();
  requestCache.set(cacheKey, data);
  return data;
}


// ── TAB NAVIGATION ────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    switchTab(tab);
  });
});

function switchTab(tab) {
  const currentActive = document.querySelector('.tab-content.active');
  const targetTab = document.getElementById(`tab-${tab}`);
  
  if (currentActive === targetTab) return;
  
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`nav-${tab}`).classList.add('active');
  
  if (currentActive) {
    currentActive.classList.add('exiting');
    currentActive.classList.remove('active');
    
    setTimeout(() => {
      currentActive.classList.remove('exiting');
      
      targetTab.classList.add('active');
      if (tab === 'dataset')  loadDataset();
      if (tab === 'insights') loadInsights();
      if (tab === 'mlops')    loadMLOps();
    }, 220); // Syncs with CSS transition speed (220ms)
  } else {
    targetTab.classList.add('active');
    if (tab === 'dataset')  loadDataset();
    if (tab === 'insights') loadInsights();
    if (tab === 'mlops')    loadMLOps();
  }
}

// ── SLIDER LIVE LABELS ────────────────────────────────
function bindSliders() {
  const map = [
    { id: 'attendance',           valId: 'val-attendance',  suffix: '%' },
    { id: 'previous_marks',       valId: 'val-marks',       suffix: '' },
  ];
  map.forEach(({ id, valId, suffix }) => {
    const slider = document.getElementById(id);
    const label  = document.getElementById(valId);
    if (!slider || !label) return;
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

  const zsToggle = document.getElementById('zero_shot_toggle');
  const payload = {
    student_id:     (document.getElementById('student_id').value || '').trim() || 'default_student',
    attendance:     parseFloat(document.getElementById('attendance').value),
    previous_marks: parseFloat(document.getElementById('previous_marks').value),
    notes:          (document.getElementById('notes') ? document.getElementById('notes').value : ''),
    zero_shot:      zsToggle ? zsToggle.checked : false
  };
  for (let w = 1; w <= 4; w++) {
    payload[`study_hours_w${w}`] =           parseFloat(document.getElementById(`study_hours_w${w}`).value) || 0;
    payload[`sleep_hours_w${w}`] =           parseFloat(document.getElementById(`sleep_hours_w${w}`).value) || 0;
    payload[`lms_logins_w${w}`] =            parseFloat(document.getElementById(`lms_logins_w${w}`).value) || 0;
    payload[`assignments_completed_w${w}`] = parseFloat(document.getElementById(`assignments_completed_w${w}`).value) || 0;
    payload[`mock_exams_w${w}`] =            parseFloat(document.getElementById(`mock_exams_w${w}`).value) || 0;
  }

  const skeletonTimeout = setTimeout(() => {
    document.getElementById('result-placeholder').classList.add('hidden');
    document.getElementById('result-content').classList.add('hidden');
    const skeleton = document.getElementById('prediction-skeleton');
    if (skeleton) {
      skeleton.style.display = 'flex';
      skeleton.classList.remove('hidden');
    }
  }, 200);

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
    clearTimeout(skeletonTimeout);
    const skeleton = document.getElementById('prediction-skeleton');
    if (skeleton) {
      skeleton.style.display = 'none';
      skeleton.classList.add('hidden');
    }
    btn.classList.remove('loading');
    btn.querySelector('span:last-child').textContent = 'Predict Score';
  }
}

let lastPredictionPayload = null;
let lastPredictionResult = null;

function showResult(data) {
  lastPredictionResult = data;
  document.getElementById('result-placeholder').classList.add('hidden');
  const skeleton = document.getElementById('prediction-skeleton');
  if (skeleton) {
    skeleton.style.display = 'none';
    skeleton.classList.add('hidden');
  }
  const content = document.getElementById('result-content');
  content.classList.remove('hidden');
  const advisorCard = document.getElementById('ai-advisor-card');
  if (advisorCard) advisorCard.classList.remove('hidden');

  const studentId = (document.getElementById('student_id').value || '').trim() || 'default_student';
  loadDKTMastery(studentId);

  const score = data.predicted_score;

  // Save prediction details for AI Advisor
  let sumStudy = 0, sumSleep = 0, sumLms = 0, sumAssign = 0, sumMock = 0;
  for (let w = 1; w <= 4; w++) {
    sumStudy +=  parseFloat(document.getElementById(`study_hours_w${w}`).value) || 0;
    sumSleep +=  parseFloat(document.getElementById(`sleep_hours_w${w}`).value) || 0;
    sumLms +=    parseFloat(document.getElementById(`lms_logins_w${w}`).value) || 0;
    sumAssign += parseFloat(document.getElementById(`assignments_completed_w${w}`).value) || 0;
    sumMock +=   parseFloat(document.getElementById(`mock_exams_w${w}`).value) || 0;
  }

  lastPredictionPayload = {
    study_hours:           parseFloat((sumStudy / 4).toFixed(1)),
    attendance:            parseFloat(document.getElementById('attendance').value),
    previous_marks:        parseFloat(document.getElementById('previous_marks').value),
    assignments_completed: parseFloat((sumAssign / 4).toFixed(1)),
    sleep_hours:           parseFloat((sumSleep / 4).toFixed(1)),
    lms_logins:            parseFloat((sumLms / 4).toFixed(1)),
    mock_exams:            parseFloat((sumMock / 4).toFixed(1)),
    burnout_risk:          data.burnout_risk,
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

  // Update Burnout Badge
  const burnoutBadge = document.getElementById('burnout-badge');
  if (burnoutBadge) {
    burnoutBadge.textContent = data.burnout_risk;
    burnoutBadge.className = `burnout-badge risk-${data.burnout_risk.toLowerCase()}`;
  }

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

  // Render Waterfall Plot
  renderWaterfallPlot(data);

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
    const sorted = [...data.explanations];
    
    shapList.innerHTML = sorted.map(item => {
      const label = FEATURE_LABELS[item.feature] || item.feature;
      const impact = item.impact;
      
      // Get student's actual input score for this feature
      let valueText = '';
      const inputEl = document.getElementById(item.feature);
      if (inputEl) {
        let suffix = '';
        if (item.feature === 'attendance') suffix = '%';
        else if (item.feature.startsWith('study_hours')) suffix = 'h';
        else if (item.feature.startsWith('sleep_hours')) suffix = 'h';
        else if (item.feature.startsWith('assignments_completed')) suffix = '/10';
        valueText = ` (${inputEl.value}${suffix})`;
      }
      
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
          <span class="shap-feat-name">${label}${valueText}</span>
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

  // Update Personalization Bias
  const biasEl = document.getElementById('personalization-bias');
  if (biasEl) {
    const biasVal = data.personalization_bias || 0.0;
    const sign = biasVal >= 0 ? '+' : '';
    biasEl.textContent = `${sign}${biasVal.toFixed(2)}`;
  }
  
  const adaptBadge = document.getElementById('adaptation-badge');
  if (adaptBadge && data.profile_status) {
    adaptBadge.textContent = data.profile_status;
    if (data.profile_status === "One-Shot (Adapted)") {
      adaptBadge.style.background = 'rgba(34, 197, 94, 0.08)';
      adaptBadge.style.color = '#22c55e';
    } else if (data.profile_status === "Zero-Shot (MAML Baseline)") {
      adaptBadge.style.background = 'rgba(234, 179, 8, 0.08)';
      adaptBadge.style.color = '#eab308';
    } else {
      adaptBadge.style.background = 'rgba(0, 124, 255, 0.08)';
      adaptBadge.style.color = '#007cff';
    }
  }
  
  // Update Uncertainty Stats (visible in Admin view only)
  const boundsEl = document.getElementById('score-bounds');
  if (boundsEl) {
    const isChecked = document.getElementById('admin-mode-toggle') ? document.getElementById('admin-mode-toggle').checked : false;
    if (data.uncertainty !== undefined && isChecked) {
      boundsEl.textContent = `/ 100 (± ${data.uncertainty.toFixed(2)})`;
    } else {
      boundsEl.textContent = `/ 100`;
    }
  }

  const confBadge = document.getElementById('confidence-badge');
  if (confBadge && data.uncertainty !== undefined) {
    const u = data.uncertainty;
    if (u < 1.0) {
      confBadge.textContent = `High (σ = ${u.toFixed(2)})`;
      confBadge.style.background = 'rgba(34, 197, 94, 0.1)';
      confBadge.style.color = '#22c55e';
    } else if (u < 2.0) {
      confBadge.textContent = `Medium (σ = ${u.toFixed(2)})`;
      confBadge.style.background = 'rgba(234, 179, 8, 0.1)';
      confBadge.style.color = '#eab308';
    } else {
      confBadge.textContent = `Low / Active Query (σ = ${u.toFixed(2)})`;
      confBadge.style.background = 'rgba(239, 68, 68, 0.1)';
      confBadge.style.color = '#ef4444';
    }
  }

  // Bind Log Grade feedback submit
  const submitBtn = document.getElementById('submit-feedback-btn');
  if (submitBtn) {
    submitBtn.onclick = async () => {
      const actualInput = document.getElementById('actual-score-input');
      const actualVal = parseFloat(actualInput.value);
      if (isNaN(actualVal) || actualVal < 0 || actualVal > 100) {
        showToast('Please enter a valid actual score between 0 and 100.');
        return;
      }
      
      submitBtn.disabled = true;
      submitBtn.textContent = 'Logging...';
      
      try {
        const studentId = (document.getElementById('student_id').value || '').trim() || 'default_student';
        const res = await fetch('/api/log-feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            student_id: studentId,
            actual_score: actualVal,
            predicted_score: data.predicted_score,
            features: {
              attendance: parseFloat(document.getElementById('attendance').value),
              previous_marks: parseFloat(document.getElementById('previous_marks').value),
              study_hours_w1: parseFloat(document.getElementById('study_hours_w1').value) || 0,
              sleep_hours_w1: parseFloat(document.getElementById('sleep_hours_w1').value) || 0,
              lms_logins_w1: parseFloat(document.getElementById('lms_logins_w1').value) || 0,
              assignments_completed_w1: parseFloat(document.getElementById('assignments_completed_w1').value) || 0,
              mock_exams_w1: parseFloat(document.getElementById('mock_exams_w1').value) || 0,
              study_hours_w2: parseFloat(document.getElementById('study_hours_w2').value) || 0,
              sleep_hours_w2: parseFloat(document.getElementById('sleep_hours_w2').value) || 0,
              lms_logins_w2: parseFloat(document.getElementById('lms_logins_w2').value) || 0,
              assignments_completed_w2: parseFloat(document.getElementById('assignments_completed_w2').value) || 0,
              mock_exams_w2: parseFloat(document.getElementById('mock_exams_w2').value) || 0,
              study_hours_w3: parseFloat(document.getElementById('study_hours_w3').value) || 0,
              sleep_hours_w3: parseFloat(document.getElementById('sleep_hours_w3').value) || 0,
              lms_logins_w3: parseFloat(document.getElementById('lms_logins_w3').value) || 0,
              assignments_completed_w3: parseFloat(document.getElementById('assignments_completed_w3').value) || 0,
              mock_exams_w3: parseFloat(document.getElementById('mock_exams_w3').value) || 0,
              study_hours_w4: parseFloat(document.getElementById('study_hours_w4').value) || 0,
              sleep_hours_w4: parseFloat(document.getElementById('sleep_hours_w4').value) || 0,
              lms_logins_w4: parseFloat(document.getElementById('lms_logins_w4').value) || 0,
              assignments_completed_w4: parseFloat(document.getElementById('assignments_completed_w4').value) || 0,
              mock_exams_w4: parseFloat(document.getElementById('mock_exams_w4').value) || 0
            }
          })
        });
        const logData = await res.json();
        if (logData.success) {
          showToast(`Feedback logged successfully! Personalized bias updated to ${logData.new_bias.toFixed(2)}.`);
          actualInput.value = '';
          runPrediction();
        } else {
          showToast('Failed to log feedback: ' + logData.error);
        }
      } catch (e) {
        showToast('Error sending feedback to server.');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Log Grade';
      }
    };
  }
  // Render Dynamic Attention Progress Bars
  const attnWrapper = document.getElementById('attention-bars-wrapper');
  if (attnWrapper && data.attention_weights) {
    attnWrapper.innerHTML = data.attention_weights.map((weight, idx) => {
      const pct = (weight * 100).toFixed(1);
      return `
        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.75rem; margin-bottom: 4px;">
          <span style="font-weight: 600; color: var(--text); width: 110px;">Week ${idx + 1} Attention</span>
          <div style="flex-grow: 1; margin: 0 12px; height: 8px; background: rgba(0,0,0,0.06); border-radius: 4px; overflow: hidden; position: relative;">
            <div style="width: 0%; height: 100%; background: linear-gradient(90deg, var(--primary), var(--secondary)); border-radius: 4px; transition: width 1.2s cubic-bezier(0.1, 0.8, 0.2, 1);" id="attn-bar-w${idx + 1}"></div>
          </div>
          <span style="font-weight: 700; color: var(--primary); width: 45px; text-align: right;">${pct}%</span>
        </div>
      `;
    }).join('');
    
    // Trigger animations sequentially for micro-animation aesthetics!
    data.attention_weights.forEach((weight, idx) => {
      setTimeout(() => {
        const bar = document.getElementById(`attn-bar-w${idx + 1}`);
        if (bar) {
          bar.style.width = (weight * 100).toFixed(1) + '%';
        }
      }, 100 + idx * 80);
    });
  }

  // Smooth scroll to result card after a short delay for rendering
  const resultCard = document.getElementById('result-card');
  if (resultCard) {
    setTimeout(() => {
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 350);
  }
}

function renderWaterfallPlot(data) {
  const container = document.getElementById('waterfall-wrapper');
  if (!container || !data.explanations) return;
  
  const baseValue = data.base_value || 70.0;
  const predScore = data.predicted_score;
  
  const sortedExps = [...data.explanations].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  
  const maxFeaturesToShow = 6;
  const topFeatures = sortedExps.slice(0, maxFeaturesToShow);
  const otherFeatures = sortedExps.slice(maxFeaturesToShow);
  
  const steps = [];
  
  steps.push({
    label: 'Cohort Base Value',
    impact: 0,
    cumulative: baseValue,
    isBase: true
  });
  
  let current = baseValue;
  
  topFeatures.forEach(feat => {
    current += feat.impact;
    const cleanLabel = FEATURE_LABELS[feat.feature] || feat.feature;
    let valText = '';
    const inputEl = document.getElementById(feat.feature);
    if (inputEl) {
      let suffix = '';
      if (feat.feature === 'attendance') suffix = '%';
      else if (feat.feature.startsWith('study_hours')) suffix = 'h';
      else if (feat.feature.startsWith('sleep_hours')) suffix = 'h';
      else if (feat.feature.startsWith('assignments_completed')) suffix = '/10';
      valText = ` (${inputEl.value}${suffix})`;
    }
    steps.push({
      label: cleanLabel + valText,
      impact: feat.impact,
      cumulative: current
    });
  });
  
  if (otherFeatures.length > 0) {
    const otherImpact = otherFeatures.reduce((sum, f) => sum + f.impact, 0);
    current += otherImpact;
    steps.push({
      label: `${otherFeatures.length} Other Features`,
      impact: otherImpact,
      cumulative: current
    });
  }
  
  const biasVal = data.personalization_bias || 0.0;
  if (biasVal !== 0) {
    current += biasVal;
    steps.push({
      label: 'Personalization Bias',
      impact: biasVal,
      cumulative: current
    });
  }
  
  steps.push({
    label: 'Predicted Score',
    impact: 0,
    cumulative: predScore,
    isFinal: true
  });
  
  const allValues = steps.map(s => s.cumulative);
  allValues.push(baseValue);
  allValues.push(predScore);
  const minVal = Math.min(...allValues) - 1.5;
  const maxVal = Math.max(...allValues) + 1.5;
  const range = maxVal - minVal;
  
  const rowHeight = 32;
  const marginTop = 30;
  const marginBottom = 30;
  const marginLeft = 155;
  const marginRight = 55;
  const width = 600;
  const height = marginTop + marginBottom + (steps.length * rowHeight);
  
  const getX = (val) => {
    return marginLeft + ((val - minVal) / range) * (width - marginLeft - marginRight);
  };
  
  let svgContent = `<svg class="waterfall-svg" width="100%" height="100%" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  
  const gridTicks = 5;
  for (let i = 0; i < gridTicks; i++) {
    const val = minVal + (range / (gridTicks - 1)) * i;
    const x = getX(val);
    svgContent += `
      <line x1="${x}" y1="${marginTop - 10}" x2="${x}" y2="${height - marginBottom}" stroke="rgba(0,0,0,0.04)" stroke-dasharray="3,3" />
      <text x="${x}" y="${height - 10}" font-size="9" fill="var(--muted)" font-weight="600" text-anchor="middle">${val.toFixed(1)}</text>
    `;
  }
  
  const baseX = getX(baseValue);
  svgContent += `<line x1="${baseX}" y1="${marginTop - 10}" x2="${baseX}" y2="${height - marginBottom}" stroke="rgba(0,0,0,0.15)" stroke-width="1.5" stroke-dasharray="4,4" />`;
  svgContent += `<text x="${baseX}" y="${marginTop - 15}" font-size="9" fill="var(--muted)" font-weight="700" text-anchor="middle">Base (${baseValue.toFixed(1)})</text>`;
  
  let prevVal = baseValue;
  
  steps.forEach((step, idx) => {
    const y = marginTop + idx * rowHeight + (rowHeight - 16) / 2;
    const barHeight = 16;
    
    let color = 'var(--text)';
    let x1 = 0, x2 = 0;
    
    if (step.isBase) {
      x1 = getX(step.cumulative) - 3;
      x2 = getX(step.cumulative) + 3;
    } else if (step.isFinal) {
      x1 = getX(step.cumulative) - 4;
      x2 = getX(step.cumulative) + 4;
    } else {
      if (step.impact > 0) {
        color = '#3d6b00';
        x1 = getX(prevVal);
        x2 = getX(step.cumulative);
      } else {
        color = '#a82000';
        x1 = getX(step.cumulative);
        x2 = getX(prevVal);
      }
    }
    
    const barWidth = Math.max(2, Math.abs(x2 - x1));
    const rectX = Math.min(x1, x2);
    
    if (idx > 0) {
      const prevY = marginTop + (idx - 1) * rowHeight + (rowHeight - 16) / 2 + barHeight / 2;
      const currY = y + barHeight / 2;
      const connX = getX(prevVal);
      svgContent += `<line x1="${connX}" y1="${prevY}" x2="${connX}" y2="${currY}" stroke="rgba(0,0,0,0.15)" stroke-width="1" stroke-dasharray="2,2" />`;
    }
    
    const labelY = y + barHeight - 4;
    svgContent += `
      <text x="10" y="${labelY}" font-size="10" font-weight="600" fill="var(--text)" text-anchor="start">${step.label}</text>
    `;
    
    let fillColor = 'rgba(120, 120, 120, 0.7)';
    if (step.isBase) {
      fillColor = 'var(--muted)';
    } else if (step.isFinal) {
      fillColor = 'var(--primary)';
    } else if (step.impact > 0) {
      fillColor = '#5a8a00';
    } else {
      fillColor = '#c97a00';
    }
    
    svgContent += `
      <rect x="${rectX}" y="${y}" width="${barWidth}" height="${barHeight}" rx="3" fill="${fillColor}" />
    `;
    
    const impactText = step.isBase ? `${step.cumulative.toFixed(1)}` :
                       step.isFinal ? `${step.cumulative.toFixed(1)}` :
                       (step.impact > 0 ? `+${step.impact.toFixed(2)}` : `${step.impact.toFixed(2)}`);
                       
    const textAnchor = (step.isBase || step.isFinal || step.impact > 0) ? 'start' : 'end';
    const textX = (step.isBase || step.isFinal || step.impact > 0) ? rectX + barWidth + 6 : rectX - 6;
    
    let textStyle = `font-weight: 700; font-size: 10px; fill: ${color};`;
    svgContent += `
      <text x="${textX}" y="${labelY}" style="${textStyle}" text-anchor="${textAnchor}">${impactText}</text>
    `;
    
    if (!step.isBase && !step.isFinal) {
      prevVal = step.cumulative;
    }
  });
  
  svgContent += `</svg>`;
  container.innerHTML = svgContent;
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
    const data = await cachedFetch('/api/dataset');
    if (data.success) renderTable(data.data);
  } catch (e) {
    document.getElementById('table-body').innerHTML =
      '<tr><td colspan="10" class="loading-row">Failed to load data.</td></tr>';
  }
}

function renderTable(rows) {
  document.getElementById('record-count').textContent = `${rows.length} student records`;
  const tbody = document.getElementById('table-body');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="loading-row">No records yet.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((r, i) => `
    <tr>
      <td style="color:var(--muted)">${i + 1}</td>
      <td>${r.study_hours_w4 !== undefined ? r.study_hours_w4 : r.study_hours}h</td>
      <td>${r.attendance}%</td>
      <td>${r.previous_marks}</td>
      <td>${r.assignments_completed_w4 !== undefined ? r.assignments_completed_w4 : (r.assignments_completed || 0)}/10</td>
      <td>${r.sleep_hours_w4 !== undefined ? r.sleep_hours_w4 : (r.sleep_hours || 7.5)}h</td>
      <td>${r.lms_logins_w4 !== undefined ? r.lms_logins_w4 : (r.lms_logins || 30)}</td>
      <td>${r.mock_exams_w4 !== undefined ? r.mock_exams_w4 : (r.mock_exams || r.previous_marks || 70)}</td>
      <td class="score-cell">${r.final_score}</td>
      <td style="font-size:0.75rem; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${r.notes || ''}">${r.notes || '—'}</td>
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
    sleep_hours:           parseFloat(document.getElementById('add-sleep').value),
    lms_logins:            parseFloat(document.getElementById('add-lms').value),
    mock_exams:            parseFloat(document.getElementById('add-mock').value),
    final_score:           parseFloat(document.getElementById('add-score').value),
  };

  // Validate
  for (const [k, v] of Object.entries(payload)) {
    if (isNaN(v)) { showFeedback('add-feedback', 'error', '⚠️ Please fill in all fields correctly.'); return; }
  }

  const schoolEl = document.getElementById('add-school');
  if (schoolEl) {
    payload.school_id = schoolEl.value;
  }
  
  const notesEl = document.getElementById('add-notes');
  if (notesEl) {
    payload.notes = notesEl.value || '';
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
      requestCache.delete('/api/dataset');
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
      requestCache.delete('/api/dataset');
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
    const data = await cachedFetch('/api/feature-importance');
    if (data.success) renderInsights(data);
  } catch (e) {
    document.getElementById('chart-bars').innerHTML =
      '<div class="loading-placeholder">Failed to load insights.</div>';
  }
}

const FEATURE_LABELS = {
  attendance:             'Attendance',
  previous_marks:         'Previous Marks'
};
for (let w = 1; w <= 4; w++) {
  FEATURE_LABELS[`study_hours_w${w}`] =           `Week ${w} Study Hours`;
  FEATURE_LABELS[`sleep_hours_w${w}`] =           `Week ${w} Sleep Hours`;
  FEATURE_LABELS[`lms_logins_w${w}`] =            `Week ${w} LMS Logins`;
  FEATURE_LABELS[`assignments_completed_w${w}`] = `Week ${w} Assignments`;
  FEATURE_LABELS[`mock_exams_w${w}`] =            `Week ${w} Mock Score`;
}

function renderInsights(data) {
  document.getElementById('ins-students').textContent = data.total_students || 0;
  
  const avgScoreEl = document.getElementById('ins-avg-score');
  if (avgScoreEl) avgScoreEl.textContent = (data.avg_predicted_score || 0.0).toFixed(1) + '%';
  
  const avgAttEl = document.getElementById('ins-avg-attendance');
  if (avgAttEl) avgAttEl.textContent = (data.avg_attendance || 0.0).toFixed(1) + '%';
  
  const burnoutPctEl = document.getElementById('ins-burnout-pct');
  if (burnoutPctEl) burnoutPctEl.textContent = (data.burnout_pct || 0.0).toFixed(1) + '%';
  
  const schoolSizes = data.school_sizes || {};
  const alphaEl = document.getElementById('fed-alpha-size');
  if (alphaEl) alphaEl.textContent = (schoolSizes.alpha || 0) + ' students';
  const betaEl = document.getElementById('fed-beta-size');
  if (betaEl) betaEl.textContent = (schoolSizes.beta || 0) + ' students';
  const gammaEl = document.getElementById('fed-gamma-size');
  if (gammaEl) gammaEl.textContent = (schoolSizes.gamma || 0) + ' students';

  // Chart bars
  const container = document.getElementById('chart-bars');
  const maxImp = Math.max(...data.importance);

  const sumImp = data.importance.reduce((a, b) => a + b, 0) || 1.0;
  container.innerHTML = data.features.map((feat, i) => {
    const pct = ((data.importance[i] / maxImp) * 100).toFixed(1);
    const relativePct = ((data.importance[i] / sumImp) * 100).toFixed(1);
    const label = FEATURE_LABELS[feat] || feat;
    return `
      <div class="chart-bar-row">
        <div class="chart-bar-label">${label}</div>
        <div class="chart-bar-track">
          <div class="chart-bar-fill" data-width="${pct}">
            <span class="chart-bar-pct">${relativePct}%</span>
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
  tbody.innerHTML = '<tr><td colspan="6" class="loading-row">Loading registry runs…</td></tr>';
  
  try {
    const data = await cachedFetch('/api/mlops/history');
    if (data.success) {
      renderMLOpsDashboard(data);
      loadActiveLearningQueue();
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
    tbody.innerHTML = '<tr><td colspan="8" class="loading-row">No runs registered yet. Retrain the model to create one.</td></tr>';
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
      
    const fd = run.fairness_district || {};
    const fg = run.fairness_gender || {};
    
    const fdText = fd.demographic_parity_diff !== undefined 
      ? `${(fd.demographic_parity_diff * 100).toFixed(1)}% / ${(fd.equalized_odds_diff * 100).toFixed(1)}%` 
      : '0.0% / 0.0%';
    const fgText = fg.demographic_parity_diff !== undefined 
      ? `${(fg.demographic_parity_diff * 100).toFixed(1)}% / ${(fg.equalized_odds_diff * 100).toFixed(1)}%` 
      : '0.0% / 0.0%';
      
    const fdVal = fd.demographic_parity_diff || 0.0;
    const fgVal = fg.demographic_parity_diff || 0.0;
    
    const fdColor = fdVal < 0.12 ? '#10b981' : '#f59e0b';
    const fgColor = fgVal < 0.12 ? '#10b981' : '#f59e0b';
      
    return `
      <tr class="${isActive ? 'row-active' : ''}">
        <td class="ver-cell font-bold">${run.version}</td>
        <td>${run.created_at}</td>
        <td>${run.data_size} students</td>
        <td>${run.mae}</td>
        <td><span style="color: ${fdColor}; font-weight: 700; font-family: monospace;">⚖️ ${fdText}</span></td>
        <td><span style="color: ${fgColor}; font-weight: 700; font-family: monospace;">⚖️ ${fgText}</span></td>
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
      requestCache.delete('/api/mlops/history');
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
      requestCache.delete('/api/mlops/history');
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

  const skeletonTimeout = setTimeout(() => {
    if (spinner) spinner.classList.add('hidden');
    const skeleton = document.getElementById('ai-skeleton');
    if (skeleton) {
      skeleton.style.display = 'flex';
      skeleton.classList.remove('hidden');
    }
  }, 200);

  try {
    const rlToggle = document.getElementById('rl_advisor_toggle');
    const advicePayload = {
      ...lastPredictionPayload,
      explanations: lastPredictionResult ? lastPredictionResult.explanations : [],
      enable_rl: rlToggle ? rlToggle.checked : true
    };
    const res = await fetch('/api/generate-advice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(advicePayload)
    });
    const data = await res.json();
    if (data.success) {
      const skeleton = document.getElementById('ai-skeleton');
      if (skeleton) {
        skeleton.style.display = 'none';
        skeleton.classList.add('hidden');
      }
      responseText.innerHTML = parseMarkdown(data.advice);
      
      // Render Agent ReAct Console Logs
      const consoleWrapper = document.getElementById('agent-console-wrapper');
      const consoleLogs = document.getElementById('agent-console-logs');
      const isAdmin = document.getElementById('admin-mode-toggle').checked;
      
      if (consoleWrapper && consoleLogs && data.agent_logs && isAdmin) {
        consoleWrapper.classList.remove('hidden');
        consoleLogs.innerHTML = '';
        data.agent_logs.forEach((log) => {
          let color = '#f8fafc';
          if (log.startsWith('Thought:')) color = '#38bdf8';
          else if (log.startsWith('Action:')) color = '#a78bfa';
          else if (log.startsWith('Observation:')) color = '#34d399';
          
          const logLine = document.createElement('div');
          logLine.style.color = color;
          logLine.style.marginBottom = '6px';
          logLine.style.borderLeft = `3px solid ${color}`;
          logLine.style.paddingLeft = '8px';
          logLine.textContent = log;
          consoleLogs.appendChild(logLine);
        });
      } else if (consoleWrapper) {
        consoleWrapper.classList.add('hidden');
      }

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
    clearTimeout(skeletonTimeout);
    const skeleton = document.getElementById('ai-skeleton');
    if (skeleton) {
      skeleton.style.display = 'none';
      skeleton.classList.add('hidden');
    }
    spinner.classList.add('hidden');
  }
}

function parseMarkdown(md) {
  if (!md) return '';
  let html = md;
  
  const isAdmin = document.getElementById('admin-mode-toggle') ? document.getElementById('admin-mode-toggle').checked : false;
  if (!isAdmin) {
    // Strip diagnostic details block
    html = html.replace(/<details\b[^>]*>([\s\S]*?)<\/details>/gi, '');
    // Strip multi-agent RL simulation details up to end
    html = html.replace(/(?:###?\s*(?:🎯\s*)?Multi-Agent\s+RL\s+Cooperative[\s\S]*)$/i, '');
  }

  // Basic escaping for security
  html = html
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Restore allowed diagnostics HTML tags
  html = html
    .replace(/&lt;details(.*?)&gt;/g, '<details$1>')
    .replace(/&lt;\/details&gt;/g, '</details>')
    .replace(/&lt;summary(.*?)&gt;/g, '<summary$1>')
    .replace(/&lt;\/summary&gt;/g, '</summary>')
    .replace(/&lt;div(.*?)&gt;/g, '<div$1>')
    .replace(/&lt;\/div&gt;/g, '</div>')
    .replace(/&lt;br\s*\/&gt;/g, '<br/>')
    .replace(/&lt;strong&gt;/g, '<strong>')
    .replace(/&lt;\/strong&gt;/g, '</strong>');

  // Re-allow blockquotes since we escaped '>'
  html = html.replace(/^&gt;\s*(.*)$/gm, '<blockquote>$1</blockquote>');
  
  // Headings
  html = html.replace(/^######\s+(.*)$/gm, '<h6>$1</h6>');
  html = html.replace(/^#####\s+(.*)$/gm, '<h5>$1</h5>');
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
      lines[i] = '<li>' + line.substring(line.indexOf(' ') + 1).trim() + '</li>';
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
      lines[i] = '<li>' + line.substring(line.indexOf(' ') + 1).trim() + '</li>';
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
  setupLiveValidation();
  
  // Initial check for prefers-reduced-motion
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.querySelectorAll('.orb').forEach(orb => orb.style.animation = 'none');
  }
  
  const zsToggle = document.getElementById('zero_shot_toggle');
  if (zsToggle) {
    zsToggle.addEventListener('change', () => {
      const weeklyLogs = document.querySelector('.weekly-logs-section');
      if (weeklyLogs) {
        if (zsToggle.checked) {
          weeklyLogs.classList.add('disabled-section');
        } else {
          weeklyLogs.classList.remove('disabled-section');
        }
      }
    });
  }
});

async function runFederatedTraining() {
  const btn = document.getElementById('run-fed-btn');
  const consoleEl = document.getElementById('fed-console');
  if (!btn || !consoleEl) return;
  
  const rounds = parseInt(document.getElementById('fed-rounds').value) || 3;
  const noiseScale = parseFloat(document.getElementById('fed-noise').value) || 0.01;
  
  btn.disabled = true;
  btn.style.background = '#80bfff';
  btn.textContent = 'Training Aggregated Model...';
  
  consoleEl.innerHTML = `&gt; Starting Federated Learning Simulation round with ${rounds} communication rounds...<br/>&gt; Distributing base hybrid LSTM-Transformer weight structures to Alpha, Beta, and Gamma client databases...`;
  
  try {
    const res = await fetch('/api/federated-train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rounds, noise_scale: noiseScale, epochs: 5 })
    });
    const data = await res.json();
    if (data.success) {
      let index = 0;
      consoleEl.innerHTML = '';
      function showNextLog() {
        if (index < data.logs.length) {
          consoleEl.innerHTML += `&gt; ${data.logs[index]}<br/>`;
          consoleEl.scrollTop = consoleEl.scrollHeight;
          index++;
          setTimeout(showNextLog, 400);
        } else {
          consoleEl.innerHTML += `<br/>&gt; 🎉 Federated training round succeeded! Registered active model as version ${data.active_version}.`;
          consoleEl.scrollTop = consoleEl.scrollHeight;
          btn.disabled = false;
          btn.style.background = '#007cff';
          btn.textContent = 'Run Federated Aggregation';
          
          fetchActiveModelVersion();
          loadInsights();
          showToast('Federated Learning training completed successfully!');
        }
      }
      showNextLog();
    } else {
      consoleEl.innerHTML += `<br/>&gt; ❌ Error: ${data.error}`;
      if (data.logs) {
        data.logs.forEach(l => { consoleEl.innerHTML += `<br/>&gt; ${l}`; });
      }
      btn.disabled = false;
      btn.style.background = '#007cff';
      btn.textContent = 'Run Federated Aggregation';
    }
  } catch (e) {
    consoleEl.innerHTML += `<br/>&gt; ❌ Error: Failed to connect to local federation server.`;
    btn.disabled = false;
    btn.style.background = '#007cff';
    btn.textContent = 'Run Federated Aggregation';
  }
}

function toggleAdminMode() {
  const isChecked = document.getElementById('admin-mode-toggle').checked;
  const label = document.getElementById('view-mode-label');
  
  if (isChecked) {
    if (label) {
      label.textContent = "Admin View";
      label.style.color = "#007cff";
    }
    const datasetTab = document.getElementById('nav-dataset');
    if (datasetTab) datasetTab.style.display = "inline-block";
    const mlopsTab = document.getElementById('nav-mlops');
    if (mlopsTab) mlopsTab.style.display = "inline-block";
    
    const fedCard = document.getElementById('federated-card');
    if (fedCard) fedCard.style.display = "block";
    
    const biasRow = document.getElementById('bias-row');
    if (biasRow) biasRow.style.display = "flex";
    const persCard = document.getElementById('personalization-card');
    if (persCard) persCard.style.display = "block";
    const confRow = document.getElementById('confidence-row');
    if (confRow) confRow.style.display = "flex";
    const adaptRow = document.getElementById('adaptation-row');
    if (adaptRow) adaptRow.style.display = "flex";
    
    const zsContainer = document.getElementById('zero-shot-toggle-container');
    if (zsContainer) zsContainer.style.display = "flex";
    const rlContainer = document.getElementById('rl-advisor-toggle-container');
    if (rlContainer) rlContainer.style.display = "flex";
    
    const dktCard = document.getElementById('dkt-card');
    if (dktCard) dktCard.style.display = "block";
    const counterfactualContainer = document.getElementById('counterfactual-container');
    if (counterfactualContainer) counterfactualContainer.style.display = "block";
    
    const boundsEl = document.getElementById('score-bounds');
    if (boundsEl && lastPredictionResult && lastPredictionResult.uncertainty !== undefined) {
      boundsEl.textContent = `/ 100 (± ${lastPredictionResult.uncertainty.toFixed(2)})`;
    }
    
    showToast("Switched to Admin MLOps Portal");
    loadActiveLearningQueue();
  } else {
    if (label) {
      label.textContent = "User View";
      label.style.color = "var(--muted)";
    }
    const datasetTab = document.getElementById('nav-dataset');
    if (datasetTab) datasetTab.style.display = "none";
    const mlopsTab = document.getElementById('nav-mlops');
    if (mlopsTab) mlopsTab.style.display = "none";
    
    // Fallback switch to predict tab if current active tab is dataset or mlops
    const activeTab = document.querySelector('.nav-btn.active');
    if (activeTab && (activeTab.getAttribute('data-tab') === 'dataset' || activeTab.getAttribute('data-tab') === 'mlops')) {
      // simulate tab switch click
      const predictNavBtn = document.getElementById('nav-predict');
      if (predictNavBtn) predictNavBtn.click();
    }
    
    const fedCard = document.getElementById('federated-card');
    if (fedCard) fedCard.style.display = "none";
    
    const biasRow = document.getElementById('bias-row');
    if (biasRow) biasRow.style.display = "none";
    const persCard = document.getElementById('personalization-card');
    if (persCard) persCard.style.display = "none";
    const confRow = document.getElementById('confidence-row');
    if (confRow) confRow.style.display = "none";
    const adaptRow = document.getElementById('adaptation-row');
    if (adaptRow) adaptRow.style.display = "none";
    
    const zsContainer = document.getElementById('zero-shot-toggle-container');
    if (zsContainer) zsContainer.style.display = "none";
    const rlContainer = document.getElementById('rl-advisor-toggle-container');
    if (rlContainer) rlContainer.style.display = "none";
    
    const dktCard = document.getElementById('dkt-card');
    if (dktCard) dktCard.style.display = "none";
    const counterfactualContainer = document.getElementById('counterfactual-container');
    if (counterfactualContainer) counterfactualContainer.style.display = "none";
    
    const boundsEl = document.getElementById('score-bounds');
    if (boundsEl) {
      boundsEl.textContent = `/ 100`;
    }
    
    showToast("Switched to Student/Teacher View");
  }
}

async function loadActiveLearningQueue() {
  const tbody = document.getElementById('al-queue-body');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="8" class="loading-row">Calculating cohort uncertainty bounds...</td></tr>';
  
  try {
    const res = await fetch('/api/active-learning-queue');
    const data = await res.json();
    if (data.success) {
      if (data.queue.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="loading-row">No student records found in database.</td></tr>';
        return;
      }
      tbody.innerHTML = data.queue.map(row => {
        const u = row.uncertainty;
        let priorityClass = 'risk-low';
        if (row.priority === 'High') priorityClass = 'risk-high';
        else if (row.priority === 'Medium') priorityClass = 'risk-medium';
        
        return `
          <tr>
            <td style="font-weight: 700; color: var(--text);">${row.student_id}</td>
            <td>${row.attendance}%</td>
            <td>${row.previous_marks}</td>
            <td style="font-size: 0.75rem; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${row.notes || ''}">${row.notes || '—'}</td>
            <td style="font-weight: 600;">${row.predicted_score.toFixed(2)}</td>
            <td style="font-weight: 700; font-family: monospace; color: var(--primary);">± ${u.toFixed(2)} (σ²=${row.variance !== undefined ? row.variance.toFixed(2) : (u*u).toFixed(2)})</td>
            <td><span class="burnout-badge ${priorityClass}">${row.priority}</span></td>
            <td>
              <div style="display: flex; gap: 8px; align-items: center;">
                <input type="number" min="0" max="100" placeholder="Actual" class="table-input" id="al-input-${row.student_id}" style="max-width: 70px; padding: 6px; font-size: 0.75rem; border-radius: 6px; border: 1px solid rgba(0,0,0,0.15);" />
                <button class="btn" onclick="submitActiveLearningFeedback('${row.student_id}')" style="padding: 6px 12px; font-size: 0.75rem; border-radius: 6px; background: var(--primary); color: white; border: none; font-weight: 700; cursor: pointer; width: auto;">Log</button>
              </div>
            </td>
          </tr>
        `;
      }).join('');
    } else {
      tbody.innerHTML = `<tr><td colspan="8" class="loading-row" style="color:#ef4444">Error: ${data.error}</td></tr>`;
    }
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="8" class="loading-row">Failed to fetch active learning queue.</td></tr>';
  }
}

async function submitActiveLearningFeedback(studentId) {
  const inputEl = document.getElementById(`al-input-${studentId}`);
  if (!inputEl) return;
  
  const scoreVal = parseFloat(inputEl.value);
  if (isNaN(scoreVal) || scoreVal < 0 || scoreVal > 100) {
    showToast('Please enter a valid actual score between 0 and 100.');
    return;
  }
  
  const btnEl = inputEl.nextElementSibling;
  btnEl.disabled = true;
  btnEl.textContent = 'Logging...';
  
  try {
    const res = await fetch('/api/log-feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_id: studentId, actual_score: scoreVal })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Logged feedback for ${studentId}. Fine-tuned personalized layer!`);
      loadActiveLearningQueue();
    } else {
      showToast('Error logging feedback: ' + data.error);
      btnEl.disabled = false;
      btnEl.textContent = 'Log';
    }
  } catch (e) {
    showToast('Failed to connect to server feedback API.');
    btnEl.disabled = false;
    btnEl.textContent = 'Log';
  }
}

async function calculateCounterfactualRecourse() {
  if (!lastPredictionPayload) {
    showToast('Please run a score prediction first.');
    return;
  }
  
  const targetInput = document.getElementById('target-score-input');
  const resultsArea = document.getElementById('recourse-results-area');
  if (!targetInput || !resultsArea) return;
  
  const targetScore = parseFloat(targetInput.value);
  if (isNaN(targetScore) || targetScore < 50 || targetScore > 100) {
    showToast('Please enter a target score between 50 and 100.');
    return;
  }
  
  resultsArea.style.display = 'block';
  resultsArea.innerHTML = '<div style="font-size: 0.78rem; color: var(--muted);">Optimizing recourse gradient path...</div>';
  
  try {
    const payload = {
      ...lastPredictionPayload,
      target_score: targetScore
    };
    const res = await fetch('/api/counterfactual-recourse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success && data.recourse) {
      const rec = data.recourse;
      const actionsHtml = rec.recourse_actions.map(act => `<li style="margin-bottom: 4px; font-size: 0.78rem; color: #007cff; font-weight: 600;">💡 ${act}</li>`).join('');
      const opt = rec.optimized_metrics;
      
      resultsArea.innerHTML = `
        <div style="background: rgba(0,124,255,0.06); padding: 12px; border-radius: 8px; font-size: 0.8rem; line-height: 1.5; color: var(--text);">
          <div style="font-weight: 700; color: var(--text); margin-bottom: 6px;">
            Target Score: <strong>${rec.target_score}</strong> | Projected Achievable Score: <strong style="color: var(--primary);">${rec.projected_score}</strong> (+${rec.score_gain} marks)
          </div>
          <div style="font-weight: 700; margin-bottom: 4px; font-size: 0.75rem; color: var(--muted);">Minimal Actionable Recourse Steps:</div>
          <ul style="margin: 0 0 10px 15px; padding: 0;">${actionsHtml}</ul>
          <div style="font-size: 0.72rem; color: var(--muted); font-weight: 600;">
            Target Averages: ${opt.avg_study_hours}h/day study, ${opt.avg_sleep_hours}h/night sleep, ${opt.avg_lms_logins} logins/wk, ${opt.avg_assignments_completed} assignments/mod, ${opt.avg_mock_exams} mock score.
          </div>
        </div>
      `;
      showToast('Calculated optimal counterfactual habit shift!');
    } else {
      resultsArea.innerHTML = `<div style="font-size: 0.78rem; color: #ef4444;">Recourse error: ${data.error}</div>`;
    }
  } catch (e) {
    resultsArea.innerHTML = '<div style="font-size: 0.78rem; color: #ef4444;">Failed to calculate counterfactual recourse.</div>';
  }
}

let currentDKTAlgebra = 0.6;
let currentDKTCalculus = 0.55;
let currentDKTMechanics = 0.58;

function redrawDKTRadar(pAlg, pCalc, pMech) {
  document.getElementById('dkt-val-algebra').textContent = (pAlg * 100).toFixed(1) + '%';
  document.getElementById('dkt-val-calculus').textContent = (pCalc * 100).toFixed(1) + '%';
  document.getElementById('dkt-val-mechanics').textContent = (pMech * 100).toFixed(1) + '%';
  
  // Calculate radar polygon coordinates
  const x1 = 70;
  const y1 = 70 - 60 * pAlg;
  
  const x2 = 70 + 51.96 * pCalc;
  const y2 = 70 + 30.00 * pCalc;
  
  const x3 = 70 - 51.96 * pMech;
  const y3 = 70 + 30.00 * pMech;
  
  const poly = document.getElementById('dkt-poly');
  if (poly) {
    poly.setAttribute('points', `${x1.toFixed(1)},${y1.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)} ${x3.toFixed(1)},${y3.toFixed(1)}`);
  }
}

async function loadDKTMastery(studentId) {
  try {
    const data = await cachedFetch(`/api/dkt/mastery/${encodeURIComponent(studentId)}`);
    if (data.success && data.mastery) {
      const mastery = data.mastery;
      currentDKTAlgebra = mastery.Algebra !== undefined ? mastery.Algebra : 0.6;
      currentDKTCalculus = mastery.Calculus !== undefined ? mastery.Calculus : 0.55;
      currentDKTMechanics = mastery.Mechanics !== undefined ? mastery.Mechanics : 0.58;
      
      redrawDKTRadar(currentDKTAlgebra, currentDKTCalculus, currentDKTMechanics);
    }
  } catch (e) {
    console.error("Failed to load DKT mastery data:", e);
  }
}

async function submitQuizResponse() {
  const studentId = (document.getElementById('student_id').value || '').trim() || 'default_student';
  const skillSelect = document.getElementById('quiz-skill-select');
  const outcomeSelect = document.getElementById('quiz-outcome-select');
  if (!skillSelect || !outcomeSelect) return;
  
  const skillId = skillSelect.value;
  const isCorrect = parseInt(outcomeSelect.value);
  
  // Optimistic UI Update
  const shift = isCorrect ? 0.08 : -0.05;
  if (skillId === 'Algebra') currentDKTAlgebra = Math.max(0.1, Math.min(1.0, currentDKTAlgebra + shift));
  else if (skillId === 'Calculus') currentDKTCalculus = Math.max(0.1, Math.min(1.0, currentDKTCalculus + shift));
  else if (skillId === 'Mechanics') currentDKTMechanics = Math.max(0.1, Math.min(1.0, currentDKTMechanics + shift));
  
  redrawDKTRadar(currentDKTAlgebra, currentDKTCalculus, currentDKTMechanics);
  
  // Immediate visual tactile feedback: pulse glow on button
  const logBtn = document.querySelector('button[onclick="submitQuizResponse()"]');
  if (logBtn) {
    logBtn.classList.add('optimistic-success');
    setTimeout(() => logBtn.classList.remove('optimistic-success'), 800);
  }
  
  try {
    const res = await fetch('/api/dkt/log-quiz', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        student_id: studentId,
        skill_id: skillId,
        is_correct: isCorrect,
        week: 4
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Logged quiz response. Updated DKT mastery!`);
      // Update cache manually with the new mastery values
      const cacheKey = `/api/dkt/mastery/${encodeURIComponent(studentId)}`;
      requestCache.set(cacheKey, { success: true, mastery: data.mastery });
      
      if (data.mastery) {
        currentDKTAlgebra = data.mastery.Algebra !== undefined ? data.mastery.Algebra : currentDKTAlgebra;
        currentDKTCalculus = data.mastery.Calculus !== undefined ? data.mastery.Calculus : currentDKTCalculus;
        currentDKTMechanics = data.mastery.Mechanics !== undefined ? data.mastery.Mechanics : currentDKTMechanics;
        redrawDKTRadar(currentDKTAlgebra, currentDKTCalculus, currentDKTMechanics);
      }
    } else {
      showToast('Failed to log quiz response: ' + data.error);
    }
  } catch (e) {
    showToast('Could not connect to quiz logger API.');
  }
}

// ── DEBOUNCE UTILITY ──────────────────────────────────
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

// ── LIVE VALIDATION ───────────────────────────────────
function setupLiveValidation() {
  const inputs = document.querySelectorAll('input[type="number"], .table-input');
  inputs.forEach(input => {
    input.addEventListener('input', debounce(() => {
      validateField(input);
    }, 250));
  });
}

function validateField(input) {
  const val = parseFloat(input.value);
  let isValid = true;
  
  if (input.id.startsWith('study_hours') || input.id === 'add-study') {
    if (isNaN(val) || val < 0 || val > 24) isValid = false;
  } else if (input.id.startsWith('sleep_hours') || input.id === 'add-sleep') {
    if (isNaN(val) || val < 0 || val > 24) isValid = false;
  } else if (input.id.startsWith('assignments_completed') || input.id === 'add-assignments') {
    if (isNaN(val) || val < 0 || val > 10) isValid = false;
  } else if (input.id.startsWith('mock_exams') || input.id === 'add-mock' || input.id === 'add-score') {
    if (isNaN(val) || val < 0 || val > 100) isValid = false;
  } else if (input.id.startsWith('lms_logins') || input.id === 'add-lms') {
    if (isNaN(val) || val < 0) isValid = false;
  }
  
  if (!isValid) {
    input.classList.add('invalid-input');
  } else {
    input.classList.remove('invalid-input');
  }
}

// ── GLOBAL RIPPLE EFFECT ──────────────────────────────
document.body.addEventListener('click', (e) => {
  const target = e.target.closest('button, .btn, .nav-btn, .predict-btn, .add-btn, .refresh-btn, .delete-btn, .rollback-btn');
  if (!target) return;
  
  const ripple = document.createElement('span');
  ripple.classList.add('btn-ripple');
  
  const rect = target.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  const x = e.clientX - rect.left - size / 2;
  const y = e.clientY - rect.top - size / 2;
  
  ripple.style.width = ripple.style.height = `${size}px`;
  ripple.style.left = `${x}px`;
  ripple.style.top = `${y}px`;
  
  target.appendChild(ripple);
  
  setTimeout(() => {
    ripple.remove();
  }, 600);
});

// ── ACCESSIBILITY & RESOURCE CONSERVATION ──────────────
document.addEventListener('visibilitychange', () => {
  const orbs = document.querySelectorAll('.orb');
  const playState = document.hidden ? 'paused' : 'running';
  orbs.forEach(orb => {
    orb.style.animationPlayState = playState;
  });
});

const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
function handleReducedMotion(e) {
  const orbs = document.querySelectorAll('.orb');
  if (e.matches) {
    orbs.forEach(orb => {
      orb.style.animation = 'none';
    });
  } else {
    document.querySelectorAll('.orb-1').forEach(o => o.style.animation = 'float 20s ease-in-out infinite');
    document.querySelectorAll('.orb-2').forEach(o => o.style.animation = 'float 20s ease-in-out infinite -8s');
    document.querySelectorAll('.orb-3').forEach(o => o.style.animation = 'float 20s ease-in-out infinite -14s');
  }
}
mediaQuery.addEventListener('change', handleReducedMotion);

