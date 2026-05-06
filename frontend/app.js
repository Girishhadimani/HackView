/* ================================================================
   DeepShield — Frontend Application Logic
   Handles: file upload, drag-drop, API calls, result rendering
   ================================================================ */

const API_BASE = ['localhost', '127.0.0.1'].includes(window.location.hostname) && window.location.port !== '8000'
  ? 'http://localhost:8000'
  : window.location.origin;

initHeroScene();

async function parseApiResponse(resp) {
  const contentType = resp.headers.get('content-type') || '';
  const body = contentType.includes('application/json')
    ? await resp.json()
    : await resp.text();

  if (!resp.ok) {
    const detail = typeof body === 'object'
      ? (body.detail || body.error || JSON.stringify(body))
      : body.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    throw new Error(detail || `Request failed with HTTP ${resp.status}`);
  }

  if (typeof body !== 'object') {
    throw new Error('Backend returned HTML instead of JSON. Make sure the FastAPI server is running on port 8000.');
  }

  return body;
}

function initHeroScene() {
  const canvas = document.getElementById('heroScene');
  if (!canvas) return;

  const ctx = canvas.getContext('2d', { alpha: true });
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const pointer = { x: 0, y: 0 };
  let width = 0;
  let height = 0;
  let dpr = 1;
  let points = [];
  let rafId = 0;
  let time = 0;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const count = Math.min(130, Math.max(70, Math.floor(width / 12)));
    points = Array.from({ length: count }, (_, i) => {
      const layer = i % 5;
      return {
        x: (Math.random() - 0.5) * width * 1.3,
        y: (Math.random() - 0.5) * height * 0.95,
        z: 120 + Math.random() * 780,
        r: 0.8 + layer * 0.22,
        hue: layer % 3,
      };
    });
  }

  function project(p) {
    const depth = 720 / (720 + p.z);
    const cx = width * 0.5 + pointer.x * 24;
    const cy = height * 0.38 + pointer.y * 16;
    return {
      x: cx + p.x * depth,
      y: cy + p.y * depth,
      s: depth,
    };
  }

  function draw() {
    time += 0.008;
    ctx.clearRect(0, 0, width, height);

    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, 'rgba(0,245,255,0.16)');
    gradient.addColorStop(0.52, 'rgba(139,92,246,0.10)');
    gradient.addColorStop(1, 'rgba(34,197,94,0.08)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(width * 0.5 + pointer.x * 18, height * 0.36 + pointer.y * 12);
    ctx.rotate(time * 0.35);
    for (let ring = 0; ring < 5; ring++) {
      ctx.beginPath();
      ctx.ellipse(0, 0, 190 + ring * 72, 42 + ring * 22, ring * 0.28, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${ring % 2 ? '139,92,246' : '0,245,255'},${0.15 - ring * 0.018})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();

    for (const p of points) {
      p.z -= prefersReducedMotion ? 0.15 : 1.15;
      p.x += Math.sin(time + p.z * 0.01) * 0.18;
      if (p.z < 40) {
        p.z = 900;
        p.x = (Math.random() - 0.5) * width * 1.3;
        p.y = (Math.random() - 0.5) * height * 0.95;
      }

      const q = project(p);
      if (q.x < -60 || q.x > width + 60 || q.y < -60 || q.y > height + 60) continue;
      const alpha = Math.max(0.08, Math.min(0.75, 1 - p.z / 900));
      const color = p.hue === 0 ? '0,245,255' : p.hue === 1 ? '139,92,246' : '52,211,153';
      ctx.beginPath();
      ctx.arc(q.x, q.y, p.r + q.s * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${color},${alpha})`;
      ctx.fill();
    }

    rafId = requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  window.addEventListener('pointermove', (event) => {
    pointer.x = (event.clientX / Math.max(width, 1) - 0.5) * 2;
    pointer.y = (event.clientY / Math.max(height, 1) - 0.5) * 2;
  }, { passive: true });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) cancelAnimationFrame(rafId);
    else rafId = requestAnimationFrame(draw);
  });

  resize();
  draw();
}

// ── DOM refs ────────────────────────────────────────────────────
const uploadZone      = document.getElementById('uploadZone');
const fileInput       = document.getElementById('fileInput');
const browseBtn       = document.getElementById('browseBtn');
const urlInput        = document.getElementById('urlInput');
const urlBtn          = document.getElementById('urlBtn');
const platformSelect  = document.getElementById('platformSelect');
const analysisSection = document.getElementById('analysisSection');
const resetBtn        = document.getElementById('resetBtn');
const techToggle      = document.getElementById('techToggle');
const techDetails     = document.getElementById('techDetails');

// Preview
const previewImg   = document.getElementById('previewImg');
const previewVid   = document.getElementById('previewVid');
const previewAud   = document.getElementById('previewAud');
const previewMeta  = document.getElementById('previewMeta');
const mediaTypeBadge = document.getElementById('mediaTypeBadge');

// Verdict
const scanningOverlay = document.getElementById('scanningOverlay');
const verdictContent  = document.getElementById('verdictContent');
const verdictIcon     = document.getElementById('verdictIcon');
const verdictLabel    = document.getElementById('verdictLabel');
const verdictSub      = document.getElementById('verdictSub');
const gaugeFill       = document.getElementById('gaugeFill');
const gaugeScore      = document.getElementById('gaugeScore');
const confValue       = document.getElementById('confValue');

// Classification and authenticity
const modelBadge          = document.getElementById('modelBadge');
const classificationLabel = document.getElementById('classificationLabel');
const classificationSub   = document.getElementById('classificationSub');
const modelName           = document.getElementById('modelName');
const modelConfidence     = document.getElementById('modelConfidence');
const authScore           = document.getElementById('authScore');
const authLabel           = document.getElementById('authLabel');
const authCopy            = document.getElementById('authCopy');
const authBar             = document.getElementById('authBar');
const authFactors         = document.getElementById('authFactors');

// Social integration
const socialSource    = document.getElementById('socialSource');
const detectedPlatform= document.getElementById('detectedPlatform');
const copyReportBtn   = document.getElementById('copyReportBtn');
const shareX          = document.getElementById('shareX');
const shareLinkedIn   = document.getElementById('shareLinkedIn');
let lastShareText = 'DeepShield media authenticity report';

// Breakdown
const imgScore    = document.getElementById('imgScore');
const imgBar      = document.getElementById('imgBar');
const imgArtifacts= document.getElementById('imgArtifacts');
const vidScore    = document.getElementById('vidScore');
const vidBar      = document.getElementById('vidBar');
const vidArtifacts= document.getElementById('vidArtifacts');
const audScore    = document.getElementById('audScore');
const audBar      = document.getElementById('audBar');
const audArtifacts= document.getElementById('audArtifacts');
const metaScore   = document.getElementById('metaScore');
const metaBar     = document.getElementById('metaBar');
const metaArtifacts = document.getElementById('metaArtifacts');

// Heatmap
const heatmapCard = document.getElementById('heatmapCard');
const heatmapArea = document.getElementById('heatmapArea');

// Tech
const techGrid = document.getElementById('techGrid');

// ── Drag & Drop ─────────────────────────────────────────────────
uploadZone.addEventListener('dragover', e => {
  e.preventDefault(); uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
uploadZone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

// ── URL scan ────────────────────────────────────────────────────
urlBtn.addEventListener('click', () => {
  const url = urlInput.value.trim();
  if (!url) return;
  analyzeUrl(url);
});
urlInput.addEventListener('keydown', e => { if (e.key === 'Enter') urlBtn.click(); });
copyReportBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(lastShareText);
    copyReportBtn.textContent = 'Copied';
    setTimeout(() => { copyReportBtn.textContent = 'Copy'; }, 1200);
  } catch {
    copyReportBtn.textContent = 'Copy failed';
  }
});

// ── Reset ────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  analysisSection.style.display = 'none';
  fileInput.value = '';
  urlInput.value  = '';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ── Tech toggle ──────────────────────────────────────────────────
techToggle.addEventListener('click', () => {
  const shown = techDetails.style.display !== 'none';
  techDetails.style.display = shown ? 'none' : 'block';
  techToggle.textContent    = shown ? 'Show' : 'Hide';
});

// ── File handler ────────────────────────────────────────────────
function handleFile(file) {
  const url    = URL.createObjectURL(file);
  const type   = detectType(file.name);
  detectedPlatform.textContent = 'Local upload';
  socialSource.textContent = 'Local file';
  showPreview(url, type, file.name, file.size);
  scrollToAnalysis();
  startScanning();
  uploadAndAnalyze(file);
}

async function analyzeUrl(url) {
  const type = detectType(url);
  const platform = platformSelect.value === 'auto' ? detectSocialPlatform(url) : platformSelect.value;
  analysisSection.style.display = 'block';
  previewMeta.textContent = `${platformLabel(platform)} source · ${url}`;
  updateSocialSource(platform, url);
  mediaTypeBadge.textContent = type.toUpperCase();
  if (type === 'image') {
    previewImg.src = url; previewImg.style.display = 'block';
    previewVid.style.display = 'none'; previewAud.style.display = 'none';
  }
  scrollToAnalysis();
  startScanning();
  try {
    const resp = await fetch(`${API_BASE}/v1/analyze/url?url=${encodeURIComponent(url)}`);
    const data = await parseApiResponse(resp);
    renderResult(data);
  } catch (err) { showError(formatApiError(err)); }
}

async function uploadAndAnalyze(file) {
  const form = new FormData();
  form.append('file', file);
  try {
    const resp = await fetch(`${API_BASE}/v1/analyze`, { method: 'POST', body: form });
    const data = await parseApiResponse(resp);
    renderResult(data);
  } catch (err) { showError(formatApiError(err)); }
}

// ── Preview ──────────────────────────────────────────────────────
function showPreview(url, type, name, size) {
  analysisSection.style.display = 'block';
  mediaTypeBadge.textContent = type.toUpperCase();
  previewImg.style.display = previewVid.style.display = previewAud.style.display = 'none';
  if (type === 'image') { previewImg.src = url; previewImg.style.display = 'block'; }
  else if (type === 'video') { previewVid.src = url; previewVid.style.display = 'block'; }
  else if (type === 'audio') { previewAud.src = url; previewAud.style.display = 'block'; }
  previewMeta.textContent = `${name}  ·  ${formatBytes(size)}`;
}

function detectType(name) {
  const ext = name.split('.').pop().toLowerCase();
  if (['jpg','jpeg','png','bmp','webp','gif'].includes(ext)) return 'image';
  if (['mp4','mov','avi','mkv','webm'].includes(ext)) return 'video';
  if (['wav','mp3','flac','ogg','m4a'].includes(ext)) return 'audio';
  return 'image';
}

// ── Scanning state ───────────────────────────────────────────────
function startScanning() {
  scanningOverlay.style.display = 'flex';
  verdictContent.style.display  = 'none';
  resetModality();
  resetClassification();
  resetAuthenticity();
}

function scrollToAnalysis() {
  setTimeout(() => analysisSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

// ── Render result ────────────────────────────────────────────────
function renderResult(data) {
  scanningOverlay.style.display = 'none';
  verdictContent.style.display  = 'block';

  const score   = data.authenticity_score || 0;
  const verdict = data.verdict || 'UNCERTAIN';
  const conf    = data.confidence || 0;
  const mediaType = data.media_type || mediaTypeBadge.textContent.toLowerCase();

  // Verdict label
  const icons = { FAKE:'❌', SUSPICIOUS:'⚠️', UNCERTAIN:'🔍', AUTHENTIC:'✅' };
  verdictIcon.textContent  = icons[verdict] || '❓';
  verdictLabel.textContent = verdict;
  verdictLabel.className   = 'verdict-label ' + verdict;
  verdictSub.textContent   = verdictSubtext(verdict);
  confValue.textContent    = `${(conf * 100).toFixed(1)}%`;

  // Gauge animation
  animateGauge(score);
  renderClassification(score, verdict, conf, mediaType);
  renderAuthenticity(score, verdict, data);
  updateShareLinks(data);

  // Modality breakdown
  const img  = data.image_score || {};
  const vid  = data.video_score || {};
  const aud  = data.audio_score || {};
  const meta = data.metadata_score || 0;

  setModality(imgScore, imgBar, imgArtifacts, img.score, img.available, img.artifacts || []);
  setModality(vidScore, vidBar, vidArtifacts, vid.score, vid.available, vid.artifacts || []);
  setModality(audScore, audBar, audArtifacts, aud.score, aud.available, aud.artifacts || []);
  setModality(metaScore, metaBar, metaArtifacts, meta, true, data.exif_anomalies || []);

  // Heatmap
  if (data.heatmap_url) {
    heatmapCard.style.display = 'block';
    heatmapArea.innerHTML = `<img src="${data.heatmap_url}" alt="Grad-CAM heatmap" />`;
  }

  // Tech details
  renderTechGrid({
    'Job ID':           data.job_id || '—',
    'Media Hash':       (data.media_hash || '').slice(0, 16) + '…',
    'Media Type':       data.media_type || '—',
    'Processing':       `${data.processing_time_ms || 0} ms`,
    'Raw Score':        score.toFixed(4),
    'Confidence':       (conf * 100).toFixed(1) + '%',
    'Risk Level':       data.risk_level || '—',
    'AI Class':         classificationLabel.textContent,
    'Authenticity':     authScore.textContent,
    'Social Platform':  detectedPlatform.textContent,
    'Analyzed At':      (data.analyzed_at || '').replace('T',' ').slice(0,19) + ' UTC',
  });
}

function renderClassification(score, verdict, confidence, mediaType) {
  const fakePct = Math.round(score * 100);
  const classInfo = score >= 0.8
    ? ['AI-generated or manipulated', 'Strong synthetic or tampering signals were detected.']
    : score >= 0.6
      ? ['Possibly manipulated', 'Multiple suspicious signals need manual review.']
      : score >= 0.3
        ? ['Uncertain authenticity', 'The classifier found mixed evidence.']
        : ['Likely authentic', 'No strong AI-generation pattern was detected.'];

  classificationLabel.textContent = classInfo[0];
  classificationSub.textContent = `${classInfo[1]} Fake probability: ${fakePct}%.`;
  modelBadge.textContent = verdict;
  modelBadge.className = 'model-badge ' + verdict;
  modelName.textContent = modelNameFor(mediaType);
  modelConfidence.textContent = `${Math.round(confidence * 100)}%`;
}

function renderAuthenticity(fakeScore, verdict, data) {
  const authentic = Math.max(0, Math.min(1, 1 - fakeScore));
  const pct = Math.round(authentic * 100);
  const color = authentic >= 0.7 ? '#22c55e'
              : authentic >= 0.4 ? '#eab308'
              : '#ef4444';

  authScore.textContent = `${pct}%`;
  authScore.style.color = color;
  authBar.style.width = `${pct}%`;
  authBar.style.background = `linear-gradient(90deg, ${color}88, ${color})`;
  authLabel.textContent = verdict === 'AUTHENTIC' ? 'High authenticity'
    : verdict === 'UNCERTAIN' ? 'Mixed authenticity'
    : verdict === 'SUSPICIOUS' ? 'Low authenticity'
    : 'Very low authenticity';
  authCopy.textContent = 'Score combines image, video, audio, metadata, and consistency signals.';

  const factors = [];
  if (data.image_score?.available) factors.push('Image model');
  if (data.video_score?.available) factors.push('Video model');
  if (data.audio_score?.available) factors.push('Audio model');
  if ((data.metadata_score || 0) > 0) factors.push('Metadata anomalies');
  authFactors.innerHTML = (factors.length ? factors : ['Available model signals'])
    .map(f => `<span class="artifact-tag">${f}</span>`).join('');
}

// ── Gauge animation ──────────────────────────────────────────────
function animateGauge(score) {
  const circumference = 2 * Math.PI * 80;  // r=80 → 502.6
  const offset = circumference * (1 - score);

  // Color by score
  const color = score >= 0.8 ? '#ef4444'
              : score >= 0.6 ? '#f97316'
              : score >= 0.3 ? '#eab308'
              : '#22c55e';

  gaugeFill.style.strokeDasharray  = circumference;
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke           = color;
  gaugeScore.textContent           = `${Math.round(score * 100)}%`;
  gaugeScore.style.color           = color;
}

// ── Modality bar ─────────────────────────────────────────────────
function setModality(scoreEl, barEl, artifactsEl, score, available, artifacts) {
  if (!available || score === undefined || score === null) {
    scoreEl.textContent = 'N/A';
    scoreEl.style.color = '#8888aa';
    barEl.style.width   = '0%';
    return;
  }
  const pct   = Math.round(score * 100);
  const color = score >= 0.8 ? '#ef4444'
              : score >= 0.6 ? '#f97316'
              : score >= 0.3 ? '#eab308'
              : '#22c55e';
  scoreEl.textContent = pct + '%';
  scoreEl.style.color = color;
  barEl.style.width   = pct + '%';
  barEl.style.background = `linear-gradient(90deg, ${color}88, ${color})`;

  artifactsEl.innerHTML = artifacts.map(a =>
    `<span class="artifact-tag">${a}</span>`
  ).join('');
}

function resetModality() {
  [imgScore,vidScore,audScore,metaScore].forEach(el => {
    el.textContent = '—'; el.style.color = '';
  });
  [imgBar,vidBar,audBar,metaBar].forEach(el => el.style.width = '0%');
  [imgArtifacts,vidArtifacts,audArtifacts,metaArtifacts].forEach(el => el.innerHTML='');
  heatmapCard.style.display = 'none';
  gaugeFill.style.strokeDashoffset = '502';
  gaugeScore.textContent = '0%';
  gaugeScore.style.color = '';
}

function resetClassification() {
  modelBadge.textContent = 'SCANNING';
  modelBadge.className = 'model-badge';
  classificationLabel.textContent = 'Classifying media';
  classificationSub.textContent = 'The model is evaluating synthetic and manipulation patterns.';
  modelConfidence.textContent = '—';
}

function resetAuthenticity() {
  authScore.textContent = '0%';
  authScore.style.color = '';
  authLabel.textContent = 'Analyzing';
  authCopy.textContent = 'Combining model signals into a content authenticity score.';
  authBar.style.width = '0%';
  authFactors.innerHTML = '';
}

// ── Tech grid ────────────────────────────────────────────────────
function renderTechGrid(items) {
  techGrid.innerHTML = Object.entries(items).map(([k,v]) => `
    <div class="tech-item">
      <div class="tech-key">${k}</div>
      <div class="tech-val">${v}</div>
    </div>
  `).join('');
}

// ── Error state ──────────────────────────────────────────────────
function showError(msg) {
  scanningOverlay.style.display = 'none';
  verdictContent.style.display  = 'block';
  verdictIcon.textContent  = '⚠️';
  verdictLabel.textContent = 'ERROR';
  verdictLabel.className   = 'verdict-label SUSPICIOUS';
  verdictSub.textContent   = msg || 'Analysis failed. Please try again.';
  confValue.textContent    = '0%';
  animateGauge(0);
  resetClassification();
  resetAuthenticity();
}

function detectSocialPlatform(url) {
  const host = (() => {
    try { return new URL(url).hostname.toLowerCase(); }
    catch { return ''; }
  })();
  if (host.includes('instagram.com')) return 'instagram';
  if (host.includes('twitter.com') || host.includes('x.com')) return 'x';
  if (host.includes('facebook.com') || host.includes('fb.watch')) return 'facebook';
  if (host.includes('tiktok.com')) return 'tiktok';
  if (host.includes('youtube.com') || host.includes('youtu.be')) return 'youtube';
  if (host.includes('linkedin.com')) return 'linkedin';
  return 'web';
}

function platformLabel(platform) {
  const labels = {
    instagram: 'Instagram',
    x: 'X',
    facebook: 'Facebook',
    tiktok: 'TikTok',
    youtube: 'YouTube',
    linkedin: 'LinkedIn',
    web: 'Web',
    auto: 'Auto',
  };
  return labels[platform] || 'Web';
}

function updateSocialSource(platform, url) {
  detectedPlatform.textContent = platformLabel(platform);
  socialSource.textContent = url ? 'Linked source' : 'No source linked';
}

function updateShareLinks(data) {
  const score = Math.round((data.authenticity_score || 0) * 100);
  const authentic = 100 - score;
  const verdict = data.verdict || 'UNCERTAIN';
  lastShareText = `DeepShield result: ${verdict}. Content authenticity ${authentic}%, fake score ${score}%.`;
  shareX.href = `https://twitter.com/intent/tweet?text=${encodeURIComponent(lastShareText)}`;
  shareLinkedIn.href = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(window.location.href)}`;
}

function modelNameFor(mediaType) {
  if (mediaType === 'video') return 'DeepShield Temporal Classifier';
  if (mediaType === 'audio') return 'DeepShield Audio Classifier';
  return 'DeepShield Image Classifier';
}

function formatApiError(err) {
  const msg = err && err.message ? err.message : String(err || '');
  if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
    return 'Backend is not running. Start FastAPI on http://localhost:8000, then try again.';
  }
  return msg;
}

// ── Helpers ──────────────────────────────────────────────────────
function verdictSubtext(verdict) {
  const map = {
    FAKE:       'High probability of AI manipulation detected.',
    SUSPICIOUS: 'Anomalies detected — manual review recommended.',
    UNCERTAIN:  'Inconclusive — borderline signals found.',
    AUTHENTIC:  'No manipulation detected. Content appears genuine.',
  };
  return map[verdict] || '';
}

function formatBytes(bytes) {
  if (bytes < 1024)          return bytes + ' B';
  if (bytes < 1024 * 1024)   return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
