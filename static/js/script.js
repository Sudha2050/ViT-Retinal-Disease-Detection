/* ══════════════════════════════════════════════════
   KEY INSIGHT: Frontend class .key must EXACTLY match
   what backend Predictor.classes[] contains.

   Backend OCT:  ['CNV','DME','DRUSEN','NORMAL']
   Backend DR:   ['No DR','Mild','Moderate','Severe','Proliferative DR']

   The .key field below must equal these strings exactly.
══════════════════════════════════════════════════ */
const DR_CLASSES = [
    { key: 'No DR', name: 'No Diabetic Retinopathy', short: 'No DR', icon: '✅', badge: 'grade0', warn: 'ok', msg: 'No signs of diabetic retinopathy. Routine annual screening recommended.' },
    { key: 'Mild', name: 'Mild DR', short: 'Mild', icon: '🟡', badge: 'grade1', warn: 'caution', msg: 'Mild DR detected. Monitoring every 6–12 months advised.' },
    { key: 'Moderate', name: 'Moderate DR', short: 'Moderate', icon: '🟠', badge: 'grade2', warn: 'caution', msg: 'Moderate DR. Refer to ophthalmologist within 3–6 months.' },
    { key: 'Severe', name: 'Severe DR', short: 'Severe', icon: '🔴', badge: 'grade3', warn: 'urgent', msg: 'Severe DR. Urgent referral to retinal specialist recommended.' },
    { key: 'Proliferative DR', name: 'Proliferative DR (PDR)', short: 'PDR', icon: '🚨', badge: 'grade4', warn: 'urgent', msg: 'Proliferative DR — highest severity. Immediate specialist consultation required.' }
];
const OCT_CLASSES = [
    { key: 'CNV', name: 'Choroidal Neovascularisation (CNV)', short: 'CNV', icon: '🚨', badge: 'oct-cnv', warn: 'urgent', msg: 'CNV detected — active wet AMD. Urgent ophthalmology referral required.' },
    { key: 'DME', name: 'Diabetic Macular Edema (DME)', short: 'DME', icon: '🔴', badge: 'oct-dme', warn: 'urgent', msg: 'DME detected. Prompt evaluation needed to preserve central vision.' },
    { key: 'DRUSEN', name: 'Drusen (Dry AMD)', short: 'Drusen', icon: '🟣', badge: 'oct-drusen', warn: 'caution', msg: 'Drusen deposits — dry AMD. Monitor for progression to wet AMD.' },
    { key: 'NORMAL', name: 'Normal Retina', short: 'Normal', icon: '✅', badge: 'oct-normal', warn: 'ok', msg: 'No retinal pathology on OCT. Routine follow-up as clinically indicated.' }
];

const API_BASE = '';
let currentMode = 'dr';
let currentFile = null;
let sessionHistory = [];
let histCount = 0;

/* Health check */
async function checkServer() {
    try {
        const r = await fetch(API_BASE + '/health');
        if (r.ok) {
            const d = await r.json();
            document.getElementById('serverStatus').textContent = `Model ready · ${d.device || 'cpu'}`;
            document.getElementById('statusDot').style.background = 'var(--teal)';
        } else throw new Error();
    } catch {
        document.getElementById('serverStatus').textContent = 'Server offline';
        document.getElementById('statusDot').style.background = 'var(--red)';
    }
}
checkServer();
setInterval(checkServer, 30000);

/* Mode */
function setMode(m) {
    currentMode = m;
    document.getElementById('modeCardDR').classList.toggle('active', m === 'dr');
    document.getElementById('modeCardOCT').classList.toggle('active', m === 'oct');
    document.getElementById('uploadZone').className = 'upload-zone' + (m === 'oct' ? ' oct-mode' : '');
    document.getElementById('uploadCardSub').textContent = m === 'dr' ? 'Fundus photo · JPG / PNG' : 'OCT scan · JPG / PNG';
    document.getElementById('analyseBtn').className = 'btn ' + (m === 'dr' ? 'btn-teal' : 'btn-amber');
    document.getElementById('spinner').className = 'spinner' + (m === 'oct' ? ' oct' : '');
    resetResult();
}

/* File handling */
function handleDrag(e) { e.preventDefault(); document.getElementById('uploadZone').classList.add('drag'); }
function handleDragLeave() { document.getElementById('uploadZone').classList.remove('drag'); }
function handleDrop(e) {
    e.preventDefault(); document.getElementById('uploadZone').classList.remove('drag');
    const f = e.dataTransfer.files[0]; if (f && f.type.startsWith('image/')) handleFile(f);
}
function handleFile(file) {
    if (!file) return; currentFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById('uploadZone').style.display = 'none';
        document.getElementById('previewWrap').classList.add('show');
        document.getElementById('previewImg').src = e.target.result;
        document.getElementById('previewFilename').textContent = file.name + ' · ' + (file.size / 1024).toFixed(0) + ' KB';
        document.getElementById('analyseBtn').disabled = false;
    };
    reader.readAsDataURL(file);
}
function resetAll() {
    currentFile = null;
    document.getElementById('uploadZone').style.display = '';
    document.getElementById('previewWrap').classList.remove('show');
    document.getElementById('fileInput').value = '';
    document.getElementById('analyseBtn').disabled = true;
    resetResult();
}
function resetResult() {
    document.getElementById('resultEmpty').style.display = '';
    document.getElementById('errorBox').style.display = 'none';
    document.getElementById('diagnosisResult').classList.remove('show');
    document.getElementById('resultCardSub').textContent = 'Awaiting image';
}

/* Loading */
const STEPS = ['Uploading to server...', 'Applying CLAHE...', 'Tokenising patches...', 'Transformer inference...', 'Softmax & mapping...', 'Done!'];
let si = 0, st;
function showLoading() {
    si = 0; document.getElementById('loadingStep').textContent = STEPS[0];
    document.getElementById('loadingOverlay').classList.add('show');
    st = setInterval(() => { si = Math.min(si + 1, STEPS.length - 1); document.getElementById('loadingStep').textContent = STEPS[si]; }, 500);
}
function hideLoading() { clearInterval(st); document.getElementById('loadingOverlay').classList.remove('show'); }

/* ── MAIN API CALL ── */
async function runAnalysis() {
    if (!currentFile) return;
    showLoading();
    try {
        const fd = new FormData();
        fd.append('file', currentFile);
        fd.append('modality', currentMode);

        const resp = await fetch(API_BASE + '/predict', { method: 'POST', body: fd });
        const data = await resp.json();
        hideLoading();

        console.log('API response:', JSON.stringify(data, null, 2));

        // ── Handle image rejection from backend validation ────────────
        if (data.rejected === true || data.error === 'invalid_image') {
            showRejection(data.message || 'Image did not pass retinal validation.');
            return;
        }

        if (!resp.ok || data.error) {
            showError(data.error || `HTTP ${resp.status}`); return;
        }
        renderResult(data);
    } catch (err) {
        hideLoading();
        showError(`Cannot reach server.\n${err.message}\n\nRun: uvicorn main:app --reload`);
    }
}

/* ── REJECTION UI — shown when backend rejects a non-retinal image ── */
function showRejection(reason) {
    const modeLabel = currentMode === 'dr'
        ? 'colour fundus photograph (reddish circular retinal image with dark border)'
        : 'grayscale OCT cross-sectional scan (dark image with bright horizontal retinal layers)';

    document.getElementById('resultEmpty').style.display = 'none';
    document.getElementById('diagnosisResult').classList.remove('show');
    const eb = document.getElementById('errorBox');
    eb.style.display = 'block';
    eb.innerHTML = `
<div style="padding:22px;background:#fdecea;border:1px solid rgba(192,57,43,.25);border-radius:12px;text-align:center">
    <div style="font-size:44px;margin-bottom:14px">🚫</div>
    <div style="font-size:16px;font-weight:700;color:#7a1a0f;margin-bottom:10px;font-family:var(--sans)">
        Not a retinal image
    </div>
    <div style="font-size:13px;color:#c0392b;line-height:1.65;margin-bottom:18px;max-width:340px;margin-left:auto;margin-right:auto">
        ${reason}
    </div>
    <div style="font-size:12px;color:#7a1a0f;background:rgba(192,57,43,.07);padding:14px 16px;border-radius:8px;text-align:left;line-height:1.7">
        <strong>Required image type for ${currentMode.toUpperCase()} mode:</strong><br>
        ${modeLabel}
    </div>
    <button onclick="resetAll()" style="margin-top:16px;padding:10px 24px;background:var(--red);color:#fff;border:none;border-radius:8px;font-family:var(--sans);font-size:13px;font-weight:600;cursor:pointer">
        Upload correct image
    </button>
</div>`;
    document.getElementById('resultCardSub').textContent = 'Image rejected';
}

/* ── RENDER ──
   data = { class:"DME", confidence:0.94, probabilities:{CNV:0.02,DME:0.94,DRUSEN:0.01,NORMAL:0.03} }
   We match data.class against class.key (exact string match).
*/
function renderResult(data) {
    const classes = currentMode === 'dr' ? DR_CLASSES : OCT_CLASSES;
    const isOCT = currentMode === 'oct';

    // Find by exact key match
    const idx = classes.findIndex(c => c.key === data.class);
    if (idx === -1) {
        showError(`Backend returned unknown class: "${data.class}"\nExpected: ${classes.map(c => c.key).join(', ')}`);
        return;
    }
    const cls = classes[idx];
    const conf = data.confidence;

    // Map probs: for each frontend class, look up data.probabilities[class.key]
    const probs = classes.map(c => {
        const v = data.probabilities[c.key];
        return (v !== undefined && !isNaN(v)) ? v : 0;
    });
    const total = probs.reduce((a, b) => a + b, 0);
    if (total < 0.001) {
        showError(
            `All probabilities mapped to 0.\n` +
            `Backend keys: ${Object.keys(data.probabilities).join(', ')}\n` +
            `Frontend keys: ${classes.map(c => c.key).join(', ')}\n\n` +
            `These must match exactly. Check backend Predictor.classes.`
        ); return;
    }
    const norm = probs.map(p => p / total);

    // Update UI
    document.getElementById('resultEmpty').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
    const dr = document.getElementById('diagnosisResult');
    dr.classList.add('show', 'animate-in');

    document.getElementById('dxBadge').className = 'dx-badge ' + cls.badge;
    document.getElementById('dxIcon').textContent = cls.icon;
    document.getElementById('dxLabel').textContent = isOCT ? 'OCT Classification' : 'DR Grade';
    document.getElementById('dxName').textContent = cls.name;
    document.getElementById('dxConf').innerHTML = `Confidence: <span>${(conf * 100).toFixed(1)}%</span>`;
    document.getElementById('resultCardSub').textContent = cls.short + ' · ' + (conf * 100).toFixed(1) + '%';

    const pb = document.getElementById('probBars'); pb.innerHTML = '';
    classes.forEach((c, i) => {
        const top = i === idx;
        const fc = top ? (isOCT ? 'oct-top' : 'top') : 'low';
        const vc = top ? (isOCT ? 'oct-top' : 'top') : '';
        const d = document.createElement('div'); d.className = 'prob-row';
        d.innerHTML = `<div class="prob-label">${c.short}</div>
    <div class="prob-track"><div class="prob-fill ${fc}" style="width:0" data-w="${norm[i] * 100}"></div></div>
    <div class="prob-val ${vc}">${(norm[i] * 100).toFixed(1)}%</div>`;
        pb.appendChild(d);
    });
    setTimeout(() => { pb.querySelectorAll('.prob-fill').forEach(b => b.style.width = b.getAttribute('data-w') + '%'); }, 80);

    const wi = { ok: 'ℹ', caution: '⚠', urgent: '🚨' }[cls.warn];
    document.getElementById('warnBox').innerHTML = `<div class="warn-box ${cls.warn}"><div class="warn-icon">${wi}</div><div>${cls.msg}</div></div>`;

    addToHistory(idx, conf, cls);
}

function showError(msg) {
    document.getElementById('resultEmpty').style.display = 'none';
    document.getElementById('diagnosisResult').classList.remove('show');
    const eb = document.getElementById('errorBox'); eb.style.display = 'block';
    eb.innerHTML = `<div class="api-error"><strong>⛔ Error</strong>${msg.replace(/\n/g, '<br>')}</div>`;
    document.getElementById('resultCardSub').textContent = 'Error';
}

/* History */
function addToHistory(idx, conf, cls) {
    histCount++;
    sessionHistory.unshift({ num: histCount, mode: currentMode, file: currentFile ? currentFile.name : '—', name: cls.short, conf, time: new Date() });
    renderHistory();
}
function renderHistory() {
    const tb = document.getElementById('historyBody');
    document.getElementById('historyCount').textContent = sessionHistory.length + (sessionHistory.length === 1 ? ' analysis' : ' analyses');
    if (!sessionHistory.length) { tb.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--ink3);padding:32px;font-family:var(--mono);font-size:12px">No analyses yet</td></tr>'; return; }
    tb.innerHTML = sessionHistory.map(h => {
        const cc = h.conf > .8 ? 'high' : h.conf > .5 ? 'mid' : 'low';
        const t = h.time, ts = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}:${String(t.getSeconds()).padStart(2, '0')}`;
        return `<tr><td style="font-family:var(--mono);font-size:11px;color:var(--ink3)">${h.num}</td>
    <td><span class="hist-mode ${h.mode}">${h.mode.toUpperCase()}</span></td>
    <td style="font-family:var(--mono);font-size:11px;color:var(--ink3);max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${h.file}</td>
    <td style="font-weight:600;color:var(--ink)">${h.name}</td>
    <td><span class="hist-conf ${cc}">${(h.conf * 100).toFixed(1)}%</span></td>
    <td style="font-family:var(--mono);font-size:11px;color:var(--ink3)">${ts}</td></tr>`;
    }).join('');
}
function clearHistory() { sessionHistory = []; histCount = 0; renderHistory(); }

/* Perf bars */
const po = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) e.target.querySelectorAll('.perf-fill[data-w]').forEach(b => b.style.width = b.getAttribute('data-w') + '%'); });
}, { threshold: .3 });
document.querySelectorAll('.info-card').forEach(c => po.observe(c));
renderHistory();
