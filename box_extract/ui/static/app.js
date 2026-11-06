// The "Работа" tab: start the program in one of its modes, watch the camera and the log, stop it.
const EXTRACT_STEPS = [['измерение', /scan|survey|rim-edge/i], ['план', /plan found|planned with/], ['симуляция', /preflight passed/], ['раскладка', /phase open/], ['подход', /phase approach/], ['вход', /phase enter/],
  ['захват', /phase grasp/], ['подъём', /phase extract_vertical/], ['перенос', /phase transfer/], ['постановка', /phase lower_to_support|phase release/], ['отвод', /phase retract/], ['парковка', /--- ready:|torque off/], ['проверка', /placement confirmed/]];
let sceneData = null, lastRunning = false;

document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === b));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === b.dataset.tab));
  if (b.dataset.tab === 'learn' && !window._learnStarted) { window._learnStarted = true; window.initLearn(); }
}));

const cam = document.getElementById('cam'), overlay = document.getElementById('overlay');
function refreshCamera() { if (document.getElementById('work').classList.contains('active')) cam.src = '/snapshot.jpg?t=' + Date.now(); }
setInterval(refreshCamera, 1000); refreshCamera();
cam.addEventListener('load', drawOverlay);

function drawOverlay() {
  const r = cam.getBoundingClientRect(); overlay.width = r.width; overlay.height = r.height;
  const g = overlay.getContext('2d'), k = r.width / 1280; g.clearRect(0, 0, r.width, r.height);
  if (!sceneData) return;
  g.lineWidth = 2;
  if (sceneData.opening) { g.strokeStyle = '#22c55e'; g.beginPath(); sceneData.opening.forEach((p, i) => i ? g.lineTo(p[0] * k, p[1] * k) : g.moveTo(p[0] * k, p[1] * k)); g.closePath(); g.stroke(); }
  sceneData.cubes.forEach(c => {
    g.strokeStyle = c.inside ? '#f59e0b' : '#5b8def'; g.fillStyle = g.strokeStyle;
    (c.polygon || []).forEach(poly => { g.beginPath(); poly.forEach((p, i) => i ? g.lineTo(p[0] * k, p[1] * k) : g.moveTo(p[0] * k, p[1] * k)); g.closePath(); g.stroke(); });
    g.font = '13px sans-serif'; g.fillText(c.inside ? 'в коробке' : 'на столе', c.px[0] * k - 24, c.px[1] * k - 26);
  });
}

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

document.getElementById('scene-btn').addEventListener('click', async e => {
  const b = e.target; b.disabled = true; b.textContent = 'Измеряю…';
  try { sceneData = await api('/api/scene'); document.getElementById('n-out').textContent = sceneData.outside; document.getElementById('n-in').textContent = sceneData.inside; drawOverlay(); }
  catch (err) { document.getElementById('scene-hint').textContent = String(err.message || err); }
  b.disabled = false; b.textContent = 'Обновить сцену';
});

async function start(mode) {
  const dry = document.getElementById('dry').checked, attended = document.getElementById('attended').checked;
  try { await api('/api/run', { mode, dry, attended }); } catch (err) { alert(err.message || err); }
  poll();
}
document.getElementById('stow').addEventListener('click', () => start('stow'));
document.getElementById('extract').addEventListener('click', () => start('extract'));
document.getElementById('stop').addEventListener('click', async () => { await api('/api/stop', {}); poll(); });
const attended = document.getElementById('attended'), dry = document.getElementById('dry');
[attended, dry].forEach(x => x.addEventListener('change', gate));
function gate() { const ok = attended.checked || dry.checked; ['stow', 'extract'].forEach(id => document.getElementById(id).disabled = !ok || lastRunning); }

function chips(mode, log) {
  const box = document.getElementById('chips');
  if (mode !== 'extract' && mode !== 'stow') { box.innerHTML = ''; return; }
  let last = -1; const text = log.join('\n');
  EXTRACT_STEPS.forEach(([, re], i) => { if (re.test(text)) last = i; });
  box.innerHTML = EXTRACT_STEPS.map(([name], i) => `<span class="chip ${i < last ? 'done' : i === last ? 'now' : ''}">${name}</span>`).join('');
}

async function poll() {
  let s; try { s = await api('/api/status'); } catch (e) { return; }
  lastRunning = s.running; gate();
  const dot = document.getElementById('state-dot'), text = document.getElementById('state-text'), det = document.getElementById('state-detail');
  const modeRu = { stow: 'складываю кубики в коробку', extract: 'вытаскиваю кубики из коробки' }[s.mode] || '';
  if (s.running) { dot.className = 'dot run'; text.textContent = (s.dry ? 'Проверка: ' : 'Работает: ') + modeRu; }
  else if (s.mode) { const ok = s.exit_code === 0; dot.className = 'dot ' + (ok ? 'ok' : 'bad'); text.textContent = (ok ? 'Готово: ' : 'Остановлено: ') + modeRu; }
  else { dot.className = 'dot'; text.textContent = 'Ожидание'; }
  det.textContent = [s.project_detail, s.phase].filter(Boolean).join(' · ');
  chips(s.mode, s.log);
  const pre = document.getElementById('log'), atBottom = pre.scrollTop + pre.clientHeight > pre.scrollHeight - 30;
  if (s.log.length) { pre.textContent = s.log.join('\n'); if (atBottom) pre.scrollTop = pre.scrollHeight; }
}
setInterval(poll, 1500); poll(); gate();
window.addEventListener('resize', drawOverlay);
