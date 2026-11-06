// RRT-Connect in a 2D configuration space, step by step. The robot's real search runs the same algorithm in the
// 5-dimensional joint space (shoulder_pan ... wrist_roll); here two coordinates are enough to see how it works.
class RRTDemo {
  constructor(canvas, info) {
    this.c = canvas; this.g = canvas.getContext('2d'); this.info = info; this.step = 0.045;
    this.reset(); new ResizeObserver(() => this.resize()).observe(canvas); this.resize();
  }
  resize() { const r = this.c.getBoundingClientRect(), d = window.devicePixelRatio || 1; this.c.width = r.width * d; this.c.height = r.height * d; this.draw(); }
  reset(seed) {
    this.rand = mulberry(seed ?? (Math.random() * 1e9 | 0));
    this.obstacles = [];
    const n = 5 + (this.rand() * 3 | 0);
    for (let i = 0; i < n; i++) {
      const w = 0.08 + this.rand() * 0.16, h = 0.08 + this.rand() * 0.26;
      this.obstacles.push([0.18 + this.rand() * 0.62, 0.05 + this.rand() * 0.8, w, h]);
    }
    this.start = [0.06, 0.5]; this.goal = [0.94, 0.5];
    this.obstacles = this.obstacles.filter(o => !this.inRect(this.start, o, 0.06) && !this.inRect(this.goal, o, 0.06));
    this.A = [{ p: this.start, parent: -1 }]; this.B = [{ p: this.goal, parent: -1 }];
    this.iter = 0; this.path = null; this.smooth = null; this.running = false; this.done = false; this.samples = []; this.collisionChecks = 0;
    this.draw(); this.report();
  }
  inRect(p, o, m = 0) { return p[0] > o[0] - m && p[0] < o[0] + o[2] + m && p[1] > o[1] - m && p[1] < o[1] + o[3] + m; }
  free(p) { if (p[0] < 0 || p[0] > 1 || p[1] < 0 || p[1] > 1) return false; return !this.obstacles.some(o => this.inRect(p, o)); }
  edgeFree(a, b) { const n = Math.ceil(dist(a, b) / 0.008); for (let i = 0; i <= n; i++) { this.collisionChecks++; if (!this.free(lerp(a, b, i / n))) return false; } return true; }
  nearest(T, p) { let k = 0, bd = 1e9; T.forEach((n, i) => { const d = dist(n.p, p); if (d < bd) { bd = d; k = i; } }); return k; }
  extend(T, target) {
    const k = this.nearest(T, target), from = T[k].p, d = dist(from, target);
    const to = d <= this.step ? target : lerp(from, target, this.step / d);
    if (!this.edgeFree(from, to)) return { status: 'trapped' };
    T.push({ p: to, parent: k });
    return { status: d <= this.step ? 'reached' : 'advanced', index: T.length - 1 };
  }
  connect(T, target) { let r; do { r = this.extend(T, target); } while (r.status === 'advanced'); return r; }
  stepOnce() {
    if (this.done) return;
    this.iter++;
    const q = [this.rand(), this.rand()]; this.samples = [q];
    const r = this.extend(this.A, q);
    if (r.status !== 'trapped') {
      const c = this.connect(this.B, this.A[r.index].p);
      if (c.status === 'reached') { this.finish(r.index, c.index); return; }
    }
    [this.A, this.B] = [this.B, this.A]; this.flipped = !this.flipped;
  }
  finish(ia, ib) {
    const climb = (T, i) => { const out = []; while (i >= 0) { out.push(T[i].p); i = T[i].parent; } return out; };
    let a = climb(this.A, ia).reverse(), b = climb(this.B, ib);
    let path = a.concat(b);
    if (this.flipped) path = path.reverse();
    this.path = path; this.done = true; this.smooth = this.shortcut(path); this.draw(); this.report();
  }
  shortcut(path) {                       // keep trying to replace a stretch of the path by a straight free segment
    let p = path.slice(), rng = mulberry(7);
    for (let t = 0; t < 120 && p.length > 2; t++) {
      let i = rng() * (p.length - 2) | 0, j = i + 2 + (rng() * (p.length - i - 2) | 0);
      if (j >= p.length) j = p.length - 1;
      if (j - i >= 2 && this.edgeFree(p[i], p[j])) p = p.slice(0, i + 1).concat(p.slice(j));
    }
    return p;
  }
  run() {
    if (this.running) return; this.running = true;
    const tick = () => {
      if (!this.running) return;
      for (let k = 0; k < 2 && !this.done && this.iter < 4000; k++) this.stepOnce();
      this.draw(); this.report();
      if (this.done || this.iter >= 4000) { this.running = false; return; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  length(p) { return p ? p.slice(1).reduce((s, q, i) => s + dist(p[i], q), 0) : 0; }
  report() {
    const total = this.A.length + this.B.length;
    this.info.textContent = `итерация ${this.iter} · узлов в деревьях ${total} · проверок столкновения ${this.collisionChecks}` +
      (this.path ? ` · путь найден: ${this.path.length} точек, длина ${this.length(this.path).toFixed(3)} → после сглаживания ${this.smooth.length} точек, ${this.length(this.smooth).toFixed(3)}` : '');
  }
  draw() {
    const g = this.g, W = this.c.width, H = this.c.height, d = window.devicePixelRatio || 1, m = 8 * d;
    const X = p => m + p[0] * (W - 2 * m), Y = p => H - m - p[1] * (H - 2 * m);
    g.clearRect(0, 0, W, H); g.fillStyle = '#0b0f16'; g.fillRect(0, 0, W, H);
    g.fillStyle = 'rgba(217,45,32,.55)';
    this.obstacles.forEach(o => g.fillRect(X([o[0], 0]), Y([0, o[1] + o[3]]), o[2] * (W - 2 * m), o[3] * (H - 2 * m)));
    const tree = (T, color) => { g.strokeStyle = color; g.lineWidth = 1 * d; g.beginPath(); T.forEach(n => { if (n.parent >= 0) { g.moveTo(X(T[n.parent].p), Y(T[n.parent].p)); g.lineTo(X(n.p), Y(n.p)); } }); g.stroke(); };
    const [S, G] = this.flipped ? [this.B, this.A] : [this.A, this.B];
    tree(S, '#5b8def'); tree(G, '#f59e0b');
    if (this.path) { g.strokeStyle = '#9aa5b4'; g.lineWidth = 2 * d; g.beginPath(); this.path.forEach((p, i) => i ? g.lineTo(X(p), Y(p)) : g.moveTo(X(p), Y(p))); g.stroke(); }
    if (this.smooth) { g.strokeStyle = '#22c55e'; g.lineWidth = 3 * d; g.beginPath(); this.smooth.forEach((p, i) => i ? g.lineTo(X(p), Y(p)) : g.moveTo(X(p), Y(p))); g.stroke(); }
    this.samples.forEach(p => { g.fillStyle = '#fff'; g.beginPath(); g.arc(X(p), Y(p), 3 * d, 0, 7); g.fill(); });
    [[this.start, '#5b8def', 'старт'], [this.goal, '#f59e0b', 'цель']].forEach(([p, col, t]) => { g.fillStyle = col; g.beginPath(); g.arc(X(p), Y(p), 6 * d, 0, 7); g.fill(); g.fillStyle = '#e8ecf2'; g.font = `${12 * d}px sans-serif`; g.fillText(t, X(p) - 12 * d, Y(p) - 10 * d); });
  }
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]), lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
