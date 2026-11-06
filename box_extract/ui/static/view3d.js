// 3D scene with the real SO-101 model: the STL parts of the URDF (the same files the LeLab viewer loads), placed by the
// robot's own forward kinematics. World = the robot's base frame: x forward, y left, z up, metres. Needs three.js.
const URDF_CACHE = { data: null, geometry: {}, promise: null };
const ARM_LINKS = ['base_link', 'shoulder_link', 'upper_arm_link', 'lower_arm_link', 'wrist_link', 'gripper_link', 'moving_jaw_so101_v1_link'];
const FRAME_NAMES = { base_link: 'база', shoulder_link: 'плечо', upper_arm_link: 'верх. звено', lower_arm_link: 'локоть', wrist_link: 'запястье', gripper_link: 'кисть', gripper_frame_link: 'кончик (TCP)', moving_jaw_so101_v1_link: 'подвижная губка' };
const JOINT_OF_LINK = { shoulder_link: 'shoulder_pan', upper_arm_link: 'shoulder_lift', lower_arm_link: 'elbow_flex', wrist_link: 'wrist_flex', gripper_link: 'wrist_roll', moving_jaw_so101_v1_link: 'gripper' };

function matFrom(T) {            // row-major 4x4 (lists) -> THREE.Matrix4
  const m = new THREE.Matrix4(); m.set(T[0][0], T[0][1], T[0][2], T[0][3], T[1][0], T[1][1], T[1][2], T[1][3], T[2][0], T[2][1], T[2][2], T[2][3], 0, 0, 0, 1); return m;
}
function rpyMatrix(xyz, rpy) {   // URDF origin: R = Rz(yaw) Ry(pitch) Rx(roll)
  const R = new THREE.Matrix4().makeRotationZ(rpy[2]).multiply(new THREE.Matrix4().makeRotationY(rpy[1])).multiply(new THREE.Matrix4().makeRotationX(rpy[0]));
  R.setPosition(xyz[0], xyz[1], xyz[2]); return R;
}
function textSprite(text, color = '#ffffff', size = 0.035) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 128; const g = c.getContext('2d');
  g.font = 'bold 64px -apple-system, Helvetica, sans-serif'; g.textBaseline = 'middle'; g.fillStyle = 'rgba(11,15,22,.75)';
  const w = Math.min(500, g.measureText(text).width + 24); g.fillRect(4, 24, w, 80); g.fillStyle = color; g.fillText(text, 16, 66);
  const tex = new THREE.CanvasTexture(c); tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(size * 4, size, 1); sp.center.set(0, 0.5); sp.renderOrder = 10; return sp;
}

class Scene3D {
  constructor(canvas, opt = {}) {
    this.canvas = canvas; this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(0x0b0f16);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 10); this.camera.up.set(0, 0, 1);
    const p = opt.cam || [0.55, -0.7, 0.5]; this.camera.position.set(p[0], p[1], p[2]);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true }); this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.controls = new THREE.OrbitControls(this.camera, canvas); const t = opt.target || [0.12, -0.05, 0.1]; this.controls.target.set(t[0], t[1], t[2]);
    this.controls.enableDamping = false; this.controls.addEventListener('change', () => this.render());
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75)); const d = new THREE.DirectionalLight(0xffffff, 0.8); d.position.set(0.5, -0.6, 1); this.scene.add(d);
    const d2 = new THREE.DirectionalLight(0x8899ff, 0.35); d2.position.set(-0.5, 0.6, 0.4); this.scene.add(d2);
    this.links = {}; this.frames = {}; this.dynamic = new THREE.Group(); this.scene.add(this.dynamic);
    if (opt.table !== false) this.addTable(opt.table || 0.5);
    new ResizeObserver(() => this.resize()).observe(canvas); this.resize();
  }
  addTable(size) {
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(size * 2, size * 2), new THREE.MeshBasicMaterial({ color: 0x1a2230, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
    plane.position.z = -0.0005; this.scene.add(plane);
    const grid = new THREE.GridHelper(size * 2, size * 20, 0x3a475c, 0x263041); grid.rotation.x = Math.PI / 2; this.scene.add(grid);
  }
  resize() { const r = this.canvas.getBoundingClientRect(); if (r.width < 4) return; this.renderer.setSize(r.width, r.height, false); this.camera.aspect = r.width / r.height; this.camera.updateProjectionMatrix(); this.render(); }
  render() { this.renderer.render(this.scene, this.camera); }
  static async loadUrdf() {
    if (URDF_CACHE.promise) return URDF_CACHE.promise;
    URDF_CACHE.promise = (async () => {
      URDF_CACHE.data = await (await fetch('/api/urdf')).json();
      const loader = new THREE.STLLoader(), names = new Set();
      Object.values(URDF_CACHE.data.links).forEach(ps => ps.forEach(p => names.add(p.mesh)));
      await Promise.all([...names].map(n => new Promise(res => loader.load('/mesh/' + n, g => { g.computeVertexNormals(); URDF_CACHE.geometry[n] = g; res(); }, undefined, () => res()))));
      return URDF_CACHE.data;
    })();
    return URDF_CACHE.promise;
  }
  async addRobot(opt = {}) {
    const urdf = await Scene3D.loadUrdf(); this.robotOpacity = opt.opacity ?? 1;
    this.robot = new THREE.Group(); this.scene.add(this.robot);
    for (const [link, pieces] of Object.entries(urdf.links)) {
      const g = new THREE.Group(); g.matrixAutoUpdate = false; this.links[link] = g; this.robot.add(g);
      for (const p of pieces) {
        const geo = URDF_CACHE.geometry[p.mesh]; if (!geo) continue;
        const col = new THREE.Color(p.rgba[0], p.rgba[1], p.rgba[2]);
        const mat = new THREE.MeshStandardMaterial({ color: col, metalness: 0.15, roughness: 0.55, transparent: this.robotOpacity < 1, opacity: this.robotOpacity });
        const mesh = new THREE.Mesh(geo, mat); mesh.matrixAutoUpdate = false; mesh.matrix.copy(rpyMatrix(p.xyz, p.rpy)); mesh.userData.base = col.clone(); g.add(mesh);
      }
    }
    return this;
  }
  setPose(frames) {            // frames: {link: 4x4}
    this.lastFrames = frames;
    for (const [link, g] of Object.entries(this.links)) if (frames[link]) { g.matrix.copy(matFrom(frames[link])); g.matrixWorldNeedsUpdate = true; }
    this.updateFrameHelpers(); this.render();
  }
  tint(link, hex) {            // colour one link (e.g. by clearance); null restores the model's own colours
    const g = this.links[link]; if (!g) return;
    g.children.forEach(m => m.material.color.set(hex === null ? m.userData.base : hex));
  }
  frameHelper(key, size = 0.05, label = '', opt = {}) {
    if (this.frames[key]) { this.frames[key].size = size; return this.frames[key]; }
    const grp = new THREE.Group(); grp.matrixAutoUpdate = false;
    [[0xff5d5d, [1, 0, 0]], [0x4fd16b, [0, 1, 0]], [0x5b9dff, [0, 0, 1]]].forEach(([c, d]) => {
      const a = new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(), size, c, size * 0.28, size * 0.16); a.line.material.linewidth = 2; a.userData.axis = true; grp.add(a);
    });
    if (label) { const s = textSprite(label, opt.color || '#ffffff', 0.026); s.position.set(size * 0.1, size * 0.1, size * 0.1); grp.add(s); }
    this.scene.add(grp); this.frames[key] = { grp, size, key }; return this.frames[key];
  }
  setFrame(key, T) { const f = this.frames[key]; if (!f) return; f.T = T; f.grp.matrix.copy(matFrom(T)); f.grp.matrixWorldNeedsUpdate = true; this.render(); }
  updateFrameHelpers() {       // frames tied to robot links follow the last pose
    for (const f of Object.values(this.frames)) if (f.link && this.lastFrames && this.lastFrames[f.link]) { f.grp.matrix.copy(matFrom(this.lastFrames[f.link])); f.grp.matrixWorldNeedsUpdate = true; }
  }
  linkFrame(link, size = 0.05, label = '') { const f = this.frameHelper('link:' + link, size, label); f.link = link; return f; }
  showFrame(key, on) { const f = this.frames[key]; if (f) { f.grp.visible = on; this.render(); } }
  clearDynamic() { this.dynamic.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); } }); this.scene.remove(this.dynamic); this.dynamic = new THREE.Group(); this.scene.add(this.dynamic); }
  line(points, color = 0xffffff, opacity = 1) {
    const g = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p)));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity })); this.dynamic.add(l); return l;
  }
  tube(points, color = 0xffffff, radius = 0.0016) {
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)));
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, points.length * 2), radius, 6, false), new THREE.MeshBasicMaterial({ color })); this.dynamic.add(m); return m;
  }
  dot(p, color = 0xffffff, r = 0.006) { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({ color })); m.position.set(...p); this.dynamic.add(m); return m; }
  arrow(from, to, color = 0xffffff, head = 0.012) {
    const d = new THREE.Vector3(to[0] - from[0], to[1] - from[1], to[2] - from[2]), L = d.length(); if (L < 1e-6) return null;
    const a = new THREE.ArrowHelper(d.normalize(), new THREE.Vector3(...from), L, color, Math.min(head, L * 0.5), head * 0.6); this.dynamic.add(a); return a;
  }
  label(p, text, color = '#fff', size = 0.03) { const s = textSprite(text, color, size); s.position.set(...p); this.dynamic.add(s); return s; }
  cubeBox(vertices, color = 0xfacc15, fill = 0.25) {
    const v = vertices.map(p => new THREE.Vector3(...p)); const idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    this.line(idx.flatMap(([a, b]) => [[...v[a].toArray()], [...v[b].toArray()]]).reduce((acc, p, i, arr) => acc, []), color); // placeholder to keep API simple
    idx.forEach(([a, b]) => this.line([v[a].toArray(), v[b].toArray()], color));
    const faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], pos = [];
    faces.forEach(f => { [0, 1, 2, 0, 2, 3].forEach(i => pos.push(...v[f[i]].toArray())); });
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.computeVertexNormals();
    this.dynamic.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: fill, side: THREE.DoubleSide, depthWrite: false })));
  }
  boxFrame(center, size, yaw, height, color = 0xc7d0dc) {      // the box opening and its rim height
    const [cx, cy] = center, [sx, sy] = size, c = Math.cos(yaw), s = Math.sin(yaw);
    const P = (a, b, z) => [cx + a * c - b * s, cy + a * s + b * c, z];
    const base = [[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]];
    const lo = base.map(p => P(p[0], p[1], 0)), hi = base.map(p => P(p[0], p[1], height));
    [lo, hi].forEach(ring => this.line([...ring, ring[0]], color)); base.forEach((_, i) => this.line([lo[i], hi[i]], color));
    const g = new THREE.BufferGeometry(); const pos = []; [[0, 1], [1, 2], [2, 3], [3, 0]].forEach(([a, b]) => { [lo[a], lo[b], hi[b], lo[a], hi[b], hi[a]].forEach(p => pos.push(...p)); });
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); this.dynamic.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false })));
  }
  jointAxis(frameKey, T, angleLabel, color = 0xe879f9, len = 0.07) {   // the axis a joint turns about = z of its frame, with the sense of positive turn
    const o = [T[0][3], T[1][3], T[2][3]], z = [T[0][2], T[1][2], T[2][2]];
    this.arrow([o[0] - z[0] * len / 2, o[1] - z[1] * len / 2, o[2] - z[2] * len / 2], [o[0] + z[0] * len / 2, o[1] + z[1] * len / 2, o[2] + z[2] * len / 2], color, 0.014);
    if (angleLabel) this.label([o[0] + z[0] * len / 2, o[1] + z[1] * len / 2, o[2] + z[2] * len / 2 + 0.012], angleLabel, '#f0abfc', 0.024);
  }
  setCamera(pos, target) { this.camera.position.set(...pos); this.controls.target.set(...target); this.controls.update(); this.render(); }
}
window.Scene3D = Scene3D;
