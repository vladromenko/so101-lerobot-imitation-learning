// Учебник. Все числа на странице считает сама рука (её модель и её последний реальный план) — ничего не нарисовано «от руки».
const $ = (s, r = document) => r.querySelector(s);
const f = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toFixed(n);
const mat = (M, n = 3) => `<span class="mat">${M.map(r => `<div style="grid-template-columns:repeat(${r.length},${n > 2 ? 58 : 50}px)">${r.map(v => `<span>${f(v, n)}</span>`).join('')}</div>`).join('')}</span>`;
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
function math(node) { if (window.renderMathInElement) renderMathInElement(node, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false }); }
const PHASE_COLORS = { open: '#9aa5b4', approach: '#5b8def', enter: '#22c55e', grasp: '#facc15', extract_vertical: '#f59e0b', transfer: '#ef4444', lower_to_support: '#d946ef', release: '#a78bfa', retract: '#14b8a6' };
const PHASE_RU = { open: 'раскрыть губки', approach: 'подход', enter: 'вход к кубику', grasp: 'захват', extract_vertical: 'подъём над кромкой', transfer: 'перенос', lower_to_support: 'опускание', release: 'отпускание', retract: 'отвод' };
const JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
const JOINT_RU = { shoulder_pan: 'поворот основания', shoulder_lift: 'плечо', elbow_flex: 'локоть', wrist_flex: 'наклон кисти', wrist_roll: 'вращение кисти', gripper: 'губки' };
const JOINT_COLORS = ['#5b8def', '#22c55e', '#f59e0b', '#ef4444', '#a78bfa', '#14b8a6'];
let ROBOT = null, PLAN = null;
const pose6 = q => Object.fromEntries(JOINTS.map((j, i) => [j, Array.isArray(q) ? q[i] : q[j]]));
const hex = s => parseInt(s.slice(1), 16);
const frames0 = q => api('/api/fk', q);

function lazy(el, fn) { const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); Promise.resolve().then(fn).catch(e => console.error(el.id, e)); } }, { rootMargin: '300px' }); io.observe(el); }
function lineChart(canvas, series, opt = {}) {
  const d = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect(); canvas.width = r.width * d; canvas.height = r.height * d; const g = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  g.clearRect(0, 0, W, H); const pad = { l: 46 * d, r: 8 * d, t: 8 * d, b: 22 * d };
  const xs = series.flatMap(s => s.x || s.y.map((_, i) => i)), ys = series.flatMap(s => s.y);
  const x0 = opt.x0 ?? Math.min(...xs), x1 = opt.x1 ?? Math.max(...xs), lo = opt.y0 ?? Math.min(...ys), hi = opt.y1 ?? Math.max(...ys), span = (hi - lo) || 1;
  const X = x => pad.l + (x - x0) / ((x1 - x0) || 1) * (W - pad.l - pad.r), Y = y => H - pad.b - (y - lo) / span * (H - pad.t - pad.b);
  g.strokeStyle = 'rgba(150,170,200,.25)'; g.fillStyle = '#9aa5b4'; g.font = `${11 * d}px sans-serif`; g.lineWidth = d;
  for (let i = 0; i <= 4; i++) { const y = lo + span * i / 4; g.beginPath(); g.moveTo(pad.l, Y(y)); g.lineTo(W - pad.r, Y(y)); g.stroke(); g.fillText(y.toFixed(Math.abs(span) < 5 ? 1 : 0), 4 * d, Y(y) + 4 * d); }
  (opt.bands || []).forEach(b => { g.fillStyle = b.color + '22'; g.fillRect(X(b.from), pad.t, X(b.to) - X(b.from), H - pad.t - pad.b); g.fillStyle = b.color; g.fillText(b.label, X(b.from) + 3 * d, pad.t + 11 * d); });
  series.forEach(s => { g.strokeStyle = s.color; g.lineWidth = 1.8 * d; g.beginPath(); s.y.forEach((y, i) => { const x = s.x ? s.x[i] : i; i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)); }); g.stroke(); });
  if (opt.cursor !== undefined) { g.strokeStyle = '#fff'; g.lineWidth = d; g.beginPath(); g.moveTo(X(opt.cursor), pad.t); g.lineTo(X(opt.cursor), H - pad.b); g.stroke(); }
  g.fillStyle = '#9aa5b4'; g.fillText(opt.xLabel || '', W - 130 * d, H - 4 * d);
}
const legend = series => `<div class="legend">${series.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('')}</div>`;

const sections = [
  ['live', 'Живая рука'], ['frames', 'Системы координат'], ['rot', 'Повороты: $M_x$ $M_y$ $M_z$'], ['fk', 'Прямая кинематика'], ['ik', 'Обратная кинематика'],
  ['camera', 'Камера → стол'], ['grasp', 'Захват'], ['gravity', 'Гравитация'], ['collide', 'Столкновения'], ['search', 'Поиск пути и плавность'], ['plan', 'Реальный план']
];
function build() {
  $('#toc').innerHTML = sections.map(([id, t]) => `<a href="#s-${id}" data-id="${id}">${t.replace(/\$/g, '').replace(/_/g, '')}</a>`).join('');
  $('#content').innerHTML = `
  <section id="s-live"><h2>Живая рука</h2>
    <p class="lead">Это настоящая рука, прямо сейчас. Программа, которая двигает моторы, каждые 0,1 секунды сообщает углы суставов; страница рисует по ним модель (те же STL-детали, что в LeLab) и считает остальное. Запустите работу на первой вкладке и смотрите здесь.</p>
    <div class="grid2"><div><canvas id="c-live" class="viewer"></canvas><p class="muted" id="live-state"></p></div>
    <div><div id="live-table"></div><h3>Углы суставов за последние 40 секунд</h3><canvas id="ch-live" class="chart"></canvas><div id="lg-live"></div><h3>Скорость кончика, мм/с</h3><canvas id="ch-speed" class="chart small"></canvas></div></div>
    <div class="note"><b>Что смотреть.</b> Углы меняются плавно — это нормальная работа; рывки на графике — признак «ломаного» движения. Скорость кончика сначала растёт, потом падает: так устроен профиль «минимального рывка» (мягко для редукторов).</div>
  </section>

  <section id="s-frames"><h2>Системы координат</h2>
    <p class="lead">Система координат — это «линейка с осями»: точка, откуда меряем, и три направления. Одно и то же положение кубика записывается по-разному в зависимости от того, откуда смотреть, поэтому роботу их нужно много.</p>
    <div class="grid2"><div><canvas id="c-frames" class="viewer"></canvas>
      <p class="muted">Оси везде одинаковые: <b style="color:#ff5d5d">красная</b> — $x$, <b style="color:#4fd16b">зелёная</b> — $y$, <b style="color:#5b9dff">синяя</b> — $z$. Крутите мышью, колесо — масштаб.</p></div>
    <div><div id="fr-list"></div><div id="fr-info" class="note"></div></div></div>
  </section>

  <section id="s-rot"><h2>Повороты: матрицы $M_x$, $M_y$, $M_z$</h2>
    <p class="lead">Поворот вокруг оси — это умножение вектора на матрицу $3\\times3$. Угол $\\varphi$ — на сколько провернули (правило правой руки: большой палец вдоль оси, пальцы показывают направление).</p>
    <div class="grid2"><div>
      <div class="formula">$$M_x(\\varphi)=\\begin{pmatrix}1&0&0\\\\0&\\cos\\varphi&-\\sin\\varphi\\\\0&\\sin\\varphi&\\cos\\varphi\\end{pmatrix}$$ $$M_y(\\varphi)=\\begin{pmatrix}\\cos\\varphi&0&\\sin\\varphi\\\\0&1&0\\\\-\\sin\\varphi&0&\\cos\\varphi\\end{pmatrix}$$ $$M_z(\\varphi)=\\begin{pmatrix}\\cos\\varphi&-\\sin\\varphi&0\\\\\\sin\\varphi&\\cos\\varphi&0\\\\0&0&1\\end{pmatrix}$$</div>
      <div class="sliders" style="grid-template-columns:90px 1fr 60px"><label>угол $\\varphi$</label><input type="range" id="rot-a" min="-180" max="180" step="1" value="40"><output id="rot-ao">40°</output>
        <label>ось</label><select id="rot-axis"><option value="x">x</option><option value="y">y</option><option value="z" selected>z</option></select><span></span></div>
      <div id="rot-mats"></div></div>
      <div><canvas id="c-rot" class="viewer"></canvas><p class="muted">Белая стрелка — вектор $\\vec v$ до поворота, жёлтая — $\\vec v'=M\\vec v$ после, дуга — путь. Фиолетовая — ось. Длина не меняется, меняется только направление.</p></div></div>
    <h3>Как это выглядит на руке</h3>
    <p>Каждый сустав — это стержень (фиолетовая стрелка), вокруг которого проворачивается следующее звено. В описании робота (URDF) ось каждого сустава — его собственная ось $z$, поэтому поворот сустава на угол $q$ — это просто $M_z(q)$ в его локальной системе. Выберите сустав и двигайте:</p>
    <div class="grid2"><div><canvas id="c-rot2" class="viewer"></canvas></div>
      <div><div class="sliders" style="grid-template-columns:140px 1fr 60px"><label>сустав</label><select id="rj-joint">${JOINTS.slice(0, 5).map(j => `<option value="${j}">${JOINT_RU[j]} (${j})</option>`).join('')}</select><span></span>
        <label>угол сустава $q$</label><input type="range" id="rj-q" min="-60" max="60" step="1" value="0"><output id="rj-qo">0°</output></div>
        <div id="rj-info"></div></div></div>
  </section>

  <section id="s-fk"><h2>Прямая кинематика</h2>
    <p class="lead">Вопрос: суставы повернуты на такие-то углы — где окажутся все звенья и кончик пальцев? Ответ — цепочка склеенных шагов: из системы одного звена в систему следующего.</p>
    <div class="formula">$$T=\\begin{pmatrix}R&\\vec p\\\\0&1\\end{pmatrix}$$</div>
    <p>Матрица $4\\times4$ хранит всё о положении звена: $R$ ($3\\times3$) — куда повёрнуты его оси, $\\vec p$ — где его начало. Чтобы перейти к следующему звену, матрицы умножают: <b>сначала фиксированный сдвиг и поворот из чертежа робота, потом поворот сустава</b> ($M_z$ на угол). Звено за звеном — до кончика.</p>
    <div class="grid2"><div><canvas id="c-fk" class="viewer"></canvas></div>
      <div><div class="sliders" id="fk-sliders"></div><div id="fk-result" style="margin-top:10px"></div></div></div>
    <h3>Идём по цепочке: шаг за шагом</h3>
    <div class="sliders" style="grid-template-columns:90px 1fr 80px"><label>шаг цепочки</label><input type="range" id="fk-step" min="1" max="6" step="1" value="1"><output id="fk-stepo">1</output></div>
    <div id="fk-steps"></div>
  </section>

  <section id="s-ik"><h2>Обратная кинематика</h2>
    <p class="lead">Обратная задача: известно, куда должен прийти кончик пальцев, а углы суставов надо найти. Рука решает её перед каждым движением.</p>
    <p>Трудность: у руки 5 суставов, у цели — 3 координаты, поэтому свободным остаётся наклон кисти. Решается в три приёма:</p>
    <ol><li><b>Развернуться к цели</b> (поворот основания) — рука становится плоской: плечо, локоть, запястье лежат в одной вертикальной плоскости.</li>
      <li><b>Треугольник.</b> Плечо и предплечо — две стороны, расстояние от плеча до запястья — третья. По трём сторонам угол в локте даёт теорема косинусов.</li>
      <li><b>Поправка.</b> Если решение чуть промахнулось, его «подкручивают» методом Ньютона: смотрят, как промах зависит от каждого сустава, и сдвигают на нужную долю.</li></ol>
    <div class="grid2"><div>
      <div class="sliders" style="grid-template-columns:40px 1fr 60px"><label>x, м</label><input type="range" id="ik-x" min="0.08" max="0.32" step="0.005" value="0.20"><output id="ik-xo"></output>
        <label>y, м</label><input type="range" id="ik-y" min="-0.30" max="0.30" step="0.005" value="-0.10"><output id="ik-yo"></output>
        <label>z, м</label><input type="range" id="ik-z" min="0.01" max="0.30" step="0.005" value="0.08"><output id="ik-zo"></output></div>
      <canvas id="c-ik" class="viewer" style="margin-top:8px"></canvas>
      <div class="sliders" style="grid-template-columns:110px 1fr 60px;margin-top:8px"><label>шаг Ньютона</label><input type="range" id="ik-iter" min="0" max="3" step="1" value="0"><output id="ik-itero">0</output></div>
      <p class="muted" id="ik-iterinfo"></p></div>
      <div><h3>Треугольник в плоскости руки (в масштабе)</h3><svg id="ik-svg" viewBox="0 0 360 260" style="width:100%;background:#0b0f16;border-radius:10px"></svg><div id="ik-result"></div></div></div>
  </section>

  <section id="s-camera"><h2>Камера → стол</h2>
    <p class="lead">Камера даёт пиксели, а рука живёт в метрах. Мост между ними — калибровка (числа, найденные по красному кресту на подвижной губке).</p>
    <ol><li>Через пиксель $(u,v)$ проводим <b>луч</b> из центра камеры (матрица камеры $K$: фокус и центр кадра).</li>
      <li>Луч переводим в систему основания (положение камеры $T$ — где она висит и куда смотрит).</li>
      <li>Ищем, где луч пересекает горизонтальную плоскость: стол ($z=0$) или кромку коробки ($z\\approx0{,}068$).</li></ol>
    <div class="grid2"><div><div style="position:relative"><img id="cam2" style="width:100%;border-radius:10px;cursor:crosshair" alt=""><div id="cam-mark" style="position:absolute;width:14px;height:14px;border:2px solid #f59e0b;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;display:none"></div></div>
      <canvas id="c-cam" class="viewer" style="margin-top:8px"></canvas></div>
      <div id="cam-result"><p class="muted">Нажмите на кубик или край коробки на картинке: слева покажется луч в 3D, справа — все числа.</p></div></div>
  </section>

  <section id="s-grasp"><h2>Захват</h2>
    <p class="lead">Схват — две губки: <b>неподвижная</b> (часть кисти) и <b>подвижная</b>. Кубик зажимается между ними вдоль оси смыкания (жёлтая стрелка). На картинке — настоящая поза захвата из последнего плана.</p>
    <div class="grid2"><div><canvas id="c-grasp" class="viewer"></canvas></div><div id="grasp-info"></div></div>
    <ul><li>Перед смыканием неподвижная губка стоит <b>рядом</b> с гранью кубика, не касаясь (зазор): если коснуться раньше времени, кубик сдвинется.</li>
      <li>Подвижная губка идёт навстречу, упирается в противоположную грань и <b>двигает кубик</b> на этот зазор до неподвижной. После захвата кубик лежит вплотную к неподвижной губке — программа так и считает.</li>
      <li>Ширина раскрытия — размер кубика плюс запас: губки проходят по бокам и не цепляют.</li></ul>
  </section>

  <section id="s-gravity"><h2>Гравитация</h2>
    <p class="lead">Рука тяжёлая, а моторы не жёсткие: под весом звена мотор проседает на градус-другой. Чтобы не проседать, драйвер даёт мотору цель <b>чуть дальше</b>, чем нужно.</p>
    <div class="formula">$$q_{\\text{команда}}=q_{\\text{цель}}+s\\cdot\\tau(q_{\\text{цель}}),\\qquad s=3{,}7\\ \\text{град/(Н·м)}$$</div>
    <p>$\\tau$ — момент силы тяжести на суставе в этой позе. Чем дальше вытянута рука, тем он больше (фиолетовые стрелки).</p>
    <div class="grid2"><div><canvas id="c-gravity" class="viewer"></canvas></div><div id="gravity-info"></div></div>
    <p>Побочный эффект: «заказанная» поза (оранжевая точка) не совпадает с плановой (белая). У захвата это ~5 мм — поэтому зазоры берутся с запасом и проверяются <b>обе</b> позы.</p>
  </section>

  <section id="s-collide"><h2>Столкновения</h2>
    <p class="lead">Перед любым движением программа проверяет каждую позу по настоящим 3D-деталям руки: не пересекают ли они кубики, стенки коробки и стол. Двигайте ползунок по реальному плану — звенья окрасятся по зазору.</p>
    <div class="grid2"><div><canvas id="c-collide" class="viewer"></canvas><div id="co-ctl"></div></div><div id="co-table"></div></div>
    <ul><li><span style="color:#22c55e">зелёный</span> — зазор больше 15 мм, <span style="color:#facc15">жёлтый</span> — 5–15 мм, <span style="color:#ef4444">красный</span> — меньше 5 мм.</li>
      <li>Запасы складываются: неточность измерения (3 мм и больше), отставание руки от плана (до $3^\\circ$ на суставе — около сантиметра у кончика) и намеренный зазор планировщика (12 мм до стенок, 24 мм вокруг клапанов; сужается, только если иначе пути нет).</li>
      <li>Сама проверка: два тела <b>не пересекаются</b>, если между ними можно провести разделяющую плоскость (тест разделяющей оси).</li></ul>
  </section>

  <section id="s-search"><h2>Поиск пути и плавность</h2>
    <p class="lead">Перебирать углы нельзя: у руки 5 суставов, вариантов бесконечно много. Используется <b>RRT-Connect</b> — два дерева случайных путей растут навстречу друг другу.</p>
    <ol><li>Берём случайную комбинацию углов $q_\\text{rand}$.</li>
      <li>Дерево А делает короткий шаг от ближайшего узла в её сторону; если шаг свободен от столкновений — добавляет узел.</li>
      <li>Дерево Б жадно тянется к этому новому узлу, пока не дойдёт или не упрётся.</li>
      <li>Дошло — деревья сошлись, путь найден. Нет — деревья меняются ролями, повторяем.</li></ol>
    <div class="grid2"><div><canvas id="v-rrt" class="viewer" style="aspect-ratio:16/10"></canvas><p><button id="rrt-run">Запустить</button> <button id="rrt-new" class="ghost" style="margin-left:6px">Новое поле</button></p><p class="muted" id="rrt-info"></p></div>
      <div><p>Это то же самое в двух измерениях. Синее дерево — от старта, оранжевое — от цели, красное — препятствия, серая линия — найденный ломаный путь, зелёная — после <b>спрямления</b> (куски заменяются прямыми, если они свободны).</p>
      <p>В роботе пространство 5-мерное (по углу на сустав), а «свободно ли» отвечает та самая проверка столкновений из предыдущего раздела.</p>
      <h3>Почему рука двигалась рывками, и что сделано</h3>
      <p>Найденный путь — ломаная в пространстве углов; на каждом изломе направление меняется скачком. Теперь путь после поиска <b>сглаживается сплайном</b> (каждая точка заново проверяется на столкновения), а скорость задана по фазам: быстро в свободном месте, осторожно рядом с кубиком и стенками.</p>
      <div id="speed-table"></div></div></div>
  </section>

  <section id="s-plan"><h2>Реальный план</h2>
    <p class="lead">Последний план, по которому двигалась рука. Выберите фазу и двигайте ползунок: рука встанет в позу, которую программа реально отправляла на моторы, справа — её углы.</p>
    <div class="grid2"><div><canvas id="c-plan" class="viewer"></canvas><div id="plan-ctl"></div><div class="legend" id="plan-legend"></div></div>
      <div><div id="plan-result"></div><h3>Углы суставов по всему плану</h3><canvas id="ch-plan" class="chart"></canvas><div id="lg-plan"></div></div></div>
  </section>`;
  math($('#content'));
  document.querySelectorAll('.toc a').forEach(a => a.addEventListener('click', e => { e.preventDefault(); document.getElementById('s-' + a.dataset.id).scrollIntoView({ behavior: 'smooth', block: 'start' }); }));
}

function sceneWithObjects(sc, plan) {
  if (!plan) return;
  const b = plan.box; if (b && b.center_xy_m) sc.boxFrame(b.center_xy_m, b.inner_size_m, b.yaw_rad || 0, b.rim_height_m || 0.073);
  Object.entries(plan.cubes || {}).forEach(([k, c]) => { sc.cubeBox(c.vertices_m, c.unmeasured ? 0x9aa5b4 : 0xfacc15, c.unmeasured ? 0.12 : 0.28); sc.label([c.center_m[0], c.center_m[1], c.center_m[2] + 0.035], k + (c.unmeasured ? ' (не измерен)' : ''), '#facc15', 0.022); });
}
function cubeFrame(c) {
  const v = c.vertices_m, a = [1, 3, 4].map(i => { const w = [v[i][0] - v[0][0], v[i][1] - v[0][1], v[i][2] - v[0][2]], n = Math.hypot(...w); return w.map(x => x / n); });
  return [[a[0][0], a[1][0], a[2][0], c.center_m[0]], [a[0][1], a[1][1], a[2][1], c.center_m[1]], [a[0][2], a[1][2], a[2][2], c.center_m[2]], [0, 0, 0, 1]];
}
const baseFromCamera = () => { const T = ROBOT.camera.camera_from_robot, R = [0, 1, 2].map(i => [0, 1, 2].map(j => T[j][i])), t = [T[0][3], T[1][3], T[2][3]], p = [0, 1, 2].map(i => -(R[i][0] * t[0] + R[i][1] * t[1] + R[i][2] * t[2])); return [[...R[0], p[0]], [...R[1], p[1]], [...R[2], p[2]], [0, 0, 0, 1]]; };
function frustum(sc, cam) {
  const K = ROBOT.camera.matrix, C = [cam[0][3], cam[1][3], cam[2][3]];
  [[0, 0], [1280, 0], [1280, 720], [0, 720]].forEach(([u, v]) => { const r = [(u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1], 1], d = [0, 1, 2].map(i => cam[i][0] * r[0] + cam[i][1] * r[1] + cam[i][2] * r[2]), lam = -C[2] / d[2]; sc.line([C, [C[0] + lam * d[0], C[1] + lam * d[1], 0]], 0xfacc15, 0.45); });
}

/* ---------------------------------------------------------------- live */
function initLive() {
  const sc = new Scene3D($('#c-live'), { cam: [0.5, -0.65, 0.45], target: [0.1, 0, 0.1] }); sc.addRobot();
  sc.frameHelper('tcp', 0.06, 'кончик');
  const hist = [], speed = []; let last = null;
  const tick = async () => {
    try {
      const L = await api('/api/live'); if (!L.pos || !L.frames) { $('#live-state').textContent = 'Рука пока не прислала данных (программа не запущена).'; return; }
      if (sc.links.base_link) { sc.setPose(L.frames); sc.setFrame('tcp', L.frames.gripper_frame_link); }
      const t = Date.now() / 1000; hist.push({ t, q: JOINTS.map(j => L.pos[j]) }); while (hist.length && t - hist[0].t > 40) hist.shift();
      if (last) { const dt = t - last.t; if (dt > 0.05) { speed.push({ t, v: Math.hypot(...L.tip.map((x, i) => x - last.tip[i])) / dt * 1000 }); last = { t, tip: L.tip }; } } else last = { t, tip: L.tip };
      while (speed.length && t - speed[0].t > 40) speed.shift();
      const lim = ROBOT.ranges_deg;
      $('#live-table').innerHTML = `<table class="num"><tr><th>сустав</th><th>угол</th><th>в пределах</th></tr>${JOINTS.map((j, i) => { const v = L.pos[j], r = i < 5 ? lim[i] : [0, 100], p = (v - r[0]) / (r[1] - r[0]) * 100; return `<tr><td>${JOINT_RU[j]}</td><td>${f(v, 1)}${i < 5 ? '°' : '%'}</td><td style="width:45%"><div class="bar"><i style="left:${Math.max(0, Math.min(100, p))}%"></i></div></td></tr>`; }).join('')}</table>
        <div>Кончик: x = <b>${f(L.tip[0], 3)}</b> м, y = <b>${f(L.tip[1], 3)}</b> м, z = <b>${f(L.tip[2], 3)}</b> м · наклон кисти ${f(L.pitch_deg, 1)}°</div>`;
      $('#live-state').textContent = L.age_s < 3 ? `данные свежие (${f(L.age_s, 1)} с назад)` : `последнее измерение ${f(L.age_s, 0)} с назад: рука стоит или программа не запущена`;
      const series = JOINTS.slice(0, 5).map((j, i) => ({ name: JOINT_RU[j], color: JOINT_COLORS[i], x: hist.map(h => h.t - t), y: hist.map(h => h.q[i]) }));
      lineChart($('#ch-live'), series, { x0: -40, x1: 0, xLabel: 'секунды назад' }); $('#lg-live').innerHTML = legend(series);
      if (speed.length > 2) lineChart($('#ch-speed'), [{ color: '#f59e0b', x: speed.map(s => s.t - t), y: speed.map(s => s.v) }], { x0: -40, x1: 0, y0: 0 });
    } catch (e) { }
  };
  setInterval(() => { if ($('#learn').classList.contains('active') && !document.hidden) tick(); }, 500); tick();
}

/* ---------------------------------------------------------------- frames */
async function initFrames() {
  const sc = new Scene3D($('#c-frames'), { cam: [0.6, -0.7, 0.5], target: [0.15, -0.1, 0.08] }); await sc.addRobot({ opacity: 0.55 });
  const fk = await frames0({ shoulder_pan: 15, shoulder_lift: -35, elbow_flex: 45, wrist_flex: 55, wrist_roll: 0, gripper: 30 });
  PLAN = PLAN || await api('/api/plan').catch(() => null);
  const I = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const groups = {
    base: { name: 'Основание (мировая)', color: '#ff9', info: 'Начало — у основания руки. Все положения в программе записаны в ней: «кубик в 0,2 м вперёд и 0,1 м вправо». Ось $x$ — вперёд от руки, $y$ — влево, $z$ — вверх.' },
    links: { name: 'Звенья руки', color: '#5b8def', info: 'У каждого звена своя система, она едет вместе с ним. Положение звена внутри предыдущего задаёт чертёж робота (URDF), а поворот сустава — угол. Подвигайте суставы в разделе «Прямая кинематика».' },
    tool: { name: 'Кончик (TCP)', color: '#22c55e', info: 'Точка между губками. Планировщик ставит именно её: «кончик в этой точке, ось $z$ вниз». Ось $z$ инструмента — куда смотрят пальцы; $x$ — вдоль смыкания губок.' },
    camera: { name: 'Камера', color: '#facc15', info: 'Обзорная камера висит над столом и смотрит вниз. Её система «оптическая»: $z$ вдоль луча, $x$ вправо по кадру, $y$ вниз по кадру. Жёлтые линии — границы кадра на столе.' },
    table: { name: 'Стол', color: '#9aa5b4', info: 'Плоскость $z=0$. Всё лежащее на столе имеет $z$ равным своей высоте. Клетка сетки — 10 см.' },
    box: { name: 'Коробка', color: '#c7d0dc', info: 'Начало — в центре проёма, оси повёрнуты вместе с коробкой. Планировщик проверяет стенки в этой системе: «внутри ли кубик», «выше ли кромка».' },
    cube: { name: 'Кубик', color: '#facc15', info: 'Начало — центр кубика, оси вдоль его граней: по ним программа выбирает, где сомкнуть губки.' }
  };
  const names = { shoulder_link: 'плечо', upper_arm_link: 'верх', lower_arm_link: 'локоть', wrist_link: 'запястье', gripper_link: 'кисть' };
  sc.frameHelper('base', 0.12, 'основание'); sc.setFrame('base', I);
  Object.keys(names).forEach(l => sc.linkFrame(l, 0.045, names[l]));
  sc.frameHelper('tcp', 0.06, 'TCP');
  const cam = baseFromCamera(); sc.frameHelper('camera', 0.08, 'камера'); sc.setFrame('camera', cam);
  const tableT = [[1, 0, 0, 0.35], [0, 1, 0, -0.35], [0, 0, 1, 0], [0, 0, 0, 1]]; sc.frameHelper('table', 0.06, 'стол'); sc.setFrame('table', tableT);
  sc.clearDynamic(); frustum(sc, cam);
  let boxT = null, cubeT = null;
  if (PLAN && PLAN.box && PLAN.box.center_xy_m) { const b = PLAN.box, c = Math.cos(b.yaw_rad || 0), s = Math.sin(b.yaw_rad || 0); boxT = [[c, -s, 0, b.center_xy_m[0]], [s, c, 0, b.center_xy_m[1]], [0, 0, 1, 0], [0, 0, 0, 1]]; sc.frameHelper('box', 0.07, 'коробка'); sc.setFrame('box', boxT); sceneWithObjects(sc, PLAN); }
  const cube0 = PLAN && Object.values(PLAN.cubes || {}).find(c => !c.unmeasured); if (cube0) { cubeT = cubeFrame(cube0); sc.frameHelper('cube', 0.05, 'кубик'); sc.setFrame('cube', cubeT); }
  sc.setPose(fk.frames); sc.setFrame('tcp', fk.frames.gripper_frame_link);
  const show = { base: ['base'], links: Object.keys(sc.frames).filter(k => k.startsWith('link:')), tool: ['tcp'], camera: ['camera'], table: ['table'], box: boxT ? ['box'] : [], cube: cubeT ? ['cube'] : [] };
  const pos = { base: I, tool: fk.frames.gripper_frame_link, camera: cam, table: tableT, box: boxT, cube: cubeT };
  $('#fr-list').innerHTML = `<table class="num"><tr><th></th><th>система</th><th>начало в основании, м</th></tr>${Object.entries(groups).map(([k, g]) => {
    const T = pos[k], o = T ? [T[0][3], T[1][3], T[2][3]] : null;
    return `<tr data-k="${k}" style="cursor:pointer"><td><input type="checkbox" checked data-k="${k}"></td><td style="color:${g.color}"><b>${g.name}</b></td><td>${o ? `(${o.map(x => f(x, 3)).join(', ')})` : (k === 'links' ? 'у каждого своё' : 'ещё нет данных')}</td></tr>`;
  }).join('')}</table>`;
  const info = k => { $('#fr-info').innerHTML = `<b>${groups[k].name}.</b> ${groups[k].info}<p class="muted">Чтобы перевести точку из одной системы в другую, её умножают на матрицу $4\\times4$ этой системы (раздел «Прямая кинематика»); камера → стол описана в разделе «Камера → стол».</p>`; math($('#fr-info')); };
  $('#fr-list').addEventListener('change', e => { const k = e.target.dataset.k; if (k) (show[k] || []).forEach(x => sc.showFrame(x, e.target.checked)); });
  $('#fr-list').addEventListener('click', e => { const tr = e.target.closest('tr[data-k]'); if (tr) info(tr.dataset.k); });
  info('base');
}

/* ---------------------------------------------------------------- rotations */
async function initRot() {
  const sc = new Scene3D($('#c-rot'), { cam: [1.7, -1.5, 1.2], target: [0, 0, 0.25], table: false });
  const sc2 = new Scene3D($('#c-rot2'), { cam: [0.55, -0.7, 0.45], target: [0.1, 0, 0.12] }); await sc2.addRobot({ opacity: 0.8 });
  const Rm = (ax, a) => { const c = Math.cos(a), s = Math.sin(a); return ax === 'x' ? [[1, 0, 0], [0, c, -s], [0, s, c]] : ax === 'y' ? [[c, 0, s], [0, 1, 0], [-s, 0, c]] : [[c, -s, 0], [s, c, 0], [0, 0, 1]]; };
  const apply = (M, v) => M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
  sc.frameHelper('o', 1.1, '');
  sc.setFrame('o', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
  const draw = () => {
    const deg = +$('#rot-a').value, a = deg * Math.PI / 180, ax = $('#rot-axis').value; $('#rot-ao').textContent = deg + '°';
    const v = [0.9, 0.5, 0.4], v2 = apply(Rm(ax, a), v);
    sc.clearDynamic();
    sc.arrow([0, 0, 0], v, 0xffffff, 0.08); sc.label(v, 'v', '#fff', 0.1); sc.arrow([0, 0, 0], v2, 0xfacc15, 0.08); sc.label(v2, "v' = M·v", '#facc15', 0.1);
    const axv = { x: [1.2, 0, 0], y: [0, 1.2, 0], z: [0, 0, 1.2] }[ax]; sc.arrow([0, 0, 0], axv, 0xe879f9, 0.07); sc.label(axv, 'ось ' + ax, '#f0abfc', 0.1);
    const arc = []; for (let t = 0; t <= 1.0001; t += 0.05) arc.push(apply(Rm(ax, a * t), v)); sc.line(arc, 0xfacc15, 0.7); sc.render();
    $('#rot-mats').innerHTML = ['x', 'y', 'z'].map(k => `<div style="${k === ax ? 'outline:1px solid var(--accent);border-radius:8px;' : 'opacity:.55;'}padding:2px 6px"><b>$M_${k}$(${deg}°)</b> ${mat(Rm(k, a), 3)}</div>`).join('') + `<div class="muted">$\\vec v$ = (${v.join('; ')}) → $\\vec v'$ = (${v2.map(x => f(x, 3)).join('; ')})</div>`;
    math($('#rot-mats'));
  };
  ['rot-a', 'rot-axis'].forEach(id => $('#' + id).addEventListener('input', draw)); draw();
  const joint = async () => {
    const j = $('#rj-joint').value, q = +$('#rj-q').value; $('#rj-qo').textContent = q + '°';
    const base = { shoulder_pan: 0, shoulder_lift: -30, elbow_flex: 45, wrist_flex: 55, wrist_roll: 0, gripper: 30 }; const r = await frames0({ ...base, [j]: base[j] + q });
    sc2.setPose(r.frames); sc2.clearDynamic(); const link = { shoulder_pan: 'shoulder_link', shoulder_lift: 'upper_arm_link', elbow_flex: 'lower_arm_link', wrist_flex: 'wrist_link', wrist_roll: 'gripper_link' }[j];
    sc2.jointAxis(j, r.frames[link], `${j}: ${q >= 0 ? '+' : ''}${q}°`, 0xe879f9, 0.12); sc2.render();
    const ang = base[j] + q + ROBOT.offsets_deg[JOINTS.indexOf(j)];
    $('#rj-info').innerHTML = `<p>Сустав <b>${JOINT_RU[j]}</b> вращается вокруг фиолетовой оси. Его матрица в собственной системе звена:</p>${mat(Rm('z', ang * Math.PI / 180), 3)}
      <p class="muted">$M_z(q)$ при $q$ = ${f(ang, 1)}° (угол сустава + смещение нуля из калибровки). Всё, что стоит дальше по цепочке, едет вместе с суставом.</p>`;
    math($('#rj-info'));
  };
  ['rj-joint', 'rj-q'].forEach(id => $('#' + id).addEventListener('input', joint)); joint();
}

/* ---------------------------------------------------------------- forward kinematics */
async function initFK() {
  const sc = new Scene3D($('#c-fk'), { cam: [0.5, -0.6, 0.42], target: [0.1, 0, 0.1] }); await sc.addRobot({ opacity: 0.8 });
  const lim = ROBOT.ranges_deg, init = { shoulder_pan: 20, shoulder_lift: -30, elbow_flex: 45, wrist_flex: 55, wrist_roll: 0, gripper: 30 };
  $('#fk-sliders').innerHTML = JOINTS.map((n, i) => `<label>${JOINT_RU[n]}</label><input type="range" id="q-${n}" min="${i < 5 ? lim[i][0] : 0}" max="${i < 5 ? lim[i][1] : 100}" step="0.5" value="${init[n]}"><output id="o-${n}"></output>`).join('');
  let busy = false, again = false, steps = null;
  sc.frameHelper('step', 0.07, 'шаг');
  const showStep = () => {
    if (!steps) return; const k = +$('#fk-step').value - 1, s = steps[k]; $('#fk-stepo').textContent = `${k + 1} из ${steps.length}`;
    sc.clearDynamic(); const P = steps.slice(0, k + 1).map(st => [st.cumulative[0][3], st.cumulative[1][3], st.cumulative[2][3]]);
    P.forEach((p, i) => sc.dot(p, i === k ? 0xfacc15 : 0x9aa5b4, i === k ? 0.01 : 0.006)); if (P.length > 0) sc.line([[0, 0, 0], ...P], 0xfacc15);
    sc.setFrame('step', s.cumulative);
    $('#fk-steps').innerHTML = `<p><b>Шаг ${k + 1}: ${s.parent} → ${s.child}.</b> ${s.angle_deg === null || s.angle_deg === undefined ? 'Здесь сустава нет — просто фиксированный сдвиг от кисти к кончику.' : `Сустав <code>${s.joint}</code> повёрнут на ${f(s.angle_deg, 2)}° (с учётом смещения нуля).`}</p>
      <div class="muted">Сдвиг из чертежа: (${s.xyz.map(x => f(x, 4)).join(', ')}) м; поворот из чертежа: (${s.rpy.map(x => f(x, 3)).join(', ')}) рад.</div>
      <div class="rowmat"><span>фиксированная часть</span>${mat(s.fixed)}<span>×</span><span>поворот сустава</span>${mat(s.turn)}</div>
      <div class="rowmat"><span>итог по цепочке (где это звено в основании)</span>${mat(s.cumulative, 4)}</div>
      <div class="muted">Левый верхний блок $3\\times3$ — повёрнутые оси звена; правый столбец — его начало, м. Каждый шаг умножает прежний итог на новую матрицу.</div>`;
    math($('#fk-steps'));
  };
  const update = async () => {
    if (busy) { again = true; return; } busy = true;
    const q = Object.fromEntries(JOINTS.map(n => [n, +$('#q-' + n).value])); JOINTS.forEach(n => $('#o-' + n).textContent = (+$('#q-' + n).value).toFixed(1) + (n === 'gripper' ? '%' : '°'));
    try {
      const r = await frames0(q); steps = r.steps; $('#fk-step').max = steps.length; sc.setPose(r.frames);
      $('#fk-result').innerHTML = `<b>Кончик пальцев:</b> x = ${f(r.tip[0], 4)} м, y = ${f(r.tip[1], 4)} м, z = ${f(r.tip[2], 4)} м, наклон ${f(r.pitch_deg, 1)}°.<br><span class="muted">Библиотека so101ik считает то же и расходится с цепочкой на ${r.max_difference_to_library_m.toExponential(1)} м — это округление.</span>`;
      showStep();
    } catch (e) { $('#fk-result').textContent = String(e); }
    busy = false; if (again) { again = false; update(); }
  };
  JOINTS.forEach(n => $('#q-' + n).addEventListener('input', update)); $('#fk-step').addEventListener('input', showStep); update();
}

/* ---------------------------------------------------------------- inverse kinematics */
async function initIK() {
  const sc = new Scene3D($('#c-ik'), { cam: [0.5, -0.6, 0.42], target: [0.15, -0.05, 0.08] }); await sc.addRobot({ opacity: 0.8 });
  let busy = false, again = false, last = null;
  const drawTarget = () => { const t = last.target; sc.clearDynamic(); sc.dot(t, 0xff5d5d, 0.008); sc.label([t[0], t[1], t[2] + 0.025], 'цель', '#ff9', 0.026); sc.render(); };
  const showIter = async () => {
    if (!last || !last.newton) return; const k = Math.min(+$('#ik-iter').value, last.newton.length - 1), row = last.newton[k]; $('#ik-itero').textContent = k;
    const fk = await frames0({ shoulder_pan: row.q[0], shoulder_lift: row.q[1], elbow_flex: row.q[2], wrist_flex: row.q[3], wrist_roll: 0, gripper: 30 });
    sc.setPose(fk.frames); drawTarget();
    $('#ik-iterinfo').innerHTML = k === 0 ? `Начало: решение испорчено на 8°, −6°, 10°, −9°. Кончик промахивается на <b>${f(row.error_mm, 2)} мм</b>.` : `После ${k}-го шага Ньютона промах <b>${f(row.error_mm, 4)} мм</b>.`;
  };
  const solve = async () => {
    if (busy) { again = true; return; } busy = true;
    const t = { x: +$('#ik-x').value, y: +$('#ik-y').value, z: +$('#ik-z').value }; ['x', 'y', 'z'].forEach(k => $('#ik-' + k + 'o').textContent = t[k].toFixed(3));
    try {
      const r = await api('/api/ik', t); last = { ...r, target: [t.x, t.y, t.z] }; $('#ik-iter').max = Math.max(0, Math.min(3, (r.newton || []).length - 1));
      let html = '';
      if (r.best.deg) {
        const tri = r.triangle, q = r.best.deg, elbow = r.gamma0_deg - tri.gamma_deg - r.elbow_offset_deg;
        html += `<p>Лучшее решение: поворот ${f(q.shoulder_pan, 1)}°, плечо ${f(q.shoulder_lift, 1)}°, локоть ${f(q.elbow_flex, 1)}°, кисть ${f(q.wrist_flex, 1)}° (наклон ${f(r.best.pitch, 0)}°). Из семейства решений выбрано то, где моторы несут меньше всего нагрузки от тяжести.</p>
        <table class="num"><tr><td>плечо→локоть $L_1$</td><td>${f(tri.L1 * 1000, 1)} мм</td></tr><tr><td>локоть→запястье $L_2$</td><td>${f(tri.L2 * 1000, 1)} мм</td></tr><tr><td>плечо→запястье $d$</td><td>${f(tri.d * 1000, 1)} мм</td></tr><tr><td>угол в локте $\\gamma$ (теорема косинусов)</td><td>${f(tri.gamma_deg, 2)}°</td></tr><tr><td>локоть по формуле $\\gamma_0-\\gamma-o$</td><td><b>${f(elbow, 3)}°</b></td></tr><tr><td>локоть из библиотеки</td><td><b>${f(q.elbow_flex, 3)}°</b></td></tr></table>`;
        const sc_ = 900, S = [40, 210], cosA = (tri.L1 ** 2 + tri.d ** 2 - tri.L2 ** 2) / (2 * tri.L1 * tri.d), sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA)), W = [S[0] + tri.d * sc_, S[1]], E = [S[0] + tri.L1 * sc_ * cosA, S[1] - tri.L1 * sc_ * sinA];
        $('#ik-svg').innerHTML = `<polygon points="${S} ${E} ${W}" fill="rgba(245,158,11,.15)" stroke="#f59e0b" stroke-width="2"/><circle cx="${S[0]}" cy="${S[1]}" r="5" fill="#5b8def"/><circle cx="${E[0]}" cy="${E[1]}" r="5" fill="#22c55e"/><circle cx="${W[0]}" cy="${W[1]}" r="5" fill="#ef4444"/>
          <text x="${S[0] - 14}" y="${S[1] + 20}" fill="#9aa5b4" font-size="12">плечо</text><text x="${E[0] - 10}" y="${E[1] - 8}" fill="#9aa5b4" font-size="12">локоть</text><text x="${W[0] - 30}" y="${W[1] + 20}" fill="#9aa5b4" font-size="12">запястье</text>
          <text x="${(S[0] + E[0]) / 2 - 40}" y="${(S[1] + E[1]) / 2 - 4}" fill="#f59e0b" font-size="12">L₁ ${f(tri.L1 * 1000, 0)}</text><text x="${(E[0] + W[0]) / 2 + 6}" y="${(E[1] + W[1]) / 2 - 4}" fill="#f59e0b" font-size="12">L₂ ${f(tri.L2 * 1000, 0)}</text><text x="${(S[0] + W[0]) / 2 - 18}" y="${S[1] + 16}" fill="#f59e0b" font-size="12">d ${f(tri.d * 1000, 0)}</text><text x="${E[0] + 8}" y="${E[1] + 18}" fill="#22c55e" font-size="12">γ ${f(tri.gamma_deg, 1)}°</text>`;
      } else { html += `<div class="note">Для этой точки нет допустимой позы: ${r.best.error || ''}</div>`; $('#ik-svg').innerHTML = ''; }
      html += `<h3>Семейство решений по наклону кисти</h3><table class="num"><tr><th>наклон</th><th>pan</th><th>lift</th><th>elbow</th><th>wrist</th><th>нагрузка</th></tr>` +
        r.rows.map(x => x.ok ? `<tr${r.best.pitch !== undefined && x.pitch === r.best.pitch ? ' style="color:var(--accent);font-weight:700"' : ''}><td>${x.pitch}°</td><td>${f(x.deg.shoulder_pan, 1)}</td><td>${f(x.deg.shoulder_lift, 1)}</td><td>${f(x.deg.elbow_flex, 1)}</td><td>${f(x.deg.wrist_flex, 1)}</td><td>${f(x.cost, 3)}</td></tr>` : `<tr><td>${x.pitch}°</td><td colspan="5" style="text-align:left;color:var(--muted)">не достижимо</td></tr>`).join('') + '</table>';
      if (r.newton) html += `<h3>Поправка Ньютона</h3><table class="num"><tr><th>шаг</th><th>промах, мм</th></tr>${r.newton.slice(0, 5).map(n => `<tr><td>${n.k}</td><td>${f(n.error_mm, 4)}</td></tr>`).join('')}</table><div class="muted">Промах падает почти квадратично. Двигайте «шаг Ньютона» под 3D — рука встанет в позу каждого шага.</div>`;
      $('#ik-result').innerHTML = html; math($('#ik-result')); await showIter();
    } catch (e) { $('#ik-result').innerHTML = `<div class="note">${e}</div>`; }
    busy = false; if (again) { again = false; solve(); }
  };
  ['x', 'y', 'z'].forEach(k => $('#ik-' + k).addEventListener('input', solve)); $('#ik-iter').addEventListener('input', showIter); solve();
}

/* ---------------------------------------------------------------- camera */
async function initCamera() {
  const sc = new Scene3D($('#c-cam'), { cam: [0.75, -0.75, 0.75], target: [0.22, 0, 0.25] }); await sc.addRobot({ opacity: 0.5 });
  const fk = await frames0({ shoulder_pan: 0, shoulder_lift: -100, elbow_flex: 92, wrist_flex: 74, wrist_roll: 0, gripper: 20 }); sc.setPose(fk.frames);
  const cam = baseFromCamera(); sc.frameHelper('camera', 0.1, 'камера'); sc.setFrame('camera', cam); sc.clearDynamic(); frustum(sc, cam); sc.render();
  const img = $('#cam2'); img.src = '/snapshot.jpg?t=' + Date.now();
  img.addEventListener('click', async e => {
    const r = img.getBoundingClientRect(), u = (e.clientX - r.left) / r.width * 1280, v = (e.clientY - r.top) / r.height * 720;
    const m = $('#cam-mark'); m.style.display = 'block'; m.style.left = (e.clientX - r.left) + 'px'; m.style.top = (e.clientY - r.top) + 'px';
    const [a, b] = await Promise.all([api('/api/pixel', { u, v, z: 0 }), api('/api/pixel', { u, v, z: 0.068 })]);
    sc.clearDynamic(); frustum(sc, cam); sc.line([a.camera_centre_robot, a.point], 0xff5d5d); sc.dot(a.point, 0xff5d5d, 0.01); sc.dot(b.point, 0x4fd16b, 0.008); sc.label([a.point[0], a.point[1], 0.03], 'на столе', '#ff9', 0.024); sc.label([b.point[0], b.point[1], 0.1], 'на кромке', '#9f9', 0.024); sc.render();
    $('#cam-result').innerHTML = `<h3>Что тут посчитано</h3><ol>
      <li>Пиксель: $(u,v)$ = (${f(a.u, 0)}, ${f(a.v, 0)}).</li>
      <li>Луч в системе камеры: $\\left(\\tfrac{u-c_x}{f_x},\\tfrac{v-c_y}{f_y},1\\right)$ = (${a.ray_camera.map(x => f(x, 4)).join('; ')}). Матрица камеры $K$:${mat(a.camera_matrix, 1)}<div class="muted">$f_x,f_y$ — «увеличение» в пикселях, $c_x,c_y$ — центр кадра.</div></li>
      <li>Положение камеры (из калибровки по красному кресту), 4×4:${mat(a.camera_from_robot, 4)}<div class="muted">центр камеры в основании: (${a.camera_centre_robot.map(x => f(x, 3)).join('; ')}) м.</div></li>
      <li>Луч в системе основания: (${a.ray_robot.map(x => f(x, 4)).join('; ')}). До стола луч идёт $\\lambda$ = ${f(a.lambda, 3)}, до кромки — ${f(b.lambda, 3)}.</li>
      <li>Точка на столе: <b>(${a.point.slice(0, 2).map(x => f(x, 4)).join('; ')})</b> м; если это кромка — <b>(${b.point.slice(0, 2).map(x => f(x, 4)).join('; ')})</b> м.</li></ol>
      <div class="muted">Для одного пикселя «стол» и «кромка» различаются на ${f(Math.hypot(a.point[0] - b.point[0], a.point[1] - b.point[1]) * 1000, 0)} мм, поэтому проём коробки мерится на высоте кромки. Ошибка калибровки ${f(a.reprojection_rmse_px, 2)} px.</div>`;
    math($('#cam-result'));
  });
}

/* ---------------------------------------------------------------- grasp / gravity at the newest plan's grasp pose */
async function graspScene(canvasId) {
  const sc = new Scene3D($(canvasId), { cam: [0.34, -0.3, 0.2], target: [0.2, -0.2, 0.05] }); await sc.addRobot({ opacity: 0.85 });
  PLAN = PLAN || await api('/api/plan').catch(() => null);
  if (!PLAN) return { none: true };
  const grasp = PLAN.plan.phases.find(p => p.name === 'grasp'); if (!grasp) return { none: true };
  const q = pose6(grasp.joints[0]), fk = await frames0(q); sc.setPose(fk.frames);
  const cube = PLAN.cubes[PLAN.plan.target_id]; sc.clearDynamic(); sceneWithObjects(sc, PLAN);
  const T = fk.frames.gripper_frame_link, col = i => [T[0][i], T[1][i], T[2][i]], o = [T[0][3], T[1][3], T[2][3]];
  return { sc, fk, T, col, o, cube, q };
}
async function initGrasp() {
  const G = await graspScene('#c-grasp'); if (G.none) { $('#grasp-info').innerHTML = '<div class="note">Покажется после первого плана.</div>'; return; }
  const { sc, T, col, o, cube, q } = G, x = col(0), y = col(1), z = col(2), c = cube.center_m, s = 0.03;
  const add = (p, a, k) => [p[0] + a[0] * k, p[1] + a[1] * k, p[2] + a[2] * k];
  sc.line([add(add(o, y, -s), z, -s), add(add(o, y, s), z, -s), add(add(o, y, s), z, s * 0.3), add(add(o, y, -s), z, s * 0.3), add(add(o, y, -s), z, -s)], 0xcbd5e1);
  sc.arrow(add(c, x, 0.05), add(c, x, -0.05), 0xfacc15, 0.014); sc.label(add(c, x, -0.05), 'ось смыкания', '#facc15', 0.024); sc.label(add(o, z, -0.05), 'неподвижная губка', '#cbd5e1', 0.022);
  sc.frameHelper('tcpg', 0.05, 'TCP'); sc.setFrame('tcpg', T); sc.setCamera([c[0] + 0.2, c[1] - 0.2, c[2] + 0.16], c);
  const local = [x, y, z].map(a => (c[0] - o[0]) * a[0] + (c[1] - o[1]) * a[1] + (c[2] - o[2]) * a[2]);
  $('#grasp-info').innerHTML = `<h3>Поза захвата из последнего плана</h3><table class="num">${JOINTS.map(j => `<tr><td>${JOINT_RU[j]}</td><td>${f(q[j], 1)}${j === 'gripper' ? '%' : '°'}</td></tr>`).join('')}</table>
    <p>Центр кубика в системе кончика: $x$ = <b>${f(local[0] * 1000, 1)} мм</b> (вдоль смыкания), $y$ = ${f(local[1] * 1000, 1)} мм, $z$ = ${f(local[2] * 1000, 1)} мм (вдоль пальцев). Половина стороны кубика ${f(cube.side_m / 2 * 1000, 1)} мм; разница — зазор до неподвижной губки.</p>
    <p class="muted">Серый прямоугольник — плоскость неподвижной губки, жёлтая стрелка — путь подвижной.</p>`;
  math($('#grasp-info'));
}
async function initGravity() {
  const G = await graspScene('#c-gravity'); if (G.none) { $('#gravity-info').innerHTML = '<div class="note">Покажется после первого плана.</div>'; return; }
  const { sc, o, q } = G; const s = await api('/api/sag');
  const cmd = { ...q }; ['shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'].forEach(j => cmd[j] = q[j] + s.command_minus_goal_deg[j]); const fkc = await frames0(cmd); const Tc = fkc.frames.gripper_frame_link, oc = [Tc[0][3], Tc[1][3], Tc[2][3]];
  const k = 12, exag = [o[0] + (oc[0] - o[0]) * k, o[1] + (oc[1] - o[1]) * k, o[2] + (oc[2] - o[2]) * k];
  sc.dot(o, 0xffffff, 0.005); sc.dot(oc, 0xf59e0b, 0.004); sc.arrow(o, exag, 0xf59e0b, 0.012); sc.label(exag, `сдвиг ×${k}`, '#fbbf24', 0.022);
  const names = JOINTS.slice(0, 5), links = ['shoulder_link', 'upper_arm_link', 'lower_arm_link', 'wrist_link', 'gripper_link'];
  names.forEach((n, i) => { const tq = s.torque_nm[i]; if (i && Math.abs(tq) > 1e-3) { const Tl = G.fk.frames[links[i]], p = [Tl[0][3], Tl[1][3], Tl[2][3]], ax = [Tl[0][2], Tl[1][2], Tl[2][2]], len = Math.min(0.12, Math.abs(tq) * 0.25) * Math.sign(tq); sc.arrow(p, [p[0] + ax[0] * len, p[1] + ax[1] * len, p[2] + ax[2] * len], 0xe879f9, 0.01); sc.label([p[0], p[1], p[2] + 0.02], `${f(tq, 2)} Н·м`, '#f0abfc', 0.02); } });
  sc.render(); sc.setCamera([0.45, -0.5, 0.3], [0.15, -0.1, 0.1]);
  $('#gravity-info').innerHTML = `<h3>Момент и поправка в позе захвата</h3><table class="num"><tr><th>сустав</th><th>момент, Н·м</th><th>цель, °</th><th>команда − цель, °</th></tr>${names.map((n, i) => `<tr><td>${JOINT_RU[n]}</td><td>${f(s.torque_nm[i], 3)}</td><td>${f(q[n], 1)}</td><td><b>${f(s.command_minus_goal_deg[n], 2)}</b></td></tr>`).join('')}</table>
    <p>Фиолетовые стрелки — моменты на суставах. Оранжевая — на сколько «заказанная» поза уезжает от плановой (в ${k} раз крупнее, иначе не видно): вдоль оси смыкания <b>${f(s.shift_in_gripper_frame_mm[0], 1)} мм</b>, вдоль пальцев ${f(s.shift_in_gripper_frame_mm[2], 1)} мм, поворот ${f(s.rotation_deg, 1)}°.</p>`;
}

/* ---------------------------------------------------------------- plan viewer (collisions + real plan) */
async function planViewer(canvasId, ctlId, onPose) {
  PLAN = PLAN || await api('/api/plan').catch(() => null);
  const sc = new Scene3D($(canvasId), { cam: [0.62, -0.62, 0.5], target: [0.2, -0.12, 0.06] }); await sc.addRobot({ opacity: 0.85 });
  if (!PLAN) { $(ctlId).innerHTML = '<div class="note">Планов пока нет: запустите работу на первой вкладке.</div>'; return null; }
  const id = ctlId.slice(1), phases = PLAN.plan.phases; let ph = phases.findIndex(p => p.name === 'transfer'); if (ph < 0) ph = 0; let idx = 0;
  $(ctlId).innerHTML = `<div class="sliders" style="grid-template-columns:70px 1fr;margin-top:8px"><label>фаза</label><select id="${id}-ph">${phases.map((p, i) => `<option value="${i}" ${i === ph ? 'selected' : ''}>${PHASE_RU[p.name] || p.name}</option>`).join('')}</select></div>
    <div class="sliders" style="grid-template-columns:70px 1fr 90px"><label>точка пути</label><input type="range" id="${id}-i" min="0" max="${phases[ph].joints.length - 1}" value="0"><output id="${id}-o"></output></div>
    <div><button id="${id}-play">▶ проиграть</button> <span class="muted">${PLAN.path} · ${PLAN.mode === 'stow' ? 'складывание в коробку' : PLAN.mode === 'extract' ? 'вытаскивание' : ''}</span></div>`;
  sc.clearDynamic(); sceneWithObjects(sc, PLAN);
  phases.forEach(p => { if (p.tips.length > 1) sc.line(p.tips, hex(PHASE_COLORS[p.name] || '#ffffff')); });
  sc.dot(PLAN.plan.placement_center_m, 0x22c55e, 0.008); sc.label([PLAN.plan.placement_center_m[0], PLAN.plan.placement_center_m[1], 0.04], 'место постановки', '#9f9', 0.022);
  const state = { sc, phases, get ph() { return ph; }, get idx() { return idx; } };
  let busy = false, again = false;
  const go = async () => {
    if (busy) { again = true; return; } busy = true;
    const P = phases[ph]; idx = Math.min(idx, P.joints.length - 1); $('#' + id + '-i').value = idx; $('#' + id + '-o').textContent = `${idx + 1} из ${P.joints.length}`;
    const q = pose6(P.joints[idx]); const fk = await frames0(q); sc.setPose(fk.frames); if (onPose) await onPose({ q, fk, phase: P, idx, state });
    busy = false; if (again) { again = false; go(); }
  };
  $('#' + id + '-ph').addEventListener('input', e => { ph = +e.target.value; idx = 0; $('#' + id + '-i').max = phases[ph].joints.length - 1; go(); });
  $('#' + id + '-i').addEventListener('input', e => { idx = +e.target.value; go(); });
  let timer = null; $('#' + id + '-play').addEventListener('click', e => {
    if (timer) { clearInterval(timer); timer = null; e.target.textContent = '▶ проиграть'; return; } e.target.textContent = '■ стоп';
    timer = setInterval(() => { const P = phases[ph]; if (idx < P.joints.length - 1) idx = Math.min(P.joints.length - 1, idx + 2); else if (ph < phases.length - 1) { ph++; idx = 0; $('#' + id + '-ph').value = ph; $('#' + id + '-i').max = phases[ph].joints.length - 1; } else { clearInterval(timer); timer = null; e.target.textContent = '▶ проиграть'; } go(); }, 160);
  });
  go(); return state;
}
async function initCollide() {
  const col = mm => mm === null || mm === undefined ? '#9aa5b4' : mm < 5 ? '#ef4444' : mm < 15 ? '#facc15' : '#22c55e';
  await planViewer('#c-collide', '#co-ctl', async ({ q, state }) => {
    const r = await api('/api/clearance', q);
    r.links.forEach(l => { const m = Math.min(...[l.table_mm, l.cube && l.cube.mm, l.box_mm].filter(v => v !== null && v !== undefined)); state.sc.tint(l.link, hex(col(m))); }); state.sc.render();
    $('#co-table').innerHTML = `<h3>Зазор каждого звена, мм</h3><table class="num"><tr><th>звено</th><th>до стола</th><th>до кубика</th><th>до стенок</th></tr>${r.links.filter(l => l.link !== 'base_link').map(l => `<tr><td>${FRAME_NAMES[l.link] || l.link}</td><td style="color:${col(l.table_mm)}">${f(l.table_mm, 0)}</td><td style="color:${col(l.cube && l.cube.mm)}">${l.cube ? f(l.cube.mm, 0) : '–'}</td><td style="color:${col(l.box_mm)}">${f(l.box_mm, 0)}</td></tr>`).join('')}</table>
      <div class="muted">Считается по вершинам настоящих деталей руки до кубиков, стенок коробки (толщина 10 мм) и стола. Звенья в 3D окрашены по самому тесному месту.</div>`;
  });
}

/* ---------------------------------------------------------------- search and smoothness */
function initSearch() {
  const demo = new RRTDemo($('#v-rrt'), $('#rrt-info')); $('#rrt-run').addEventListener('click', () => demo.run()); $('#rrt-new').addEventListener('click', () => { demo.running = false; demo.reset(); });
  const scale = { approach: 2.2, open: 2.0, transfer: 2.0, retract: 1.6, extract_vertical: 1.4, lower_to_support: 1.4, enter: 1.4, release: 1.6 };
  api('/api/plan').then(p => {
    const len = pts => pts.slice(1).reduce((s, q, i) => s + Math.hypot(q[0] - pts[i][0], q[1] - pts[i][1], q[2] - pts[i][2]), 0);
    $('#speed-table').innerHTML = `<table class="num"><tr><th>фаза</th><th>путь, см</th><th>скорость, мм/с</th><th>время, с</th></tr>${p.plan.phases.filter(x => x.tips.length > 1 && len(x.tips) > 0.002).map(x => { const L = len(x.tips), v = 6 * (scale[x.name] || 1.5) * 1.5 * 10 / 1.5; return `<tr><td style="color:${PHASE_COLORS[x.name]}">${PHASE_RU[x.name] || x.name}</td><td>${f(L * 100, 1)}</td><td>~${f(v, 0)}</td><td>~${f(1.875 * L * 1000 / v, 1)}</td></tr>`; }).join('')}</table>
      <div class="muted">Время $T\\approx1{,}875\\,L/v_\\text{пик}$: у профиля «минимального рывка» пиковая скорость в 1,875 раза выше средней. Скорости — ориентировочные, по настройке фаз.</div>`; math($('#speed-table'));
  }).catch(() => { });
}

/* ---------------------------------------------------------------- real plan */
async function initPlan() {
  const st = await planViewer('#c-plan', '#plan-ctl', ({ q, phase, idx, state }) => {
    const names = JOINTS.slice(0, 5); let cursor = 0, off = 0; const bands = [];
    state.phases.forEach((p, i) => { bands.push({ from: off, to: off + p.joints.length, color: PHASE_COLORS[p.name] || '#888888', label: PHASE_RU[p.name] || p.name }); if (i < state.ph) cursor += p.joints.length; off += p.joints.length; });
    cursor += idx; const series = names.map((n, k) => ({ name: JOINT_RU[n], color: JOINT_COLORS[k], y: state.phases.flatMap(p => p.joints.map(j => j[n])) }));
    lineChart($('#ch-plan'), series, { bands, cursor, xLabel: 'точка плана' }); $('#lg-plan').innerHTML = legend(series);
    const tip = phase.tips[Math.min(idx, phase.tips.length - 1)] || [0, 0, 0];
    $('#plan-result').innerHTML = `<h3>${PHASE_RU[phase.name] || phase.name}, точка ${idx + 1} из ${phase.joints.length}</h3><table class="num">${JOINTS.map(j => `<tr><td>${JOINT_RU[j]}</td><td>${f(q[j], 1)}${j === 'gripper' ? '%' : '°'}</td></tr>`).join('')}</table><div>Кончик: (${tip.map(v => f(v, 3)).join('; ')}) м · ${phase.carrying ? 'кубик в руке' : 'рука пустая'}</div>
      <h3>Поиск пути в этом плане</h3>${PLAN.plan.searches.map((s, i) => `<div>Поиск ${i + 1}: ${s.algorithm || 'RRT-Connect'}, ${s.direct ? 'прямой путь был свободен — дерево не понадобилось' : `итераций ${s.iterations}, рёбер ${(s.edges || []).length}`}</div>`).join('') || '<div class="muted">нет данных</div>'}`;
  });
  if (st) $('#plan-legend').innerHTML = st.phases.map(p => `<span><i style="background:${PHASE_COLORS[p.name]}"></i>${PHASE_RU[p.name] || p.name}</span>`).join('');
}

async function initLearn() {
  build();
  try { ROBOT = await api('/api/robot'); } catch (e) { $('#content').insertAdjacentHTML('afterbegin', `<div class="note">Модель робота недоступна: ${e}</div>`); return; }
  await Scene3D.loadUrdf();
  initLive();
  [['frames', initFrames], ['rot', initRot], ['fk', initFK], ['ik', initIK], ['camera', initCamera], ['grasp', initGrasp], ['gravity', initGravity], ['collide', initCollide], ['search', initSearch], ['plan', initPlan]]
    .forEach(([id, fn]) => lazy($('#s-' + id), fn));
}
window.initLearn = initLearn;
