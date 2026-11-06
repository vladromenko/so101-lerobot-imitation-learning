// The existing work panel now talks only to the persistent Wristlab runtime.
let sceneData=null, statusData=null, lastRunning=false;
const feeds = ['overview','wrist'];
const imageIds={overview:'cam',wrist:'wrist-cam'}, canvasIds={overview:'overlay',wrist:'wrist-overlay'};
const displayed={}, loading={};
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===b.dataset.tab));
  if(b.dataset.tab==='learn'&&!window._learnStarted){window._learnStarted=true;window.initLearn();}
}));
async function api(path,body){
  const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await r.json();if(!r.ok)throw Error(data.detail||r.statusText);return data;
}
function error(err){document.getElementById('scene-hint').textContent=String(err.message||err);}
async function camera(name){
  if(loading[name]||!document.getElementById('work').classList.contains('active'))return;
  loading[name]=true;
  try{
    const response=await fetch('/snapshot/'+name+'.jpg?t='+Date.now());if(!response.ok)throw Error(name+': нет свежего кадра');
    const blob=await response.blob(), url=URL.createObjectURL(blob), img=document.getElementById(imageIds[name]);
    img.onload=()=>{if(displayed[name]?.url)URL.revokeObjectURL(displayed[name].url);displayed[name]={url,frameId:response.headers.get('X-Frame-Id')};drawOverlay();};
    img.src=url;
  }catch(err){error(err);}finally{loading[name]=false;}
}
function drawOverlay(){
  feeds.forEach(name=>{
    const img=document.getElementById(imageIds[name]),canvas=document.getElementById(canvasIds[name]),rect=img.getBoundingClientRect();
    canvas.width=rect.width;canvas.height=rect.height;
    const g=canvas.getContext('2d');g.clearRect(0,0,canvas.width,canvas.height);
    if(!sceneData)return;
    const k=rect.width/(img.naturalWidth||1280);g.lineWidth=2;g.font='bold 13px sans-serif';
    function polygon(points,color){g.strokeStyle=color;g.beginPath();points.forEach((p,i)=>i?g.lineTo(p[0]*k,p[1]*k):g.moveTo(p[0]*k,p[1]*k));g.closePath();g.stroke();}
    if(name==='overview')sceneData.boxes.forEach(b=>{polygon(b.polygon,b.metric_qualified?'#22c55e':'#f59e0b');const p=b.polygon[0];g.fillStyle='#f59e0b';g.fillText(b.id+' · '+(b.metric_qualified?'измерена':'кандидат'),p[0]*k,p[1]*k-5);});
    (name==='overview'?sceneData.cubes:sceneData.wrist_objects||[]).forEach(o=>{const selected=o.id===statusData?.selected,color=selected?'#fff':'#38bdf8';g.setLineDash(o.status==='occluded'?[5,4]:[]);o.polygon.forEach(p=>polygon(p,color));g.fillStyle=color;g.fillText(o.id+' '+o.color+(o.status==='occluded'?' · скрыт':''),o.px[0]*k-35,o.px[1]*k-25);});
    g.setLineDash([]);
  });
}
feeds.forEach(name=>document.getElementById(canvasIds[name]).addEventListener('click',async e=>{
  const img=document.getElementById(imageIds[name]),rect=e.currentTarget.getBoundingClientRect();
  try{
    const selected=await api('/api/select',{camera_id:name,frame_id:displayed[name]?.frameId,pixel_x:(e.clientX-rect.left)*img.naturalWidth/rect.width,pixel_y:(e.clientY-rect.top)*img.naturalHeight/rect.height});
    document.getElementById('selected').textContent=selected.object_id+' · '+selected.color;await poll();
  }catch(err){error(err);}
}));
async function refreshScene(){
  sceneData=await api('/api/scene');
  document.getElementById('n-out').textContent=sceneData.outside??'—';
  document.getElementById('n-in').textContent=sceneData.inside??'—';
  const destination=document.getElementById('destination'),previous=destination.value;
  const ids=sceneData.boxes.map(b=>b.id);
  if(Array.from(destination.options).map(o=>o.value).join()!==ids.join()){
    destination.replaceChildren(...sceneData.boxes.map(b=>{const o=document.createElement('option');o.value=b.id;o.textContent=b.id+(b.metric_qualified?'':' · не квалифицирована');return o;}));
    if(ids.includes(previous))destination.value=previous;
  }
  const hint=document.getElementById('scene-hint');
  hint.textContent=!sceneData.video_fresh?'Видеокадр устарел; показаны последние данные':
    sceneData.detection_error?'Видео обновляется. Ошибка обработки сцены: '+sceneData.detection_error:
    !sceneData.detection_fresh?'Видео обновляется; ожидается обновление детекции':
    'Кадры свежие · обзор: '+sceneData.visible_overview_count+' видно, '+sceneData.occluded_count+' скрыто рукой · wrist: '+sceneData.visible_wrist_count+' видно';
  hint.title='Обзор: '+sceneData.evidence.overview_frame_id+'; wrist: '+sceneData.evidence.wrist_frame_id+'; SceneState #'+sceneData.version;
  drawOverlay();
}
async function start(mode){
  try{await api('/api/run',{mode,dry:document.getElementById('dry').checked,destination:document.getElementById('destination').value,object_id:statusData?.selected,count:2,rows:2,columns:2,pattern:mode==='build_color_pattern'?[['green','yellow'],['red','blue']]:undefined});}
  catch(err){error(err);}await poll();
}
['relocate','stow','extract'].forEach(id=>document.getElementById(id).addEventListener('click',()=>start(id)));
document.getElementById('scene-btn').addEventListener('click',()=>refreshScene().catch(error));
document.getElementById('stop').addEventListener('click',async()=>{await api('/api/stop',{});await poll();});
document.getElementById('reset').addEventListener('click',async()=>{try{await api('/api/reset',{});}catch(err){error(err);}await poll();});
document.getElementById('pause').addEventListener('click',async()=>{await api('/api/pause',{paused:!statusData?.paused});await poll();});
document.getElementById('park').addEventListener('click',async()=>{try{await api('/api/park',{dry:true});}catch(err){error(err);}await poll();});
document.getElementById('park-run').addEventListener('click',async()=>{
  if(confirm('Выполнить показанный Park? Я наблюдаю руку, камера и кабель свободны, STOP доступен.')){
    try{await api('/api/park',{dry:false,operator_at_stop:true});}catch(err){error(err);}await poll();
  }
});
document.getElementById('experiment-start').addEventListener('click',()=>start(document.getElementById('experiment').value));
document.getElementById('task-plan').addEventListener('click',async()=>{
  const body={mode:document.getElementById('experiment').value,destination:document.getElementById('destination').value,object_id:statusData?.selected,count:2,rows:2,columns:2};
  if(body.mode==='build_color_pattern')body.pattern=[['green','yellow'],['red','blue']];
  try{const plan=await api('/api/task-plan',body);const out=document.getElementById('task-proposal');out.hidden=false;out.textContent=JSON.stringify(plan,null,2);}catch(err){error(err);}
});
document.getElementById('dry').addEventListener('change',gate);
function gate(){
  const dry=document.getElementById('dry').checked;
  ['relocate','stow','extract','experiment-start'].forEach(id=>document.getElementById(id).disabled=lastRunning||statusData?.stop_latched||(!dry&&!statusData?.ready));
  document.getElementById('park').disabled=lastRunning||statusData?.stop_latched;
  document.getElementById('park-run').disabled=lastRunning||statusData?.stop_latched||!statusData?.results?.some(r=>r.status==='PARK_CHECKED');
}
async function poll(){
  try{
    const s=await api('/api/status');statusData=s;lastRunning=s.running;gate();
    document.getElementById('state-text').textContent=({NOT_INITIALIZED:'Инициализация',READY:'Ready',PARKED:'PARKED',PLANNING:'Планирование',EXECUTING:'Движение',RECOVERING:'Восстановление',SAFETY_BLOCKED:'Движение заблокировано',COMPLETED:'Завершено',FAILED:'Задача не выполнена'}[s.state]||s.phase);
    document.getElementById('state-dot').className='dot '+(s.running?'run':s.ready?'ok':'bad');
    document.getElementById('state-detail').textContent=s.detail;
    document.getElementById('blockers').textContent=s.physical_blocker?
      'Программный запас исходной позы: выборка поверхности '+(1000*s.physical_blocker.clearance_m).toFixed(1)+' мм при требуемых '+(1000*s.physical_blocker.required_clearance_m).toFixed(0)+' мм. '+s.physical_blocker.required_action+' '+s.physical_blocker.recheck:
      s.blockers.join(' · ');
    const qualification=document.getElementById('qualification');qualification.style.whiteSpace='pre-line';
    qualification.textContent=(s.known_working_configuration?'Калибровка суставов и FK сохранены от физически работавшей системы. Драйвер '+s.startup_driver+'; режим '+s.grasp_mode+'.\n\n':'')+
      (s.blocking_conditions||[]).map(b=>b.code+' ['+b.category+']: '+b.reason+'\nСнятие: '+b.clear_when).join('\n\n');
    document.getElementById('action-result').textContent=s.results.length?
      s.results.map(r=>(r.success?'Успех: ':'Не выполнено: ')+(r.reason||r.status)+' · физически подтверждено: '+(r.evidence?.physical_verified||r.physical_verified?'да':'нет')).join('\n'):
      (s.physical_progress?.gripper_release_completed?'Захват, перенос зелёного кубика в белую коробку и отпускание физически выполнены. Retreat и Park ещё не завершены.':'Физически подтверждённых переносов пока нет.');
    const motors=Object.entries(s.motor_health||{});
    document.getElementById('motor-health').textContent=motors.map(([name,m])=>name+': '+m.temperature_c+' °C · '+(s.motor_loads?.[name]??0).toFixed(1)+'% · '+m.current_ma.toFixed(1)+' мА · '+m.voltage_v.toFixed(1)+' В · torque '+(m.torque_enabled?'ON':'OFF')).join(' | ');
    document.getElementById('motor-warnings').textContent=(s.motor_warnings||[]).join(' · ')+(s.park_error?' · Park: '+s.park_error:'');
    document.getElementById('pause').textContent=s.paused?'Resume':'Pause';
    const m=s.metrics;document.getElementById('metrics').textContent=[['Start → команда','first_command_s'],['Vision','vision_s'],['Захват','grasp_planning_s'],['IK','ik_s'],['Коллизии','collision_checking_s'],['Путь','motion_planning_s'],['Движение','actual_motion_s'],['Проверка','verification_s']].map(([label,key])=>label+': '+(m[key]==null?'не выполнялось':(1000*m[key]).toFixed(1)+' мс')).join(' · ');
    document.getElementById('log').textContent=s.log.join('\n');
    const live=await api('/api/live');document.getElementById('joints').textContent=live.pos?'Суставы · '+Object.entries(live.pos).map(([k,v])=>k+': '+v.toFixed(1)).join(' · '):'Нет свежей обратной связи';
    await refreshScene();
  }catch(err){error(err);}
}
setInterval(()=>feeds.forEach(camera),650);setInterval(poll,1000);
feeds.forEach(camera);poll();window.addEventListener('resize',drawOverlay);
