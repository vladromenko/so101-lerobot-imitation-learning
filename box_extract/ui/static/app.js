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
    if(!sceneData||name!=='overview')return;
    const k=rect.width/(img.naturalWidth||1280);g.lineWidth=2;g.font='bold 13px sans-serif';
    function polygon(points,color){g.strokeStyle=color;g.beginPath();points.forEach((p,i)=>i?g.lineTo(p[0]*k,p[1]*k):g.moveTo(p[0]*k,p[1]*k));g.closePath();g.stroke();}
    sceneData.boxes.forEach(b=>{polygon(b.polygon,b.metric_qualified?'#22c55e':'#f59e0b');const p=b.polygon[0];g.fillStyle='#f59e0b';g.fillText(b.id+' · '+(b.metric_qualified?'измерена':'кандидат'),p[0]*k,p[1]*k-5);});
    sceneData.cubes.forEach(o=>{const selected=o.id===statusData?.selected,color=selected?'#fff':'#38bdf8';o.polygon.forEach(p=>polygon(p,color));g.fillStyle=color;g.fillText(o.id+' '+o.color,o.px[0]*k-35,o.px[1]*k-25);});
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
  if(!sceneData.observation_valid)document.getElementById('scene-hint').textContent=sceneData.observation_error||'Нет достоверного свежего наблюдения';
  drawOverlay();
}
async function start(mode){
  try{await api('/api/run',{mode,dry:document.getElementById('dry').checked,destination:document.getElementById('destination').value});}
  catch(err){error(err);}await poll();
}
['relocate','stow','extract'].forEach(id=>document.getElementById(id).addEventListener('click',()=>start(id)));
document.getElementById('scene-btn').addEventListener('click',()=>refreshScene().catch(error));
document.getElementById('stop').addEventListener('click',async()=>{await api('/api/stop',{});await poll();});
document.getElementById('reset').addEventListener('click',async()=>{try{await api('/api/reset',{});}catch(err){error(err);}await poll();});
document.getElementById('pause').addEventListener('click',async()=>{await api('/api/pause',{paused:!statusData?.paused});await poll();});
document.getElementById('dry').addEventListener('change',gate);
function gate(){
  const dry=document.getElementById('dry').checked;
  ['relocate','stow','extract'].forEach(id=>document.getElementById(id).disabled=lastRunning||statusData?.stop_latched||(!dry&&!statusData?.ready));
}
async function poll(){
  try{
    const s=await api('/api/status');statusData=s;lastRunning=s.running;gate();
    document.getElementById('state-text').textContent=s.ready?'Ready':({blocked:'Движение заблокировано',stopped:'STOP',checking:'Проверка без движения',initializing:'Инициализация',completed:'Завершено'}[s.phase]||s.phase);
    document.getElementById('state-dot').className='dot '+(s.running?'run':s.ready?'ok':'bad');
    document.getElementById('state-detail').textContent=s.detail;
    document.getElementById('blockers').textContent=s.blockers.join(' · ');
    document.getElementById('pause').textContent=s.paused?'Resume':'Pause';
    const m=s.metrics;document.getElementById('metrics').textContent='Vision: '+(m.vision_s===undefined?'—':(1000*m.vision_s).toFixed(0)+' мс')+' · Start → первая команда: '+(m.first_command_s==null?'команд не было':m.first_command_s.toFixed(3)+' с');
    document.getElementById('log').textContent=s.log.join('\n');await refreshScene();
  }catch(err){error(err);}
}
setInterval(()=>feeds.forEach(camera),650);setInterval(poll,1000);
feeds.forEach(camera);poll();window.addEventListener('resize',drawOverlay);
