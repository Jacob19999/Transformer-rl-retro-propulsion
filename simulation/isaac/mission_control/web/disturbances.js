import {escapeHtml} from './flight-plan.js';

const noiseFields=[['position_std','Position','m',.5,.001],['velocity_std','Velocity','m/s',2,.01],['attitude_std','Attitude','rad',.1,.001],['angular_velocity_std','Body rate','rad/s',1,.01]];
export function createDisturbanceEditor(root,{onChange,summary}){
  let base={},settings={},ready=false;
  const controls=(path,label,unit,min,max,step,factor=1)=>`<label class="disturbance-control"><span>${label}<small>${unit}</small></span><div><input type="range" data-path="${path}" data-factor="${factor}" min="${min}" max="${max}" step="${step}" aria-label="${label} slider"><input type="number" data-path="${path}" data-factor="${factor}" min="${min}" max="${max}" step="any" required aria-label="${label}"></div></label>`;
  const title=(key,name,detail)=>`<div class="disturbance-card-heading"><label><input type="checkbox" name="disturbance" value="${key}"> ${name}</label><span>${detail}</span></div>`;
  root.innerHTML=`<div class="disturbance-heading"><div><div class="eyebrow">MISSION ENVIRONMENT</div><h3>Design your disturbances</h3><p>Set the airflow, measurement uncertainty and mass offset for this flight.</p></div><button type="button" class="reset-disturbances">RESET PARAMETERS</button></div>
    <div class="disturbance-cards">
      <section class="disturbance-card" data-source="wind">${title('wind','Wind + gusts','WORLD XYZ · Z UP')}<div class="wind-design"><svg class="wind-compass" viewBox="0 0 240 240" role="img" aria-label="Wind vector editor; drag to set horizontal speed and direction"><circle cx="120" cy="120" r="88"/><circle cx="120" cy="120" r="44" class="compass-inner"/><path d="M25 120 H215 M120 25 V215"/><text x="195" y="112">+X</text><text x="129" y="32">+Y</text><text x="28" y="112">−X</text><text x="129" y="212">−Y</text><path class="wind-arrow"/><circle class="wind-tip" r="6"/><circle cx="120" cy="120" r="3" class="wind-origin"/></svg><div><div class="wind-readout"></div><p>Drag the arrow tip.<br>Direction is where air travels, measured from +X toward +Y.</p><span class="wind-vector"></span></div></div>
      ${controls('windSpeed','Horizontal speed','m/s',0,15,.05)}${controls('windHeading','Flow direction','°',0,360,.1)}${controls('wind.steady_vector.2','Vertical airflow','m/s · +up',-15,15,.05)}
      <details class="gust-controls" open><summary>Random horizontal gusts</summary>${controls('gust.magnitude','Gust magnitude','m/s',0,15,.1)}${controls('gust.duration','Gust duration','s',.05,10,.05)}<div class="disturbance-pair">${controls('gust.interval.0','Minimum wait','s',.1,120,.1)}${controls('gust.interval.1','Maximum wait','s',.1,120,.1)}</div><p class="hint">Random direction per gust; wait is sampled between gusts. Seed controls the random realization.</p></details></section>
      <section class="disturbance-card" data-source="sensor_noise">${title('sensor_noise','Sensor noise','GAUSSIAN · 1σ')}<div class="disturbance-illustration noise-visual"></div><p class="disturbance-description">Independent observation noise. Each slider sets one standard deviation; physical states are unchanged.</p>${noiseFields.map(([key,label,unit,max,step])=>controls(`sensor_noise.${key}`,`${label} noise`,unit,0,max,step)).join('')}</section>
      <section class="disturbance-card" data-source="com_shift">${title('com_shift','Center of mass','BODY FRD · mm')}<div class="disturbance-illustration com-visual"></div><p class="disturbance-description">Offset sampled uniformly at reset within this box. Body X forward, Y right, Z down; equal bounds fix an axis.</p>${['X','Y','Z'].map((axis,i)=>`<div class="com-axis"><b>${axis} OFFSET</b><div class="disturbance-pair">${controls(`com_offset.range.0.${i}`,`${axis} minimum`,'mm',-50,50,.1,1000)}${controls(`com_offset.range.1.${i}`,`${axis} maximum`,'mm',-50,50,.1,1000)}</div></div>`).join('')}</section>
    </div><p class="hint disturbance-note">Enable each source to apply it. Visuals show configured vectors and distributions, not a forecast of the sampled run.</p>`;
  const input=path=>root.querySelector(`input[type=number][data-path="${path}"]`);
  function readPath(path){return path.split('.').reduce((v,k)=>v?.[k],settings);}
  function setPath(path,value){const keys=path.split('.'),last=keys.pop();keys.reduce((v,k)=>v[k],settings)[last]=value;}
  function wind(){const [x,y]=settings.wind.steady_vector;return {speed:Math.hypot(x,y),heading:(Math.atan2(y,x)*180/Math.PI+360)%360};}
  function sync(){
    const w=wind();root.querySelectorAll('[data-path]').forEach(el=>{const p=el.dataset.path,v=p==='windSpeed'?w.speed:p==='windHeading'?w.heading:readPath(p)*Number(el.dataset.factor);el.value=Number(v.toFixed(p==='windSpeed'||p==='windHeading'?2:6));});
  }
  function validatePairs(){
    root.querySelectorAll('input[type=number]').forEach(el=>el.setCustomValidity(''));
    if(settings.gust.interval[0]>settings.gust.interval[1])input('gust.interval.0').setCustomValidity('Minimum wait must be at most the maximum wait.');
    for(let i=0;i<3;i++)if(settings.com_offset.range[0][i]>settings.com_offset.range[1][i])input(`com_offset.range.0.${i}`).setCustomValidity('Minimum offset must be at most the maximum offset.');
  }
  function draw(){
    const chosen=[...root.querySelectorAll('input[name=disturbance]:checked')].map(el=>el.value);
    root.querySelectorAll('[data-source]').forEach(el=>el.classList.toggle('is-active',chosen.includes(el.dataset.source)));
    summary.textContent=chosen.length?chosen.map(k=>({wind:'Wind + gusts',sensor_noise:'Sensor noise',com_shift:'COM offset'}[k])).join(' · '):'Calm air · no sensor noise or COM offset';
    if(!ready)return;
    const [x,y,z]=settings.wind.steady_vector,w=wind(),cx=120+x/15*88,cy=120-y/15*88;
    root.querySelector('.wind-arrow').setAttribute('d',`M120 120 L${cx} ${cy}`);root.querySelector('.wind-tip').setAttribute('cx',cx);root.querySelector('.wind-tip').setAttribute('cy',cy);
    root.querySelector('.wind-readout').textContent=`${w.speed.toFixed(2)} m/s · ${w.heading.toFixed(1)}°`;
    root.querySelector('.wind-vector').textContent=`XYZ [${[x,y,z].map(v=>v.toFixed(2)).join(', ')}] m/s`;
    const sigma=settings.sensor_noise.position_std,width=Math.max(2,sigma*160),points=Array.from({length:81},(_,i)=>{const px=20+i*2.5,py=125-90*Math.exp(-.5*((px-120)/width)**2);return `${px},${py}`;}).join(' ');
    root.querySelector('.noise-visual').innerHTML=`<svg viewBox="0 0 240 155" role="img" aria-label="Position noise distribution with standard deviation ${sigma} meters"><path class="diagram-axis" d="M20 125 H220"/><polyline points="${points}"/><path class="diagram-axis" d="M120 20 V125"/><text x="25" y="148">POSITION σ = ${escapeHtml(sigma)} m</text></svg>`;
    const [low,high]=settings.com_offset.range,px=v=>120+v*1600,py=v=>77-v*1000;
    root.querySelector('.com-visual').innerHTML=`<svg viewBox="0 0 240 155" role="img" aria-label="Body X Y center of mass offset sampling bounds"><path class="diagram-axis" d="M20 77 H220 M120 15 V140"/><rect x="${px(low[0])}" y="${py(high[1])}" width="${Math.max(1,(high[0]-low[0])*1600)}" height="${Math.max(1,(high[1]-low[1])*1000)}"/><circle cx="120" cy="77" r="4"/><text x="195" y="69">+X</text><text x="129" y="24">+Y</text><text x="25" y="150">XY RANGE · Z SET BELOW</text></svg>`;
    validatePairs();
  }
  root.querySelectorAll('[data-path]').forEach(el=>el.oninput=()=>{
    if(!ready||el.value==='')return;
    el.setCustomValidity('');if(!el.checkValidity())return;
    const path=el.dataset.path,v=Number(el.value),w=wind();
    if(path==='windSpeed'||path==='windHeading'){const speed=path==='windSpeed'?v:w.speed,heading=(path==='windHeading'?v:w.heading)*Math.PI/180;settings.wind.steady_vector[0]=speed*Math.cos(heading);settings.wind.steady_vector[1]=speed*Math.sin(heading);}
    else setPath(path,v/Number(el.dataset.factor));
    root.querySelectorAll(`[data-path="${path}"]`).forEach(peer=>{if(peer!==el)peer.value=v;});draw();onChange?.();
  });
  root.querySelectorAll('input[name=disturbance]').forEach(el=>el.onchange=()=>{draw();onChange?.();});
  root.addEventListener('invalid',e=>{const details=e.target.closest('details');if(details)details.open=true;},true);
  const compass=root.querySelector('.wind-compass');let dragging=false;
  function drag(e){if(!ready)return;const rect=compass.getBoundingClientRect(),scale=Math.min(rect.width,rect.height)/240;let x=(e.clientX-rect.left-rect.width/2)/scale/88*15,y=-(e.clientY-rect.top-rect.height/2)/scale/88*15;const length=Math.hypot(x,y);if(length>15){x*=15/length;y*=15/length;}settings.wind.steady_vector[0]=x;settings.wind.steady_vector[1]=y;sync();draw();onChange?.();}
  compass.onpointerdown=e=>{if(e.button!==0)return;dragging=true;compass.setPointerCapture(e.pointerId);drag(e);};compass.onpointermove=e=>{if(dragging)drag(e);};compass.onpointerup=compass.onpointercancel=compass.onlostpointercapture=()=>{dragging=false;};
  root.querySelector('.reset-disturbances').onclick=()=>{if(!ready)return;settings=structuredClone(base);sync();draw();onChange?.();};
  return {
    // An already-running older service still supports the three preset toggles.
    configure(defaults){base=structuredClone(defaults);settings=structuredClone(base);ready=!!settings.wind;root.classList.toggle('presets-only',!ready);root.querySelectorAll('[data-path]').forEach(el=>{el.disabled=!ready;});root.querySelector('.reset-disturbances').disabled=!ready;if(ready)sync();else root.querySelector('.disturbance-heading p').textContent='Preset toggles are available. Restart the mission-control service to customize parameters.';draw();},
    set(selected,overrides={}){root.querySelectorAll('input[name=disturbance]').forEach(el=>{el.checked=selected.includes(el.value);});if(ready){settings=structuredClone(base);for(const [section,entries] of Object.entries(overrides))Object.assign(settings[section],structuredClone(entries));sync();}draw();},
    get(){return ready?structuredClone(settings):{};},
    preview(){return ready?{selected:[...root.querySelectorAll('input[name=disturbance]:checked')].map(el=>el.value),settings:structuredClone(settings)}:null;},
  };
}
