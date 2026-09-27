import {escapeHtml} from './flight-plan.js';

const noiseFields=[['position_std','Position','m',.5,.001],['velocity_std','Velocity','m/s',2,.01],['attitude_std','Attitude','rad',.1,.001],['angular_velocity_std','Body rate','rad/s',1,.01]];
const sources=[['wind','Wind + gusts','Airflow in world XYZ'],['sensor_noise','Sensor noise','Measurement uncertainty'],['com_shift','Center of mass','Body-frame offset']];
const sourceNames={wind:'Wind + gusts',sensor_noise:'Sensor noise',com_shift:'COM offset'};
const presetIcons={calm:'◯',breeze:'≈',moderate:'≋','gusty-crosswind':'⇶','noisy-sensors':'∿','com-offset':'⊕',stress:'⚠'};
// Descriptive bands for steady wind speed (m/s), roughly the Beaufort scale.
const windBand=speed=>speed<.5?'calm':speed<3.4?'light':speed<8?'moderate':speed<10.8?'fresh':'strong';

export function createDisturbanceEditor(root,{onChange,api}){
  let base={},settings={},ready=false,presets=[];
  const controls=(path,label,unit,min,max,step,factor=1)=>`<label class="disturbance-control"><span>${label}<small>${unit}</small></span><div><input type="range" data-path="${path}" data-factor="${factor}" min="${min}" max="${max}" step="${step}" aria-label="${label} slider"><input type="number" data-path="${path}" data-factor="${factor}" min="${min}" max="${max}" step="any" required aria-label="${label}"></div></label>`;
  const title=(key,name,detail)=>`<div class="disturbance-card-heading"><label class="switch"><input type="checkbox" name="disturbance" value="${key}"><span class="switch-track" aria-hidden="true"></span><span class="switch-label">Apply ${name.toLowerCase()} to this flight</span></label><span>${detail}</span></div><p class="source-off-note">Off · these values are kept but not flown. Switch on to apply them.</p>`;
  root.innerHTML=`<summary class="config-summary section-head"><span class="section-index">03</span><div class="section-title"><h2>Environment</h2><p>Wind and gusts, sensor noise and centre-of-mass offset for this flight.</p></div><span id="disturbanceSummary" class="summary-chips"></span><span class="fold-chevron" aria-hidden="true"></span></summary><div class="config-body">
    <div class="block-head"><div><h3>Conditions</h3><p>Start from a preset, then fine-tune each source. Presets set the sources and their values together.</p></div><button type="button" class="reset-disturbances">Reset parameters</button></div>
    <div class="env-presets" role="group" aria-label="Environment presets"></div>
    <div class="environment-tabs" role="tablist" aria-label="Disturbance source"></div><div class="disturbance-cards">
      <section class="disturbance-card" data-source="wind">${title('wind','Wind + gusts','WORLD XYZ · Z UP')}<div class="wind-design"><svg class="wind-compass" viewBox="0 0 240 240" role="img" aria-label="Wind vector editor; drag to set horizontal speed and direction"><circle cx="120" cy="120" r="88"/><circle cx="120" cy="120" r="44" class="compass-inner"/><path d="M25 120 H215 M120 25 V215"/><text x="195" y="112">+X</text><text x="129" y="32">+Y</text><text x="28" y="112">−X</text><text x="129" y="212">−Y</text><text x="168" y="164" class="compass-scale">7.5</text><text x="198" y="194" class="compass-scale">15 m/s</text><path class="wind-arrow"/><circle class="wind-tip" r="7"/><circle cx="120" cy="120" r="3" class="wind-origin"/></svg><div><div class="wind-readout"></div><div class="wind-band"></div><p>Drag the arrow tip, or use the sliders.<br>Direction is where the air travels, measured from +X toward +Y.</p><span class="wind-vector"></span></div></div>
      <div class="wind-sliders">${controls('windSpeed','Horizontal speed','m/s',0,15,.05)}${controls('windHeading','Flow direction','°',0,360,.1)}${controls('wind.steady_vector.2','Vertical airflow','m/s · +up',-15,15,.05)}</div>
      <div class="gust-controls"><h4>Random horizontal gusts</h4><div class="gust-timeline"></div><div class="gust-grid">${controls('gust.magnitude','Gust magnitude','m/s',0,15,.1)}${controls('gust.duration','Gust duration','s',.05,10,.05)}${controls('gust.interval.0','Minimum wait','s',.1,120,.1)}${controls('gust.interval.1','Maximum wait','s',.1,120,.1)}</div><p class="hint">Random direction per gust; the wait between gusts is sampled uniformly between the minimum and maximum. The seed fixes the realization.</p></div></section>
      <section class="disturbance-card" data-source="sensor_noise">${title('sensor_noise','Sensor noise','GAUSSIAN · 1σ')}<div class="disturbance-illustration noise-visual"></div><div class="disturbance-fields"><p class="disturbance-description">Independent observation noise. Each slider sets one standard deviation; physical states are unchanged.</p>${noiseFields.map(([key,label,unit,max,step])=>controls(`sensor_noise.${key}`,`${label} noise`,unit,0,max,step)).join('')}</div></section>
      <section class="disturbance-card" data-source="com_shift">${title('com_shift','Center of mass','BODY FRD · mm')}<div class="disturbance-illustration com-visual"></div><div class="disturbance-fields"><p class="disturbance-description">Offset sampled uniformly at reset within this box. Body X forward, Y right, Z down; equal bounds fix an axis.</p>${['X','Y','Z'].map((axis,i)=>`<div class="com-axis"><b>${axis} OFFSET</b><div class="disturbance-pair">${controls(`com_offset.range.0.${i}`,`${axis} minimum`,'mm',-50,50,.1,1000)}${controls(`com_offset.range.1.${i}`,`${axis} maximum`,'mm',-50,50,.1,1000)}</div></div>`).join('')}</div></section>
    </div><p class="hint disturbance-note">Visuals show the configured vectors and distributions, not a forecast of the sampled run.</p></div>`;
  const summary=root.querySelector('#disturbanceSummary');
  let activeSource='wind';
  const tabs=root.querySelector('.environment-tabs');
  tabs.innerHTML=sources.map(([key,name,detail])=>`<button type="button" role="tab" id="environment-tab-${key}" aria-controls="environment-panel-${key}" data-source-tab="${key}"><b>${name}</b><small>${detail}</small><span class="source-state"></span></button>`).join('');
  function selectSource(key,focus=false){
    activeSource=key;
    tabs.querySelectorAll('button').forEach(button=>{const active=button.dataset.sourceTab===key;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;if(active&&focus)button.focus();});
    root.querySelectorAll('.disturbance-card').forEach(panel=>{panel.hidden=panel.dataset.source!==key;});
  }
  sources.forEach(([key])=>{const panel=root.querySelector(`[data-source="${key}"]`);panel.id=`environment-panel-${key}`;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',`environment-tab-${key}`);});
  tabs.querySelectorAll('button').forEach(button=>{
    button.onclick=()=>selectSource(button.dataset.sourceTab);
    button.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();e.stopPropagation();const i=sources.findIndex(([key])=>key===activeSource);selectSource(sources[e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowLeft'?2:1))%3][0],true);};
  });
  selectSource(activeSource);
  const input=path=>root.querySelector(`input[type=number][data-path="${path}"]`);
  const chosen=()=>[...root.querySelectorAll('input[name=disturbance]:checked')].map(el=>el.value).sort();
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
  // A preset matches when it selects the same sources with the same values.
  const near=(a,b)=>Array.isArray(a)?Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>near(v,b[i])):typeof a==='number'?Math.abs(a-b)<1e-6:a===b;
  function matches(preset){
    const selected=chosen();
    if(!near(preset.selected,selected))return false;
    if(!ready)return true;
    const expected=structuredClone(base);for(const [section,entries] of Object.entries(preset.settings))Object.assign(expected[section],entries);
    const sections={wind:['wind','gust'],sensor_noise:['sensor_noise'],com_shift:['com_offset']};
    return selected.flatMap(s=>sections[s]).every(section=>Object.keys(expected[section]).every(key=>near(expected[section][key],settings[section][key])));
  }
  const chip=(v,k,cls='')=>`<span class="chip ${cls}"><b>${escapeHtml(v)}</b>${escapeHtml(k)}</span>`;
  function describe(){
    const selected=chosen(),preset=presets.find(matches);
    if(!selected.length)return {text:'Calm air',chips:[chip('Calm','air')],preset};
    const parts=[];
    if(selected.includes('wind')){const w=ready?wind():null;parts.push(w?chip(`${w.speed.toFixed(1)} m/s`,`wind → ${w.heading.toFixed(0)}°`):chip('Wind','on'));if(ready&&settings.gust.magnitude>0)parts.push(chip(`${settings.gust.magnitude} m/s`,'gusts'));}
    if(selected.includes('sensor_noise'))parts.push(chip(ready?`σ ${settings.sensor_noise.position_std} m`:'Noise',ready?'position noise':'on'));
    if(selected.includes('com_shift'))parts.push(chip(ready?`±${(1000*Math.max(...settings.com_offset.range.flat().map(Math.abs))).toFixed(0)} mm`:'COM','COM box'));
    return {text:(preset?preset.name+' · ':'')+selected.map(k=>sourceNames[k]).join(' · '),chips:[...(preset?[chip(preset.name,'preset','chip-strong')]:[]),...parts],preset};
  }
  function drawPresets(){
    const current=describe().preset;
    root.querySelector('.env-presets').innerHTML=presets.map(p=>`<button type="button" data-env-preset="${escapeHtml(p.id)}" aria-pressed="${current?.id===p.id}" title="${escapeHtml(p.summary)}"><i aria-hidden="true">${presetIcons[p.id]??'•'}</i><b>${escapeHtml(p.name)}</b><small>${escapeHtml(p.summary)}</small></button>`).join('');
    root.querySelectorAll('[data-env-preset]').forEach(el=>el.onclick=()=>{const p=presets.find(x=>x.id===el.dataset.envPreset);if(!p)return;set(p.selected,p.settings);if(p.selected.length)selectSource(p.selected.includes('wind')?'wind':p.selected[0]);onChange?.();});
  }
  function draw(){
    const selected=chosen();
    root.querySelectorAll('[data-source]').forEach(el=>{el.classList.toggle('is-active',selected.includes(el.dataset.source));el.classList.toggle('is-off',!selected.includes(el.dataset.source));});
    const description=describe();summary.innerHTML=description.chips.join('');
    tabs.querySelectorAll('button').forEach(button=>{const on=selected.includes(button.dataset.sourceTab);const state=button.querySelector('.source-state');state.textContent=on?'ON':'OFF';state.classList.toggle('is-on',on);});
    drawPresets();
    if(!ready)return;
    const [x,y,z]=settings.wind.steady_vector,w=wind(),cx=120+x/15*88,cy=120-y/15*88;
    root.querySelector('.wind-arrow').setAttribute('d',`M120 120 L${cx} ${cy}`);root.querySelector('.wind-tip').setAttribute('cx',cx);root.querySelector('.wind-tip').setAttribute('cy',cy);
    root.querySelector('.wind-readout').textContent=`${w.speed.toFixed(2)} m/s → ${w.heading.toFixed(1)}°`;
    root.querySelector('.wind-band').textContent=`${windBand(w.speed)} breeze${Math.abs(z)>.05?` · ${z>0?'updraft':'downdraft'} ${Math.abs(z).toFixed(1)} m/s`:''}`.replace('calm breeze','calm air');
    root.querySelector('.wind-vector').textContent=`XYZ [${[x,y,z].map(v=>v.toFixed(2)).join(', ')}] m/s`;
    // Illustrative gust train: pulses at the mean wait, height = magnitude.
    const g=settings.gust,span=60,mean=(g.interval[0]+g.interval[1])/2,timeline=root.querySelector('.gust-timeline'),tw=Math.max(300,Math.round(timeline.clientWidth||420)),px=v=>14+v/span*(tw-28),top=Math.max(1,w.speed+g.magnitude,7.5),yv=v=>74-v/top*52;
    const pulses=[];for(let t=mean;t<span;t+=mean)pulses.push(t);
    timeline.innerHTML=`<svg viewBox="0 0 ${tw} 96" role="img" aria-label="Illustrative gust timeline over 60 seconds"><line class="diagram-axis" x1="14" x2="${tw-14}" y1="74" y2="74"/><line class="gust-steady" x1="14" x2="${tw-14}" y1="${yv(w.speed)}" y2="${yv(w.speed)}"/>${pulses.map(t=>`<rect class="gust-pulse" x="${px(t)}" y="${yv(w.speed+g.magnitude)}" width="${Math.max(1.5,px(t+g.duration)-px(t))}" height="${Math.max(0,yv(w.speed)-yv(w.speed+g.magnitude))}"/>`).join('')}<rect class="gust-window" x="${px(Math.min(span,g.interval[0]))}" y="80" width="${Math.max(1,px(Math.min(span,g.interval[1]))-px(Math.min(span,g.interval[0])))}" height="5"><title>Wait window between gusts</title></rect><text x="14" y="94">0 s</text><text x="${tw-14}" y="94" text-anchor="end">60 s</text><text x="14" y="12">steady ${w.speed.toFixed(1)} m/s + ${g.magnitude} m/s gusts of ${g.duration} s, every ${g.interval[0]}–${g.interval[1]} s · illustrative</text></svg>`;
    const sigma=settings.sensor_noise.position_std,width=Math.max(2,sigma*160),points=Array.from({length:81},(_,i)=>{const px=20+i*2.5,py=125-90*Math.exp(-.5*((px-120)/width)**2);return `${px},${py}`;}).join(' ');
    root.querySelector('.noise-visual').innerHTML=`<svg viewBox="0 0 240 155" role="img" aria-label="Position noise distribution with standard deviation ${sigma} meters"><path class="diagram-axis" d="M20 125 H220"/><polyline points="${points}"/><path class="diagram-axis" d="M120 20 V125"/><text x="25" y="148">POSITION σ = ${escapeHtml(sigma)} m</text></svg>`;
    const [low,high]=settings.com_offset.range,cx2=v=>120+v*1600,cy2=v=>77-v*1000;
    root.querySelector('.com-visual').innerHTML=`<svg viewBox="0 0 240 155" role="img" aria-label="Body X Y center of mass offset sampling bounds"><path class="diagram-axis" d="M20 77 H220 M120 15 V140"/><rect x="${cx2(low[0])}" y="${cy2(high[1])}" width="${Math.max(1,(high[0]-low[0])*1600)}" height="${Math.max(1,(high[1]-low[1])*1000)}"/><circle cx="120" cy="77" r="4"/><text x="195" y="69">+X</text><text x="129" y="24">+Y</text><text x="25" y="150">XY RANGE · Z SET BELOW</text></svg>`;
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
  root.addEventListener('invalid',e=>{const panel=e.target.closest('[data-source]');root.open=true;if(panel)selectSource(panel.dataset.source);const details=e.target.closest('details');if(details)details.open=true;},true);
  new ResizeObserver(()=>{if(ready)draw();}).observe(root.querySelector('.gust-timeline'));
  const compass=root.querySelector('.wind-compass');let dragging=false;
  function drag(e){if(!ready)return;const rect=compass.getBoundingClientRect(),scale=Math.min(rect.width,rect.height)/240;let x=(e.clientX-rect.left-rect.width/2)/scale/88*15,y=-(e.clientY-rect.top-rect.height/2)/scale/88*15;const length=Math.hypot(x,y);if(length>15){x*=15/length;y*=15/length;}settings.wind.steady_vector[0]=x;settings.wind.steady_vector[1]=y;sync();draw();onChange?.();}
  compass.onpointerdown=e=>{if(e.button!==0)return;dragging=true;compass.setPointerCapture(e.pointerId);drag(e);};compass.onpointermove=e=>{if(dragging)drag(e);};compass.onpointerup=compass.onpointercancel=compass.onlostpointercapture=()=>{dragging=false;};
  root.querySelector('.reset-disturbances').onclick=()=>{if(!ready)return;settings=structuredClone(base);sync();draw();onChange?.();};
  function set(selected,overrides={}){root.querySelectorAll('input[name=disturbance]').forEach(el=>{el.checked=selected.includes(el.value);});if(ready){settings=structuredClone(base);for(const [section,entries] of Object.entries(overrides??{}))Object.assign(settings[section],structuredClone(entries));sync();}draw();}
  return {
    // An already-running older service still supports the three preset toggles.
    configure(defaults){base=structuredClone(defaults);settings=structuredClone(base);ready=!!settings.wind;root.classList.toggle('presets-only',!ready);root.querySelectorAll('[data-path]').forEach(el=>{el.disabled=!ready;});root.querySelector('.reset-disturbances').disabled=!ready;if(ready)sync();else root.querySelector('.block-head p').textContent='Source toggles are available. Restart the mission-control service to customize parameters.';draw();
      if(ready)api?.('/api/disturbance-presets').then(value=>{presets=value;draw();}).catch(()=>{});},
    set,
    get(){return ready?structuredClone(settings):{};},
    preview(){return ready?{selected:chosen(),settings:structuredClone(settings)}:null;},
    describe(){return describe().text;},
  };
}
