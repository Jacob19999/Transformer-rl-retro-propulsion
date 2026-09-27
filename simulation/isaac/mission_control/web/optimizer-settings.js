import {escapeHtml,categoryNames} from './flight-plan.js';
import {optimizerVisual,optimizerGroups,envelopeVisual} from './optimizer-visuals.js';

// Key constraints shown above the full parameter editor. Every control carries
// data-section/data-key, so it stays in sync with the matching full field.
const essentials=[
  ['guidance','route_corridor_mode','segmented',{soft:['Soft','Penalize excess; emergency fallback allowed'],strict:['Strict','Reject plans outside; hold if infeasible']}],
  ['guidance','route_corridor_m','slider'],
  ['guidance','max_tilt_deg','slider'],
  ['tracking','max_tilt_deg','slider'],
  ['guidance','max_speed_m_s','slider'],
  ['guidance','glide_slope_deg','slider'],
  ['guidance','objective','segmented',{energy:['Energy','Electrical Wh'],delta_v:['Delta-v','∫|T|/m, fuel analog']}],
];
const essentialLabels={'guidance.route_corridor_mode':'Corridor enforcement','guidance.route_corridor_m':'Corridor half-width','guidance.max_tilt_deg':'Planned tilt limit',
  'tracking.max_tilt_deg':'Feedback tilt limit','guidance.max_speed_m_s':'Speed limit','guidance.glide_slope_deg':'Landing glide slope','guidance.objective':'Cost'};

export function createOptimizerSettings(root,{api,onChange}){
  let schema=[],overrides={},profiles=[],presets=[],active='Route corridor',filter='all',lastApplied=null;
  root.innerHTML=`<summary class="config-summary section-head"><span class="section-index">02</span><div class="section-title"><h2>Guidance</h2><p>Convex optimizer profile: corridor, tilt, speed and landing envelope.</p></div><span id="optimizerSummary" class="summary-chips"></span><span class="fold-chevron" aria-hidden="true"></span></summary><div class="config-body">
    <section class="preset-panel" aria-labelledby="presetTitle"><div class="block-head"><div><h3 id="presetTitle">Mission profile</h3><p>Pick the profile that matches the mission. Applying one replaces every optimizer override; tune afterwards below.</p></div><div class="segmented" role="group" aria-label="Filter profiles">${[['all','All'],['hop','Hopping'],['hover','Hovering'],['land','Landing'],['saved','My profiles']].map(([key,name])=>`<button type="button" data-preset-filter="${key}" aria-pressed="${key==='all'}">${name}</button>`).join('')}</div></div>
      <div id="presetCards" class="preset-cards" role="list"></div><div id="presetDetail" class="preset-detail"></div></section>
    <section class="essentials" aria-labelledby="essentialsTitle"><div class="block-head"><div><h3 id="essentialsTitle">Key constraints</h3><p>The limits that shape the plan most. The diagram redraws as you drag.</p></div><button type="button" id="resetOptimizer">Restore repository defaults</button></div>
      <div class="essentials-grid"><div id="envelopeVisual" class="envelope-visual"></div><div id="essentialControls" class="essential-controls"></div></div></section>
    <div id="optimizerMessage" role="status" class="hint"></div>
    <details class="advanced-optimizer"><summary><span class="fold-title">All parameters & saved profiles</span><span id="advancedSummary" class="fold-note"></span></summary>
      <div class="optimizer-profiles"><div class="optimizer-profile-load"><label for="optimizerProfile">Saved profile</label><div><select id="optimizerProfile"><option value="">Choose profile</option></select><button type="button" id="loadOptimizer" disabled>Load</button></div></div><div class="optimizer-profile-save"><label for="optimizerName">Save current settings as</label><div><input id="optimizerName" maxlength="48" placeholder="My guidance settings"><button type="button" id="saveOptimizer">Save profile</button></div></div></div>
      <div class="optimizer-workspace"><div id="optimizerNav" class="optimizer-nav" role="tablist" aria-label="Optimizer parameter groups" aria-orientation="vertical"></div><div id="optimizerFields"></div></div></details></div>`;
  const fields=root.querySelector('#optimizerFields'),nav=root.querySelector('#optimizerNav'),message=root.querySelector('#optimizerMessage');
  const find=(section,key)=>schema.find(p=>p.section===section&&p.key===key);
  const valueIn=(settings,p)=>settings?.[p.section]?.[p.key]??p.default;
  const value=p=>valueIn(overrides,p);
  const changed=p=>value(p)!==p.default;
  const groups=()=>[...new Set([...Object.keys(optimizerGroups),...schema.map(p=>p.group)])].filter(g=>schema.some(p=>p.group===g));
  // Profiles are equal when every parameter resolves to the same value, however
  // the override objects spell repository defaults.
  const same=settings=>schema.length>0&&schema.every(p=>valueIn(settings,p)===value(p));
  const matching=()=>[...presets,...profiles].find(p=>same(p.settings));
  function select(group,focus=false){
    active=group;
    nav.querySelectorAll('button').forEach(el=>{const selected=el.dataset.group===group;el.setAttribute('aria-selected',String(selected));el.tabIndex=selected?0:-1;if(selected&&focus)el.focus();});
    fields.querySelectorAll('[role="tabpanel"]').forEach(el=>{el.hidden=el.dataset.group!==group;});
  }
  function setValue(p,v){
    if(v===p.default){delete overrides[p.section]?.[p.key];if(overrides[p.section]&&!Object.keys(overrides[p.section]).length)delete overrides[p.section];}
    else (overrides[p.section]??={})[p.key]=v;
  }
  // Copy the current values into every control except the one being edited.
  function sync(except){
    root.querySelectorAll('[data-section][data-key]').forEach(el=>{
      if(el===except)return;const p=find(el.dataset.section,el.dataset.key);if(!p)return;
      if(el.type==='radio')el.checked=el.value===String(value(p));else if(document.activeElement!==el||el.type==='range')el.value=value(p);
    });
    const planned=find('guidance','max_tilt_deg'),command=find('tracking','max_tilt_deg');
    if(planned&&command)root.querySelectorAll('[data-section="tracking"][data-key="max_tilt_deg"]').forEach(el=>el.setCustomValidity(value(command)<value(planned)?`Feedback tilt must be at least the planned tilt (${value(planned)}°): feedback needs margin.`:''));
  }
  const chip=(v,k,cls='')=>`<span class="chip ${cls}"><b>${escapeHtml(v)}</b>${escapeHtml(k)}</span>`;
  function describe(){const count=schema.filter(changed).length,match=matching();return {count,name:match?.name??(count?'Custom':'Repository defaults'),builtin:!!match?.builtin};}
  function update(){
    const {count,name}=describe(),g=key=>value(find('guidance',key)??{default:undefined});
    root.querySelector('#optimizerSummary').innerHTML=schema.length?chip(name,'profile','chip-strong')+chip(String(g('route_corridor_mode')).toUpperCase(),'corridor',g('route_corridor_mode')==='strict'?'chip-strict':'')+chip(`${g('max_tilt_deg')}°`,'tilt')+chip(`${g('max_speed_m_s')} m/s`,'speed'):'';
    root.querySelector('#advancedSummary').textContent=`${schema.length} parameters · ${count?`${count} adjusted`:'repository defaults'}`;
    groups().forEach((group,i)=>{
      const parameters=schema.filter(p=>p.group===group),adjusted=parameters.filter(changed).length;
      nav.querySelector(`[data-group="${group}"] small`).textContent=adjusted?`${adjusted} adjusted`:`${parameters.length} parameter${parameters.length===1?'': 's'}`;
      const panel=fields.querySelector(`#optimizer-panel-${i}`);
      panel.querySelector('.optimizer-visual').innerHTML=optimizerVisual(group,(section,key)=>{const p=find(section,key);return p?value(p):undefined;});
      panel.querySelector('[data-reset-group]').disabled=!adjusted;
    });
    fields.querySelectorAll('[data-section]').forEach(el=>el.closest('.optimizer-field').classList.toggle('is-adjusted',changed(find(el.dataset.section,el.dataset.key))));
    root.querySelectorAll('.essential-control').forEach(el=>el.classList.toggle('is-adjusted',changed(find(el.dataset.param.split('.')[0],el.dataset.param.split('.')[1]))));
    root.querySelector('#envelopeVisual').innerHTML=envelopeVisual((section,key)=>{const p=find(section,key);return p?value(p):undefined;});
    drawPresets();
  }
  function metricsOf(settings){
    const v=(section,key)=>{const p=find(section,key);return p?valueIn(settings,p):undefined;};
    return [['Tilt',`${v('guidance','max_tilt_deg')}°`],['Speed',`${v('guidance','max_speed_m_s')} m/s`],['Corridor',`${v('guidance','route_corridor_m')} m`],['Glide',`${v('guidance','glide_slope_deg')}°`]];
  }
  function drawPresets(){
    const match=matching();
    root.querySelectorAll('[data-preset-filter]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.presetFilter===filter)));
    const saved=profiles.map((p,i)=>({...p,id:`saved:${i}`,mission:'saved',corridor:p.settings?.guidance?.route_corridor_mode??find('guidance','route_corridor_mode')?.default,summary:p.note||'Saved on this computer.'}));
    const shown=filter==='saved'?saved:filter==='all'?presets:presets.filter(p=>p.mission===filter);
    const cards=root.querySelector('#presetCards');
    cards.innerHTML=shown.map(p=>{const on=match&&(match===p||(p.id?.startsWith('saved:')&&profiles[Number(p.id.split(':')[1])]===match));
      return `<article class="preset-card mission-${escapeHtml(p.mission)}${on?' is-active':''}" role="listitem"><div class="sample-tags"><span class="tag tag-${escapeHtml(p.mission)}">${p.mission==='saved'?'SAVED':categoryNames[p.mission]??escapeHtml(p.mission)}</span><span class="tag ${p.corridor==='strict'?'tag-strict':'tag-soft'}">${escapeHtml(String(p.corridor).toUpperCase())} CORRIDOR</span></div><h4>${escapeHtml(p.name)}</h4><p>${escapeHtml(p.summary)}</p><dl>${metricsOf(p.settings).map(([k,v])=>`<div><dt>${k}</dt><dd>${escapeHtml(v)}</dd></div>`).join('')}</dl><div class="preset-foot">${p.rationale?`<button type="button" class="link-button" data-preset-why="${escapeHtml(p.id)}">Why these values?</button>`:'<span></span>'}<button type="button" data-apply-preset="${escapeHtml(p.id)}" ${on?'disabled':''}>${on?'✓ Applied':'Apply'}</button></div></article>`;}).join('')||`<p class="hint">${filter==='saved'?'No saved profiles yet. Save the current settings under All parameters.':'Built-in profiles are not available from this service; restart mission control to load them.'}</p>`;
    const lookup=id=>id.startsWith('saved:')?saved[Number(id.split(':')[1])]:presets.find(p=>p.id===id);
    cards.querySelectorAll('[data-apply-preset]').forEach(el=>el.onclick=()=>{const p=lookup(el.dataset.applyPreset);if(!p)return;overrides=structuredClone(p.settings);lastApplied=p;draw();onChange?.();message.textContent=`Applied ${p.name}. Every other parameter is at its repository default.`;});
    cards.querySelectorAll('[data-preset-why]').forEach(el=>el.onclick=()=>{const p=lookup(el.dataset.presetWhy),detail=root.querySelector('#presetDetail');if(detail.dataset.id===p.id&&!detail.hidden){detail.hidden=true;return;}detail.dataset.id=p.id;detail.hidden=false;detail.innerHTML=`<b>${escapeHtml(p.name)}</b><p>${escapeHtml(p.rationale)}</p><ul>${Object.entries(p.settings).flatMap(([section,entries])=>Object.entries(entries).map(([key,v])=>{const q=find(section,key);return `<li><span>${escapeHtml(q?.label??key)}${section==='tracking'?' (tracking)':''}</span><b>${escapeHtml(v)}${q?.unit?` ${escapeHtml(q.unit)}`:''}</b><small>default ${escapeHtml(q?.default)}</small></li>`;})).join('')}</ul>`;});
    const detail=root.querySelector('#presetDetail');if(!detail.dataset.id)detail.hidden=true;
    if(lastApplied&&!match&&!message.textContent.startsWith('Modified'))message.textContent=`Modified from ${lastApplied.name}.`;
  }
  function essentialControl([section,key,kind,options]){
    const p=find(section,key);if(!p)return '';
    const id=`essential-${section}-${key}`,label=essentialLabels[`${section}.${key}`]??p.label;
    if(kind==='segmented')return `<fieldset class="essential-control segmented-field" data-param="${section}.${key}"><legend>${escapeHtml(label)}</legend><div class="segmented wide">${p.choices.map(c=>`<label><input type="radio" name="${id}" value="${escapeHtml(c)}" data-section="${section}" data-key="${key}"><span><b>${escapeHtml(options?.[c]?.[0]??c)}</b><small>${escapeHtml(options?.[c]?.[1]??'')}</small></span></label>`).join('')}</div></fieldset>`;
    return `<div class="essential-control" data-param="${section}.${key}"><label for="${id}"><span>${escapeHtml(label)}</span><small>${escapeHtml(p.unit??'')}</small></label><div class="slider-row"><input type="range" aria-label="${escapeHtml(label)} slider" min="${p.min}" max="${p.max}" step="${p.step}" data-section="${section}" data-key="${key}"><input id="${id}" type="number" min="${p.min}" max="${p.max}" step="${p.step}" data-section="${section}" data-key="${key}" aria-describedby="${id}-help" required></div><small id="${id}-help">${escapeHtml(p.help)}</small></div>`;
  }
  function draw(){
    root.querySelector('#essentialControls').innerHTML=essentials.map(essentialControl).join('');
    nav.innerHTML=groups().map((g,i)=>`<button type="button" role="tab" id="optimizer-tab-${i}" aria-controls="optimizer-panel-${i}" data-group="${escapeHtml(g)}"><span class="optimizer-nav-index">${i<5?'FLIGHT':'TUNING'}</span><span>${escapeHtml(g)}<small></small></span><span class="optimizer-nav-arrow" aria-hidden="true">›</span></button>`).join('');
    fields.innerHTML=groups().map((group,i)=>`<section id="optimizer-panel-${i}" role="tabpanel" aria-labelledby="optimizer-tab-${i}" data-group="${escapeHtml(group)}"><div class="optimizer-group-heading"><div><h3>${escapeHtml(group)}</h3><p>${escapeHtml(optimizerGroups[group]??'Adjust the parameters for this part of guidance.')}</p></div><button type="button" data-reset-group="${escapeHtml(group)}">Reset group</button></div><div class="optimizer-visual"></div><div class="optimizer-grid">${schema.filter(p=>p.group===group).map(p=>{
      const id=`optimizer-${p.section}-${p.key}`;
      return `<label class="optimizer-field" for="${id}"><span class="optimizer-field-title">${escapeHtml(p.label)}<span>${escapeHtml(p.unit??'')}</span></span>${p.kind==='choice'?`<select id="${id}" data-section="${p.section}" data-key="${p.key}" aria-label="${escapeHtml(p.label)}" aria-describedby="${id}-help">${p.choices.map(c=>`<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('')}</select>`:`<input id="${id}" data-section="${p.section}" data-key="${p.key}" aria-label="${escapeHtml(p.label)}" aria-describedby="${id}-help" type="number" min="${p.min}" max="${p.max}" step="${p.step}" required>`}<small id="${id}-help">${escapeHtml(p.help)}</small><span class="optimizer-default">Default ${escapeHtml(p.default)}${p.kind==='choice'?'':` · Range ${p.min}–${p.max}`}</span></label>`;
    }).join('')}</div></section>`).join('');
    nav.querySelectorAll('button').forEach(el=>{
      el.onclick=()=>select(el.dataset.group);
      el.onkeydown=e=>{if(!['ArrowDown','ArrowRight','ArrowUp','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();e.stopPropagation();const list=groups(),index=list.indexOf(active);select(list[e.key==='Home'?0:e.key==='End'?list.length-1:(index+(['ArrowUp','ArrowLeft'].includes(e.key)?-1:1)+list.length)%list.length],true);};
    });
    fields.querySelectorAll('[data-reset-group]').forEach(el=>el.onclick=()=>{schema.filter(p=>p.group===el.dataset.resetGroup).forEach(p=>setValue(p,p.default));draw();onChange?.();message.textContent=`${active} defaults restored.`;});
    select(groups().includes(active)?active:groups()[0]);sync();update();
  }
  // One handler for the essentials and the full editor: sliders update live,
  // the planner and launch summary refresh when the value is committed.
  function edit(el,commit){
    if(!el.dataset?.section)return;
    if(el.type!=='radio'&&el.type!=='range'&&!el.checkValidity()){if(commit)el.reportValidity();return;}
    const p=find(el.dataset.section,el.dataset.key);if(!p)return;
    if(el.type==='radio'&&!el.checked)return;
    setValue(p,p.kind==='choice'?el.value:Number(el.value));
    sync(el);update();
    if(commit){const tracking=root.querySelector('[data-section="tracking"][data-key="max_tilt_deg"]:invalid');if(tracking&&el.dataset.key==='max_tilt_deg')tracking.reportValidity();onChange?.();}
  }
  root.addEventListener('input',e=>edit(e.target,false));
  root.addEventListener('change',e=>{if(e.target.closest('.optimizer-profiles'))return;edit(e.target,true);});
  // Reveal invalid controls before native form validation tries to focus them.
  root.addEventListener('invalid',e=>{const details=e.target.closest('details');if(details)details.open=true;root.open=true;const panel=fields.querySelector(':invalid')?.closest('[role="tabpanel"]');if(panel)select(panel.dataset.group);},true);
  root.querySelectorAll('[data-preset-filter]').forEach(el=>el.onclick=()=>{filter=el.dataset.presetFilter;drawPresets();});
  async function refresh(){profiles=await api('/api/convex-profiles');root.querySelector('#optimizerProfile').replaceChildren(new Option('Choose profile',''),...profiles.map((p,i)=>new Option(p.name,String(i))));root.querySelector('#loadOptimizer').disabled=true;update();}
  root.querySelector('#optimizerProfile').onchange=e=>{root.querySelector('#loadOptimizer').disabled=e.target.value==='';};
  root.querySelector('#loadOptimizer').onclick=()=>{const i=root.querySelector('#optimizerProfile').value;if(i==='')return;const p=profiles[Number(i)];overrides=structuredClone(p.settings);lastApplied=p;root.querySelector('#optimizerName').value=p.name;draw();onChange?.();message.textContent=`Loaded ${p.name}.`;};
  root.querySelector('#saveOptimizer').onclick=async()=>{try{const invalid=root.querySelector('.config-body :invalid');if(invalid){const panel=invalid.closest('[role="tabpanel"]');if(panel)select(panel.dataset.group);invalid.reportValidity();return;}const p=await api('/api/convex-profiles',{name:root.querySelector('#optimizerName').value,settings:overrides});await refresh();message.textContent=`Saved ${p.name} on this computer.`;}catch(e){message.textContent=e.message;}};
  root.querySelector('#resetOptimizer').onclick=()=>{overrides={};lastApplied=null;draw();onChange?.();message.textContent='Repository defaults restored.';};
  return {configure:value=>{schema=value;draw();refresh().catch(e=>{message.textContent=e.message;});api('/api/convex-presets').then(value=>{presets=value;update();}).catch(()=>{presets=[];update();});},
    get:()=>structuredClone(overrides),set:value=>{overrides=structuredClone(value??{});lastApplied=null;message.textContent='';draw();onChange?.();},describe};
}
