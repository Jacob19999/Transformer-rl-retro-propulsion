import {escapeHtml} from './flight-plan.js';
import {optimizerVisual,optimizerGroups} from './optimizer-visuals.js';

export function createOptimizerSettings(root,{api,onChange}){
  let schema=[],overrides={},profiles=[],active='Route corridor';
  root.innerHTML=`<div class="optimizer-heading"><div><div class="eyebrow">GUIDANCE / SOCP</div><h2>Convex optimizer</h2><p>Shape the flight envelope, then tune how the plan is solved and tracked.</p></div><span id="optimizerSummary" class="optimizer-summary"></span></div>
    <div class="optimizer-profiles"><div class="optimizer-profile-load"><label for="optimizerProfile">Saved profile</label><div><select id="optimizerProfile"><option value="">Choose profile</option></select><button type="button" id="loadOptimizer" disabled>LOAD</button></div></div><div class="optimizer-profile-save"><label for="optimizerName">Save current settings as</label><div><input id="optimizerName" maxlength="48" placeholder="My guidance settings"><button type="button" id="saveOptimizer">SAVE PROFILE</button></div></div><button type="button" id="resetOptimizer">RESTORE ALL DEFAULTS</button></div>
    <div id="optimizerMessage" role="status" class="hint"></div><div class="optimizer-workspace"><div id="optimizerNav" class="optimizer-nav" role="tablist" aria-label="Optimizer parameter groups" aria-orientation="vertical"></div><div id="optimizerFields"></div></div>`;
  const fields=root.querySelector('#optimizerFields'),nav=root.querySelector('#optimizerNav'),message=root.querySelector('#optimizerMessage');
  const value=p=>overrides[p.section]?.[p.key]??p.default;
  const changed=p=>value(p)!==p.default;
  const groups=()=>[...new Set([...Object.keys(optimizerGroups),...schema.map(p=>p.group)])].filter(g=>schema.some(p=>p.group===g));
  function select(group,focus=false){
    active=group;
    nav.querySelectorAll('button').forEach(el=>{const selected=el.dataset.group===group;el.setAttribute('aria-selected',String(selected));el.tabIndex=selected?0:-1;if(selected&&focus)el.focus();});
    fields.querySelectorAll('[role="tabpanel"]').forEach(el=>{el.hidden=el.dataset.group!==group;});
  }
  function update(){
    const count=schema.filter(changed).length;
    root.querySelector('#optimizerSummary').textContent=`${schema.length} parameters · ${count?`${count} adjusted`:'Repository defaults'}`;
    groups().forEach((group,i)=>{
      const parameters=schema.filter(p=>p.group===group),count=parameters.filter(changed).length;
      nav.querySelector(`[data-group="${group}"] small`).textContent=count?`${count} adjusted`:`${parameters.length} parameter${parameters.length===1?'': 's'}`;
      const panel=fields.querySelector(`#optimizer-panel-${i}`);
      panel.querySelector('.optimizer-visual').innerHTML=optimizerVisual(group,(section,key)=>{const p=schema.find(p=>p.section===section&&p.key===key);return p?value(p):undefined;});
      panel.querySelector('[data-reset-group]').disabled=!count;
    });
    fields.querySelectorAll('[data-section]').forEach(el=>el.closest('.optimizer-field').classList.toggle('is-adjusted',changed(schema.find(p=>p.section===el.dataset.section&&p.key===el.dataset.key))));
  }
  function draw(){
    nav.innerHTML=groups().map((g,i)=>`<button type="button" role="tab" id="optimizer-tab-${i}" aria-controls="optimizer-panel-${i}" data-group="${escapeHtml(g)}"><span class="optimizer-nav-index">${String(i+1).padStart(2,'0')}</span><span>${escapeHtml(g)}<small></small></span><span class="optimizer-nav-arrow" aria-hidden="true">›</span></button>`).join('');
    fields.innerHTML=groups().map((group,i)=>`<section id="optimizer-panel-${i}" role="tabpanel" aria-labelledby="optimizer-tab-${i}" data-group="${escapeHtml(group)}"><div class="optimizer-group-heading"><div><h3>${escapeHtml(group)}</h3><p>${escapeHtml(optimizerGroups[group]??'Adjust the parameters for this part of guidance.')}</p></div><button type="button" data-reset-group="${escapeHtml(group)}">RESET GROUP</button></div><div class="optimizer-visual"></div><div class="optimizer-grid">${schema.filter(p=>p.group===group).map(p=>{
      const id=`optimizer-${p.section}-${p.key}`;
      return `<label class="optimizer-field" for="${id}"><span class="optimizer-field-title">${escapeHtml(p.label)}<span>${escapeHtml(p.unit??'')}</span></span>${p.kind==='choice'?`<select id="${id}" data-section="${p.section}" data-key="${p.key}" aria-label="${escapeHtml(p.label)}" aria-describedby="${id}-help">${p.choices.map(c=>`<option value="${escapeHtml(c)}" ${c===value(p)?'selected':''}>${escapeHtml(c)}</option>`).join('')}</select>`:`<input id="${id}" data-section="${p.section}" data-key="${p.key}" aria-label="${escapeHtml(p.label)}" aria-describedby="${id}-help" type="number" min="${p.min}" max="${p.max}" step="${p.step}" value="${value(p)}" required>`}<small id="${id}-help">${escapeHtml(p.help)}</small><span class="optimizer-default">Default ${escapeHtml(p.default)}${p.kind==='choice'?'':` · Range ${p.min}–${p.max}`}</span></label>`;
    }).join('')}</div></section>`).join('');
    nav.querySelectorAll('button').forEach(el=>{
      el.onclick=()=>select(el.dataset.group);
      el.onkeydown=e=>{if(!['ArrowDown','ArrowRight','ArrowUp','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();e.stopPropagation();const list=groups(),index=list.indexOf(active);select(list[e.key==='Home'?0:e.key==='End'?list.length-1:(index+(['ArrowUp','ArrowLeft'].includes(e.key)?-1:1)+list.length)%list.length],true);};
    });
    fields.querySelectorAll('input,select').forEach(el=>el.onchange=()=>{
      if(!el.reportValidity())return;
      const p=schema.find(p=>p.section===el.dataset.section&&p.key===el.dataset.key),v=el.tagName==='SELECT'?el.value:Number(el.value);
      if(v===p.default){delete overrides[p.section]?.[p.key];if(overrides[p.section]&&!Object.keys(overrides[p.section]).length)delete overrides[p.section];}
      else (overrides[p.section]??={})[p.key]=v;
      update();onChange?.();
    });
    fields.querySelectorAll('[data-reset-group]').forEach(el=>el.onclick=()=>{schema.filter(p=>p.group===el.dataset.resetGroup).forEach(p=>{delete overrides[p.section]?.[p.key];if(overrides[p.section]&&!Object.keys(overrides[p.section]).length)delete overrides[p.section];});draw();onChange?.();message.textContent=`${active} defaults restored.`;});
    select(groups().includes(active)?active:groups()[0]);update();
  }
  // Reveal invalid controls before native form validation tries to focus them.
  root.addEventListener('invalid',()=>{const panel=fields.querySelector(':invalid')?.closest('[role="tabpanel"]');if(panel)select(panel.dataset.group);},true);
  async function refresh(){profiles=await api('/api/convex-profiles');root.querySelector('#optimizerProfile').replaceChildren(new Option('Choose profile',''),...profiles.map((p,i)=>new Option(p.name,String(i))));root.querySelector('#loadOptimizer').disabled=true;}
  root.querySelector('#optimizerProfile').onchange=e=>{root.querySelector('#loadOptimizer').disabled=e.target.value==='';};
  root.querySelector('#loadOptimizer').onclick=()=>{const i=root.querySelector('#optimizerProfile').value;if(i==='')return;const p=profiles[Number(i)];overrides=structuredClone(p.settings);root.querySelector('#optimizerName').value=p.name;draw();onChange?.();message.textContent=`Loaded ${p.name}.`;};
  root.querySelector('#saveOptimizer').onclick=async()=>{try{const invalid=fields.querySelector(':invalid');if(invalid){select(invalid.closest('[role="tabpanel"]').dataset.group);invalid.reportValidity();return;}const p=await api('/api/convex-profiles',{name:root.querySelector('#optimizerName').value,settings:overrides});await refresh();message.textContent=`Saved ${p.name} on this computer.`;}catch(e){message.textContent=e.message;}};
  root.querySelector('#resetOptimizer').onclick=()=>{overrides={};draw();onChange?.();message.textContent='Repository defaults restored.';};
  return {configure:value=>{schema=value;draw();refresh().catch(e=>{message.textContent=e.message;});},get:()=>structuredClone(overrides),set:value=>{overrides=structuredClone(value??{});draw();onChange?.();}};
}
