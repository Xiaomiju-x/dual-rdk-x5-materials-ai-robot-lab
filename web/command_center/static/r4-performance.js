/* R4 rendering controller. Presentation modes are always explicit user choices. */
(function(){
'use strict';

var VALID={balanced:1,lite:1,static:1};
var WEIGHT={balanced:0,lite:1,static:2};
var tier='balanced';
var reason='default';
var visualMode='vivid';
var started=false;
var longTasks=[];
var longTaskObserver=null;
var routeObserver=null;
var recoveryTimer=null;
var reduceMotion=window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
var reduceTransparency=window.matchMedia && window.matchMedia('(prefers-reduced-transparency: reduce)');
var forcedColors=window.matchMedia && window.matchMedia('(forced-colors: active)');
var highContrast=window.matchMedia && window.matchMedia('(prefers-contrast: more)');

function emit(){
  var detail={tier:tier,reason:reason,visualMode:visualMode,hidden:document.hidden};
  window.dispatchEvent(new CustomEvent('r4renderchange',{detail:detail}));
}

function setTier(next, why, allowUpgrade){
  if(!VALID[next]) return false;
  if(visualMode==='minimal') next='static';
  else if(forcedColors && forcedColors.matches) next='static';
  else next='balanced';
  if(!allowUpgrade && WEIGHT[next]<WEIGHT[tier]) return false;
  if(next===tier && why===reason) return false;
  tier=next;
  reason=why||'runtime';
  if(document.body){
    document.body.dataset.r4Render=tier;
    document.body.dataset.r4RenderReason=reason;
  }
  applyRuntimeState();
  emit();
  return true;
}

function pauseNode(root, paused){
  if(!root || (root.nodeType!==1 && root.nodeType!==9)) return;
  var selector='.lb-wing,.lb-symbol,.lb-symbol svg,.gem,.spark,.lgr-route';
  var nodes=[];
  if(root.matches && root.matches(selector)) nodes.push(root);
  if(root.querySelectorAll) nodes=nodes.concat(Array.prototype.slice.call(root.querySelectorAll(selector)));
  nodes.forEach(function(node){
    if(paused) node.style.setProperty('animation-play-state','paused','important');
    else node.style.removeProperty('animation-play-state');
  });
}

function shouldPause(){
  return document.hidden || visualMode==='defense' || visualMode==='minimal' || tier==='static' || !!(reduceMotion && reduceMotion.matches);
}

function applyRuntimeState(){
  if(!document.body) return;
  var paused=shouldPause();
  document.body.dataset.r4Motion=paused?'paused':'active';
  document.body.dataset.r4Visibility=document.hidden?'hidden':'visible';
  document.body.dataset.r4Contrast=highContrast && highContrast.matches?'more':'normal';
  document.body.dataset.r4Transparency=(visualMode==='minimal' || (reduceTransparency && reduceTransparency.matches) ||
    (forcedColors && forcedColors.matches))?'off':'on';
  pauseNode(document,paused);
}

function capabilityTier(usePreset){
  var preset=usePreset && document.body && document.body.dataset.r4Render;
  if(VALID[preset]) return {tier:preset,reason:'preset'};
  if(forcedColors && forcedColors.matches) return {tier:'static',reason:'forced-colors'};
  var nav=navigator||{}, conn=nav.connection||nav.mozConnection||nav.webkitConnection;
  var memory=Number(nav.deviceMemory)||0, cores=Number(nav.hardwareConcurrency)||0;
  if((memory && memory<=2) || (cores && cores<=2)) return {tier:'static',reason:'low-end-device'};
  if((conn && conn.saveData) || (memory && memory<=4) || (cores && cores<=4))
    return {tier:'lite',reason:conn && conn.saveData?'save-data':'constrained-device'};
  return {tier:'balanced',reason:'capable-device'};
}

function scheduleRecovery(){
  if(recoveryTimer) clearTimeout(recoveryTimer);
  recoveryTimer=setTimeout(function(){
    recoveryTimer=null;
    if(visualMode!=='vivid' || document.hidden) return;
    var now=performance.now();
    longTasks=longTasks.filter(function(item){ return now-item.t<15000; });
    if(longTasks.length) return scheduleRecovery();
    var recovered=capabilityTier(false);
    setTier(recovered.tier,'pressure-recovered:'+recovered.reason,true);
  },30000);
}

function setVisualMode(next,why){
  if(next!=='vivid' && next!=='defense' && next!=='minimal') return false;
  visualMode=next;
  var derived=visualMode==='minimal'?'minimal':'vivid';
  document.documentElement.dataset.site32PresentationMode=visualMode;
  document.documentElement.dataset.site32VisualMode=derived;
  if(document.body){
    document.body.dataset.site32PresentationMode=visualMode;
    document.body.dataset.site32VisualMode=derived;
  }
  if(visualMode==='minimal') return setTier('static',why||'minimal-mode',true);
  var target=capabilityTier(false);
  return setTier(target.tier,why||((visualMode==='defense'?'defense-mode:':'vivid-mode:')+target.reason),true);
}

function observeLongTasks(){
  if(!window.PerformanceObserver || !PerformanceObserver.supportedEntryTypes ||
     PerformanceObserver.supportedEntryTypes.indexOf('longtask')<0) return;
  try{
    longTaskObserver=new PerformanceObserver(function(list){
      if(document.hidden || performance.now()<2000) return;
      var now=performance.now();
      list.getEntries().forEach(function(entry){ longTasks.push({t:now,d:entry.duration}); });
      longTasks=longTasks.filter(function(item){ return now-item.t<15000; });
      var total=longTasks.reduce(function(sum,item){ return sum+item.d; },0);
      if(longTasks.length>=8 || total>=700){ setTier('static','long-task-critical',false); scheduleRecovery(); }
      else if(longTasks.length>=3 || total>=240){ setTier('lite','long-task-pressure',false); scheduleRecovery(); }
    });
    longTaskObserver.observe({entryTypes:['longtask']});
  }catch(e){ longTaskObserver=null; }
}

function watchDynamicDecorations(){
  var targets=[document.getElementById('livingWingfield'),document.getElementById('livingSymbolfield')].filter(Boolean);
  if(!targets.length || !window.MutationObserver) return;
  routeObserver=new MutationObserver(function(records){
    var paused=shouldPause();
    records.forEach(function(record){
      Array.prototype.forEach.call(record.addedNodes,function(node){ pauseNode(node,paused); });
    });
  });
  targets.forEach(function(target){ routeObserver.observe(target,{childList:true,subtree:true}); });
}

function onPreferenceChange(){
  if(forcedColors && forcedColors.matches) setTier('static','forced-colors',false);
  else if(visualMode!=='minimal'){
    var target=capabilityTier(false);
    setTier(target.tier,'preference:'+target.reason,true);
  }
  applyRuntimeState();
}

function bindMedia(query, fn){
  if(!query) return;
  if(query.addEventListener) query.addEventListener('change',fn);
  else if(query.addListener) query.addListener(fn);
}

function start(){
  if(started || !document.body) return;
  started=true;
  var requested=document.documentElement.dataset.site32PresentationMode;
  visualMode=(requested==='defense'||requested==='minimal')?requested:'vivid';
  var initial=visualMode==='minimal'
    ? {tier:'static',reason:'minimal-mode'}
    : (forcedColors && forcedColors.matches
      ? {tier:'static',reason:'forced-colors'}
      : {tier:'balanced',reason:visualMode==='defense'?'defense-static-background':'full-experience'});
  tier=initial.tier; reason=initial.reason;
  document.body.dataset.r4Render=tier;
  document.body.dataset.r4RenderReason=reason;
  applyRuntimeState();
  watchDynamicDecorations();
  observeLongTasks();

  document.addEventListener('visibilitychange',applyRuntimeState);
  bindMedia(reduceMotion,onPreferenceChange);
  bindMedia(reduceTransparency,applyRuntimeState);
  bindMedia(forcedColors,onPreferenceChange);
  bindMedia(highContrast,applyRuntimeState);

  if(window.MutationObserver){
    var bodyRouteObserver=new MutationObserver(applyRuntimeState);
    bodyRouteObserver.observe(document.body,{attributes:true,attributeFilter:['data-view']});
  }
  var conn=navigator.connection||navigator.mozConnection||navigator.webkitConnection;
  if(conn && conn.addEventListener) conn.addEventListener('change',function(){
    if(conn.saveData) setTier('lite','save-data',false);
  });
  emit();
}

window.R4Performance={
  getTier:function(){ return tier; },
  getState:function(){ return {tier:tier,reason:reason,visualMode:visualMode,hidden:document.hidden,paused:shouldPause()}; },
  setTier:function(next,why){ return setTier(next,why||'manual',true); },
  downgrade:function(next,why){ return setTier(next,why||'manual-downgrade',false); },
  setVisualMode:setVisualMode,
  recompute:function(){ var target=capabilityTier(false); return setTier(target.tier,'manual-recompute:'+target.reason,true); },
  refresh:applyRuntimeState
};

if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',start,{once:true});
else start();
})();
