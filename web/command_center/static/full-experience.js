/* Site32 v1.11 default full-experience bootstrap. */
(function(){
'use strict';

var root=document.documentElement;
var attempts=0;

function isVivid(){
  return root.dataset.site32VisualMode!=='minimal';
}

function webglAvailable(){
  try{
    var canvas=document.createElement('canvas');
    return !!(window.WebGLRenderingContext &&
      (canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
  }catch(error){
    return false;
  }
}

function ensureFullThree(){
  if(!isVivid() || !document.body || document.body.dataset.view!=='home') return;
  var canvas=document.getElementById('c3d');
  var fallback=document.getElementById('scene2d');
  if(!canvas || !fallback) return;
  if(!webglAvailable()){
    document.body.dataset.fullThree='fallback-webgl';
    return;
  }
  if(window.R4Performance && window.R4Performance.setTier){
    window.R4Performance.setTier('balanced','full-experience');
  }
  if(canvas.style.display!=='none'){
    document.body.dataset.fullThree='ready';
    return;
  }
  if(window.ensureThreeLibrary) window.ensureThreeLibrary();
  attempts+=1;
  if(attempts<24) setTimeout(ensureFullThree,125);
  else document.body.dataset.fullThree='fallback-init';
}

function applyFullExperience(){
  if(!document.body) return;
  document.body.dataset.fullExperience=isVivid()?'true':'false';
  if(isVivid() && window.R4Performance && window.R4Performance.setTier){
    window.R4Performance.setTier('balanced','full-experience');
  }
  attempts=0;
  requestAnimationFrame(ensureFullThree);
}

function watchMenu(){
  var menu=document.getElementById('moreMenu');
  if(!menu || !window.MutationObserver) return;
  new MutationObserver(function(){
    var open=menu.classList.contains('show');
    menu.setAttribute('aria-hidden',open?'false':'true');
    if(open) menu.dataset.visibilityContract='visible';
    else delete menu.dataset.visibilityContract;
  }).observe(menu,{attributes:true,attributeFilter:['class']});
}

window.addEventListener('site32:appearance-change',applyFullExperience);
window.addEventListener('r4renderchange',function(event){
  if(isVivid() && event.detail && event.detail.tier!=='balanced' && window.R4Performance){
    window.R4Performance.setTier('balanced','full-experience-guard');
  }
});
window.addEventListener('three-library-ready',ensureFullThree);
document.addEventListener('visibilitychange',function(){
  if(!document.hidden) ensureFullThree();
});

function start(){
  watchMenu();
  applyFullExperience();
}

if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',start,{once:true});
else start();
})();
