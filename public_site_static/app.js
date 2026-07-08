/* PWA: 注册 service worker (装到主屏 + 离线壳) */
if('serviceWorker' in navigator){ window.addEventListener('load', ()=>navigator.serviceWorker.register('/sw.js').catch(()=>{})); }

const SYS = {
  lab: { url: "https://lab.xiaomiju.xyz/" },
  car: { url: "https://car.xiaomiju.xyz/" },
  arm: { url: "https://arm.xiaomiju.xyz/" },
};
const PAGES = { home:'overview', highlight:'highlights', status:'statusview', fleet:'fleetview', tasks:'tasksview', ops:'opsview', story:'storyview', mq:'mqview', studio:'studioview', fsd:'fsdview', replay:'replayview', command:'commandview', defense:'defenseview', benchmark:'benchmarkview', assets:'assetsview', twin:'twinview', preflight:'preflightview', atlas:'atlasview', brain:'brainview', detail:'detailview', archive:'archiveview', models:'modelsview', standards:'standardsview', cost:'costview', glossary:'glossaryview', changelog:'changelogview', build:'buildview', importw:'importview', eln:'elnview', sync:'syncview', repro:'reproview', ar:'arview', obs:'obsview', traces:'tracesview', topo:'topoview', budget:'sloview', inc:'incview', tm:'timeview', noc:'nocview', self:'selfview', oee:'oeeview', alert:'alertview', logs:'logsview', qms:'qmsview', cmms:'cmmsview', sec:'secview', release:'releaseview', data:'dataview' };  // 非 iframe 的滚动页
const PAGES_ALL=['home','highlight','status','fleet','tasks','ops','story','mq','assets','twin','preflight','atlas','detail','archive',
  'models','standards','cost','glossary','changelog','build','importw','eln','sync','repro','ar','obs','topo','budget','inc','tm','noc',
  'self','oee','alert','logs','traces','qms','cmms','sec','release','data','studio','fsd','replay','command','defense','benchmark','brain','lab','car','arm'];
let cur = "home";
document.body.dataset.view = cur;
let DETAIL_KIND='material', DETAIL_ID='', DETAIL_TAB='structure', DETAIL_DATA=null;
const frameEls = {};
const frameFallbackEls = {};
const FRAME_FALLBACK = {
  lab:{ icon:'🧠', title:'AI 脑 · 智能计算平台', sub:'9 本地 LLM + 5 BPU 推理槽 + 云 R1', chips:['NIR 配方预测','Conformal CI','GraphRAG 文献向量','BPU Transformer'] },
  car:{ icon:'🚗', title:'车载脑 · Lab-FSD v2', sub:'BEV 占据 · 时序世界模型 · 安全门禁', chips:['Nav2/MPPI 权威控制','BPU anomaly AE ≈1.7ms','策略 Token','断网本地 Agent'] },
  arm:{ icon:'🦾', title:'机械臂 · WorkCockpit', sub:'双 myCobot 280 · 10 stage 工位剧本', chips:['DH 正运动学','Capsule 防撞','研磨灌装','批次回填'] },
};

function toast(m, type){ const t=document.getElementById('toast'); t.textContent=m;
  t.className='toast show'+(type?' '+type:'');
  clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove('show'),2200); }

/* U2: 点击涟漪 (fixed 定位免 overflow 裁剪; 只对可点元素生效) */
document.addEventListener('pointerdown', (e)=>{
  if(!e.target.closest('.act,.pill,.ackbtn,.csvbtn,.tb,.kb-btn,.twvbtn,button')) return;
  const p=document.createElement('span'); p.className='clickping';
  p.style.left=e.clientX+'px'; p.style.top=e.clientY+'px';
  document.body.appendChild(p); setTimeout(()=>p.remove(), 520);
});

/* U2: 数字滚动 (KPI count-up) */
function tweenNum(el, target, fmt){
  const from=parseFloat(el.dataset.v||'0')||0;
  el.dataset.v=target;
  if(!isFinite(target) || Math.abs(target-from)<1e-9){ el.innerHTML=fmt(target); return; }
  const t0=performance.now(), dur=620;
  (function step(t){
    const u=Math.min(1,(t-t0)/dur), k=1-Math.pow(1-u,3);
    el.innerHTML=fmt(from+(target-from)*k);
    if(u<1) requestAnimationFrame(step);
  })(t0);
}

/* ---- 过渡动画 ---- */
const BOOT = {
  home:{ icon:'⬢', title:'智研指挥中心', sub:'三机异构协同 · 统一控制台', ms:330, foot:'指挥链路',
    stages:['接入三机异构链路','加载 AI 脑算力档案','同步车载脑遥测流','校验双臂工位状态','指挥中心就绪'] },
  highlight:{ icon:'✨', title:'技术亮点 · 答辩速览', sub:'5 大实测亮点 · 双 X5 异构', ms:330, foot:'亮点',
    stages:['汇总 BPU 实测指标','加载双脑协同拓扑','核对真机实测数据','生成答辩亮点卡','亮点速览就绪'] },
  status:{ icon:'●', title:'公共状态', sub:'真机 / 镜像 / 回放 / 离线', ms:260, foot:'状态',
    stages:['读取 VPS 聚合状态','核对 live / mirror 来源','生成 24h / 7d 可用性条','载入近期事件','状态页就绪'] },
  ops:{ icon:'🩺', title:'运维总览 · 链路自愈', sub:'三机健康 + 故障转移', ms:320, foot:'运维',
    stages:['连接 Cloudflare 边缘','探活真机隧道 ×3','探活 VPS 镜像 ×3','核对 active health 状态','运维总览就绪'] },
  story:{ icon:'📖', title:'项目故事 · 从痛点到飞轮', sub:'15 个月 → 15 分钟', ms:330, foot:'项目故事',
    stages:['梳理实验室痛点','装载三机异构分工','绘制闭环自主飞轮','汇总生态位定位','项目故事就绪'] },
  mq:{ icon:'📋', title:'自主实验队列 · Mission Tracker', sub:'多阶段实验编排', ms:320, foot:'实验队列',
    stages:['拉取候选配方队列','分配五阶段流水线','同步三机执行回报','计算阶段进度','实验队列就绪'] },
  studio:{ icon:'🧬', title:'科学 Agent 工作室', sub:'设计 → 预测 → 执行 → 表征 → 回流', ms:320, foot:'生命周期系统',
    stages:['装载实验生命周期对象','读取工单 / trace_id','映射 agent 工具链','核对证据与边界','工作室就绪'] },
  fsd:{ icon:'🧭', title:'Lab-FSD v2 · 世界模型', sub:'BEV 占据 · 策略 token · 安全门禁', ms:330, foot:'世界模型',
    stages:['探测车载脑只读端点','装载 BEV occupancy 网格','滚动 future BEV 预测帧','映射 policy tokens','确认公网只读安全边界','Lab-FSD 控制台就绪'] },
  replay:{ icon:'🎞', title:'自主实验回放', sub:'10 阶段样品流 · 故障注入', ms:330, foot:'实验回放',
    stages:['读取 trace_id / 工单','装载 10 阶段实验时间线','同步车 / 双臂 / 执行器回放态','挂接 5 类故障注入','生成样品流证据表','回放就绪'] },
  command:{ icon:'🧭', title:'云端实验室指挥中心', sub:'任务队列 · 资源日历 · 风险锁', ms:330, foot:'云端指挥',
    stages:['聚合任务队列与工单','读取机群 serving 状态','计算设备资源预约','标注阻塞项 / 风险锁','挂接回放 / 孪生 / 可观测性','指挥中心就绪'] },
  defense:{ icon:'🛡', title:'答辩防御模式', sub:'脚本 · 证据链接 · 评委 checklist', ms:330, foot:'证据矩阵',
    stages:['装载 3/5/8 分钟答辩脚本','聚合创新点证据卡','生成真机 / 离线演示路径','核对公网安全边界','挂接 API / 页面证据入口','答辩防御就绪'] },
  benchmark:{ icon:'🏁', title:'全球对标控制台', sub:'顶级站横向对比 · 证据门禁', ms:330, foot:'全球对标',
    stages:['读取顶级站横向参照','映射本站页面与 API 证据','计算评分门禁','核对诚实边界与差距项','全球对标就绪'] },
  assets:{ icon:'🗂', title:'资产数字孪生 · 设备档案', sub:'三机 + 公网设施 全资产台账', ms:320, foot:'资产',
    stages:['加载资产注册表','挂接实时 serving 状态','拉取维保台账','联动三维场景','资产档案就绪'] },
  twin:{ icon:'🌐', title:'实验室全景孪生', sub:'真尺寸 · 真运动学 · 遥测驱动', ms:330, foot:'数字孪生',
    stages:['装载 myCobot 280 真 DH 链','铺设工位/车区/机架场景','订阅 /api/twin 遥测流','逐帧正运动学求解','孪生场景就绪'] },
  lab:{  icon:'🧠', title:'AI 脑 · 智能计算平台', sub:'NIR 荧光粉配方 AI 预测', ms:340, foot:'计算平台',
    stages:['唤醒 9 路本地 LLM','装载 5 BPU 推理槽','连接 NIR 预测引擎','载入 25228 文献向量','计算平台就绪'] },
  brain:{ icon:'🧠', title:'AI 脑解释 · Fly-MB', sub:'公开安全的材料预测证据链', ms:320, foot:'可解释性',
    stages:['读取 AI 脑健康摘要','加载 Fly-MB 稀疏推理链','核对 MLIP/TS 科学代理','标注 BPU/LLM 共识边界','解释系统就绪'] },
  car:{  icon:'🚗', title:'车载脑 · NavCockpit', sub:'Lab-FSD v2 · BEV 世界模型 · 安全门禁', ms:340, foot:'驾驶舱',
    stages:['初始化 WebGL 三维场景','连接遥测流 ws/telemetry','加载 Lab-FSD v2 BEV','校验安全门禁','连接 Nav2/MPPI 权威控制','驾驶舱就绪'] },
  arm:{  icon:'🦾', title:'机械臂 · WorkCockpit', sub:'双 myCobot 280 · 工位控制', ms:340, foot:'双臂工位',
    stages:['初始化双臂三维场景','连接工位 socket.io','加载 DH 正运动学','校验防撞 capsule 互锁','工位就绪'] },
};
let bootTimers=[], booting=false, bootAfter=null;
function runBoot(k, after){
  const b=BOOT[k]||BOOT.home;
  bootTimers.forEach(clearTimeout); bootTimers=[];
  const ov=document.getElementById('boot');
  ov.setAttribute('data-accent', k);
  document.getElementById('bIcon').textContent=b.icon;
  document.getElementById('bTitle').textContent=b.title;
  document.getElementById('bSub').textContent=b.sub;
  document.getElementById('bFoot').textContent=b.foot;
  const rows=document.getElementById('bRows'); rows.innerHTML='';
  b.stages.forEach((s)=>{ const r=document.createElement('div'); r.className='brow queued';
    r.innerHTML='<span class="bm">○</span><span class="bt"></span><span class="btag"></span>';
    r.querySelector('.bt').textContent=s; rows.appendChild(r); });
  document.getElementById('bFill').style.width='0%';
  ov.classList.remove('out'); ov.classList.add('show'); booting=true; bootAfter=after;
  function setRow(i){
    const all=rows.children;
    for(let j=0;j<all.length;j++){ const el=all[j];
      el.className='brow '+(j<i?'done':j===i?'active':'queued');
      const m=el.querySelector('.bm'), tg=el.querySelector('.btag');
      m.textContent = j<i?'●':j===i?'◐':'○'; m.className='bm'+(j===i?' spin':'');
      tg.textContent = j<i?'OK':j===i?'…':''; }
    document.getElementById('bFill').style.width=(i/b.stages.length*100)+'%';
  }
  setRow(0);
  const step=b.ms||340;
  for(let i=1;i<=b.stages.length;i++){
    bootTimers.push(setTimeout(()=>{ if(i<b.stages.length) setRow(i); else finishBoot(); }, step*i));
  }
}
function finishBoot(){
  if(!booting) return; booting=false;
  bootTimers.forEach(clearTimeout); bootTimers=[];
  const ov=document.getElementById('boot');
  document.getElementById('bFill').style.width='100%';
  ov.classList.add('out');
  setTimeout(()=>{ ov.classList.remove('show','out'); const a=bootAfter; bootAfter=null; if(a) a(); }, 420);
}
function skipBoot(){ if(booting) finishBoot(); }

function frameFallbackHtml(k){
  const m=FRAME_FALLBACK[k]||FRAME_FALLBACK.lab;
  return '<div class="ff-card">'+
    '<div class="ff-mark">'+m.icon+'</div>'+
    '<div class="ff-copy"><div class="ff-k">本地镜像驾驶舱 · 子域在线时可手动切入</div>'+
    '<h2>'+m.title+'</h2><p>'+m.sub+'</p>'+
    '<div class="ff-chips">'+m.chips.map(x=>'<span>'+x+'</span>').join('')+'</div>'+
    '<div class="ff-actions"><button data-frame="'+k+'" onclick="frameTryEmbed(this.dataset.frame)">切入子域</button><button class="ghost" data-frame="'+k+'" onclick="window.open(SYS[this.dataset.frame].url)">新窗口打开</button></div></div>'+
    '<div class="ff-orbit"><span></span><span></span><span></span></div></div>';
}

function setFrameOnline(k, online){
  const f=frameEls[k], fb=frameFallbackEls[k];
  if(!f || !fb) return;
  f.classList.toggle('ready', !!online);
  fb.classList.toggle('show', !online);
}

function probeFrame(k, url){
  const f=frameEls[k], fb=frameFallbackEls[k];
  if(!f || !fb) return;
  setFrameOnline(k, false);
  const done=(ok)=>{
    if(frameEls[k]!==f) return;
    fb.classList.toggle('reachable', !!ok);
    if(f.dataset.userEmbed==='1') setFrameOnline(k, !!ok);
    else setFrameOnline(k, false);
  };
  try{
    if(window.AbortController){
      const ctl=new AbortController();
      const t=setTimeout(()=>ctl.abort(), 3600);
      fetch(url,{method:'GET',mode:'no-cors',cache:'no-store',signal:ctl.signal})
        .then(()=>{ clearTimeout(t); done(true); })
        .catch(()=>{ clearTimeout(t); done(false); });
    }else{
      Promise.race([
        fetch(url,{method:'GET',mode:'no-cors',cache:'no-store'}).then(()=>true).catch(()=>false),
        new Promise(r=>setTimeout(()=>r(false),3600))
      ]).then(done);
    }
  }catch(e){ done(false); }
}

function frameTryEmbed(k){
  const f=frameEls[k];
  if(!f) return;
  f.dataset.userEmbed='1';
  const url=f.dataset.src||SYS[k].url;
  if(f.dataset.loaded!=='1' || f.src!==url){
    f.src=url;
    f.dataset.loaded='1';
  }
  setFrameOnline(k, true);
}

function ensureFrame(k, path){
  const url=SYS[k].url.replace(/\/$/,'')+(path||'/');
  if(frameEls[k]){
    const f=frameEls[k];
    f.dataset.src=url;
    if(path && frameEls[k]._path!==path){
      frameEls[k]._path=path;
      if(f.dataset.userEmbed==='1'){
        f.src=url;
        f.dataset.loaded='1';
        setFrameOnline(k, true);
      }else{
        setFrameOnline(k, false);
      }
      probeFrame(k, url);
    }
    return frameEls[k];
  }
  const f=document.createElement('iframe');
  f.className='frame'; f.src='about:blank'; f._path=path||'/';
  f.loading='lazy'; f.dataset.src=url; f.dataset.loaded='0';
  f.setAttribute('allow','fullscreen; clipboard-read; clipboard-write; autoplay');
  const fb=document.createElement('div');
  fb.className='frame-fallback frame-fallback-'+k+' show';
  fb.innerHTML=frameFallbackHtml(k);
  const host=document.getElementById('frames');
  host.appendChild(f);
  host.appendChild(fb);
  frameFallbackEls[k]=fb;
  frameEls[k]=f;
  probeFrame(k, url);
  return f;
}

const ROUTE_CLUSTER={
  home:'home', highlight:'home', defense:'home', benchmark:'home',
  brain:'research', lab:'research', mq:'research', studio:'research', atlas:'research', build:'research',
  fsd:'robot', replay:'robot', car:'robot', arm:'robot', fleet:'robot', tasks:'robot', command:'robot',
  twin:'twin', assets:'twin',
  status:'ops', ops:'ops', obs:'ops', logs:'ops', traces:'ops', topo:'ops', budget:'ops', inc:'ops', tm:'ops',
  noc:'ops', self:'ops', oee:'ops', alert:'ops', qms:'ops', cmms:'ops',
  sec:'security', release:'security', data:'security'
};
function routeCluster(k){ return ROUTE_CLUSTER[k]||(!PAGES[k]?'system':'other'); }
function routeMotion(from,to,opts){
  if(opts&&opts.motion) return opts.motion;
  if(to==='home') return 'home-enter';
  if(from==='home') return 'home-leave';
  if(routeCluster(from)===routeCluster(to)) return 'same-cluster';
  if(!PAGES[to]) return 'system-boot';
  return 'cross-cluster';
}
function setRouteTransitionState(from,to,opts){
  const b=document.body, motion=routeMotion(from,to,opts||{});
  b.dataset.routeFrom=from||'home'; b.dataset.routeTo=to||'home'; b.dataset.routeMotion=motion;
  b.dataset.routeCluster=routeCluster(to); b.classList.add('route-transitioning');
}
function clearRouteTransitionState(){
  document.body.classList.remove('route-transitioning','route-fallback');
}
function transitionGo(k, opts){
  opts=opts||{};
  const from=opts.from||cur;
  const reduced=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const apply=()=>{ revealView(k); if(opts.after) opts.after(); };
  setRouteTransitionState(from,k,opts);
  site24MotionState('routing');
  if(document.startViewTransition && !reduced){
    try{
      const vt=document.startViewTransition(apply);
      vt.finished.finally(()=>{ clearRouteTransitionState(); site24MotionState('idle'); });
      return;
    }catch(e){}
  }
  document.body.classList.add('route-fallback');
  apply();
  setTimeout(()=>{ clearRouteTransitionState(); site24MotionState('idle'); }, 520);
}
function site24MotionState(state){
  document.body.dataset.motionState=state||'idle';
  if(state==='routing'){
    clearTimeout(site24MotionState._t);
    site24MotionState._t=setTimeout(()=>site24MotionState('idle'), 720);
  }
}
document.addEventListener('visibilitychange', ()=>{
  site24MotionState(document.hidden?'hidden':'idle');
});
function go(k, opts){
  opts=opts||{};
  if(k===cur && !opts.force){ if(opts.path) ensureFrame(k, opts.path); if(opts.after) opts.after(); return; }
  const from=cur;
  if(!PAGES[k]){
    ensureFrame(k, opts.path);   // 系统视图: 动画背后预加载真机/镜像 (可深链子路由)
    runBoot(k, ()=>{ revealView(k); if(opts.after) opts.after(); });
    return;
  }
  if(opts.boot){
    runBoot(k, ()=>transitionGo(k,{from:from,after:opts.after,motion:'system-boot'}));
    return;
  }
  transitionGo(k,{from:from,after:opts.after,motion:opts.motion});
}
function revealView(k){
  cur=k;
  document.body.dataset.view=k;
  if(window.livingSetView) window.livingSetView(k);
  document.querySelectorAll('#nav .pill').forEach(p=>p.classList.toggle('active', p.dataset.k===k));
  document.querySelectorAll('.ia-chip,.mtab').forEach(p=>p.classList.toggle('active', p.dataset.k===k));
  document.getElementById('overview').classList.toggle('show', k==='home');
  if(k==='home') loadThreeSceneOnce();
  document.getElementById('highlights').classList.toggle('show', k==='highlight');
  document.getElementById('statusview').classList.toggle('show', k==='status');
  if(k==='status') loadPublicStatus();
  document.getElementById('fleetview').classList.toggle('show', k==='fleet');
  if(k==='fleet') loadFleetConsole();
  document.getElementById('tasksview').classList.toggle('show', k==='tasks');
  if(k==='tasks') loadTasksConsole();
  document.getElementById('opsview').classList.toggle('show', k==='ops');
  document.getElementById('storyview').classList.toggle('show', k==='story');
  document.getElementById('mqview').classList.toggle('show', k==='mq');
  document.getElementById('studioview').classList.toggle('show', k==='studio');
  if(k==='studio') loadAgentStudio();
  document.getElementById('fsdview').classList.toggle('show', k==='fsd');
  if(k==='fsd') loadFsdConsole();
  document.getElementById('replayview').classList.toggle('show', k==='replay');
  if(k==='replay') loadExperimentReplay();
  document.getElementById('commandview').classList.toggle('show', k==='command');
  if(k==='command') loadCloudCommand();
  document.getElementById('defenseview').classList.toggle('show', k==='defense');
  if(k==='defense') loadDefenseMode();
  document.getElementById('benchmarkview').classList.toggle('show', k==='benchmark');
  if(k==='benchmark') loadGlobalBenchmark();
  document.getElementById('assetsview').classList.toggle('show', k==='assets');
  document.getElementById('twinview').classList.toggle('show', k==='twin');
  document.getElementById('preflightview').classList.toggle('show', k==='preflight');
  if(k==='preflight') loadPreflight();
  document.getElementById('atlasview').classList.toggle('show', k==='atlas');
  if(k==='atlas') loadAtlas();
  document.getElementById('brainview').classList.toggle('show', k==='brain');
  if(k==='brain') loadBrainExplain();
  document.getElementById('detailview').classList.toggle('show', k==='detail');
  if(k==='detail') loadDetail();
  document.getElementById('archiveview').classList.toggle('show', k==='archive');
  if(k==='archive') loadArchive();
  document.getElementById('modelsview').classList.toggle('show', k==='models');
  if(k==='models') loadModels();
  document.getElementById('standardsview').classList.toggle('show', k==='standards');
  if(k==='standards') loadStandards();
  document.getElementById('costview').classList.toggle('show', k==='cost');
  if(k==='cost') loadCost();
  document.getElementById('syncview').classList.toggle('show', k==='sync');
  if(k==='sync') loadMirrorSync();
  document.getElementById('reproview').classList.toggle('show', k==='repro');
  if(k==='repro') loadReproduce();
  document.getElementById('arview').classList.toggle('show', k==='ar');
  if(k==='ar') loadAR();
  document.getElementById('obsview').classList.toggle('show', k==='obs');
  if(k==='obs') loadObs(); else obsStop();
  document.getElementById('tracesview').classList.toggle('show', k==='traces');
  if(k==='traces') loadTracesConsole();
  document.getElementById('topoview').classList.toggle('show', k==='topo');
  if(k==='topo') loadTopology(); else topoStop();
  document.getElementById('sloview').classList.toggle('show', k==='budget');
  if(k==='budget') loadSloBudget();
  document.getElementById('incview').classList.toggle('show', k==='inc');
  if(k==='inc') loadIncidents();
  document.getElementById('timeview').classList.toggle('show', k==='tm');
  if(k==='tm') loadTimemachine(); else tmStop();
  document.getElementById('nocview').classList.toggle('show', k==='noc');
  if(k==='noc') loadNoc(); else nocStop();
  document.getElementById('selfview').classList.toggle('show', k==='self');
  if(k==='self') loadSelf(); else iStop('self');
  document.getElementById('oeeview').classList.toggle('show', k==='oee');
  if(k==='oee') loadOee(); else iStop('oee');
  document.getElementById('alertview').classList.toggle('show', k==='alert');
  if(k==='alert') loadAlert();
  document.getElementById('logsview').classList.toggle('show', k==='logs');
  if(k==='logs') loadLogs();
  document.getElementById('qmsview').classList.toggle('show', k==='qms');
  if(k==='qms') loadQms();
  document.getElementById('cmmsview').classList.toggle('show', k==='cmms');
  if(k==='cmms') loadCmms();
  document.getElementById('secview').classList.toggle('show', k==='sec');
  if(k==='sec') loadSec();
  document.getElementById('releaseview').classList.toggle('show', k==='release');
  if(k==='release') loadRelease();
  document.getElementById('dataview').classList.toggle('show', k==='data');
  if(k==='data') loadData();
  document.getElementById('glossaryview').classList.toggle('show', k==='glossary');
  if(k==='glossary') loadGlossary();
  document.getElementById('changelogview').classList.toggle('show', k==='changelog');
  if(k==='changelog') loadChangelog();
  document.getElementById('buildview').classList.toggle('show', k==='build');
  if(k==='build') loadBuild();
  document.getElementById('importview').classList.toggle('show', k==='importw');
  if(k==='importw') loadImport();
  document.getElementById('elnview').classList.toggle('show', k==='eln');
  if(k==='eln') loadEln();
  if(k==='ops'){ pollOps(); loadEvents(); loadAlarms(); loadSlo(); loadReports(); }
  if(k==='mq') renderMissions();
  if(k==='assets') loadAssets();
  if(k==='twin'){ if(window.twinShow) twinShow(); } else if(window.twinHide) twinHide();
  const frameOn = !PAGES[k];
  Object.entries(frameEls).forEach(([kk,ff])=>ff.classList.toggle('show', frameOn && kk===k));
  Object.entries(frameFallbackEls).forEach(([kk,fb])=>fb.classList.toggle('active', frameOn && kk===k));
}

function fullscreenActive(){
  if(PAGES[cur]){ toast('先进入一个系统再全屏'); return; }
  const f=frameEls[cur];
  if(f && f.requestFullscreen) f.requestFullscreen().catch(()=>window.open(SYS[cur].url,'_blank'));
  else window.open(SYS[cur].url,'_blank');
}

/* ---- 大屏轮播 (kiosk) — 展台无人值守自动巡播 ---- */
const KIOSK={ on:false, idx:0, timer:null, paused:false, dwell:13000,
  pl:[ {k:'home',t:'三机异构总览'}, {k:'highlight',t:'技术亮点 · 5 大实测'},
       {k:'twin',t:'数字孪生 · 真运动学'}, {k:'lab',t:'AI 脑 · 配方预测'},
       {k:'car',t:'车载脑 · 导航控制'}, {k:'arm',t:'机械臂 · 双臂工位'},
       {k:'mq',t:'批次工单闭环'} ] };
function kioskStart(){
  if(KIOSK.on) return; KIOSK.on=true; KIOSK.idx=0; KIOSK.paused=false;
  document.body.classList.add('kiosk');
  const el=document.documentElement;
  if(el.requestFullscreen) el.requestFullscreen().catch(()=>{});
  document.getElementById('kioskBar').classList.add('show');
  kioskHop(0); kioskArm();
  toast('🖥 大屏轮播已开启 — 点退出 / Esc 结束');
}
function kioskArm(){ clearTimeout(KIOSK.timer);
  const f=document.getElementById('kioskFill'); if(f){ f.style.transition='none'; f.style.width='0%';
    requestAnimationFrame(()=>{ f.style.transition='width '+KIOSK.dwell+'ms linear'; f.style.width='100%'; }); }
  if(!KIOSK.paused) KIOSK.timer=setTimeout(()=>kioskNext(), KIOSK.dwell);
}
function kioskHop(i){ const it=KIOSK.pl[i]; go(it.k);
  document.getElementById('kioskNow').textContent=it.t;
  const nx=KIOSK.pl[(i+1)%KIOSK.pl.length]; document.getElementById('kioskNext').textContent='下一站 · '+nx.t;
  const dd=document.getElementById('kioskDots'); if(dd){ dd.innerHTML='';
    KIOSK.pl.forEach((_,j)=>{ const s=document.createElement('span'); s.className='kdot'+(j===i?' on':''); dd.appendChild(s); }); } }
function kioskNext(){ if(!KIOSK.on) return; KIOSK.idx=(KIOSK.idx+1)%KIOSK.pl.length; kioskHop(KIOSK.idx); kioskArm(); }
function kioskPause(){ KIOSK.paused=!KIOSK.paused;
  document.getElementById('kioskPlay').textContent=KIOSK.paused?'▶ 继续':'⏸ 暂停';
  if(KIOSK.paused){ clearTimeout(KIOSK.timer); const f=document.getElementById('kioskFill'); if(f){ const w=getComputedStyle(f).width; f.style.transition='none'; f.style.width=w; } }
  else kioskArm(); }
function kioskStop(){ if(!KIOSK.on) return; KIOSK.on=false; clearTimeout(KIOSK.timer);
  document.body.classList.remove('kiosk');
  document.getElementById('kioskBar').classList.remove('show');
  if(document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(()=>{}); }

/* ---- 桌面通知 (浏览器 Notification) — 新 crit/warn 告警弹窗 ---- */
const NOTIFY={ on: localStorage.getItem('xrd_notify')==='1', seen:new Set() };
async function notifyToggle(){
  if(NOTIFY.on){ NOTIFY.on=false; localStorage.removeItem('xrd_notify'); notifySync(); toast('🔕 桌面通知已关'); return; }
  if(!('Notification' in window)){ toast('此浏览器不支持桌面通知'); return; }
  let p=Notification.permission;
  if(p==='default') p=await Notification.requestPermission();
  if(p!=='granted'){ toast('未授予通知权限 (浏览器已拦截)'); return; }
  NOTIFY.on=true; localStorage.setItem('xrd_notify','1'); notifySync();
  toast('🔔 桌面通知已开 — 新告警将弹窗'); notifyCheck();
}
function notifySync(){ const b=document.getElementById('notifyBtn'); if(b) b.classList.toggle('on', NOTIFY.on); }
async function notifyCheck(){
  if(!NOTIFY.on || !('Notification' in window) || Notification.permission!=='granted') return;
  try{
    const d=await fetch('/api/alarms',{cache:'no-store'}).then(r=>r.json());
    const act=(d.active||[]);
    // 首次只记录基线 (避免开启瞬间把历史告警一次性弹出)
    if(!NOTIFY._init){ act.forEach(a=>NOTIFY.seen.add(a.id)); NOTIFY._init=true; return; }
    act.forEach(a=>{ if(NOTIFY.seen.has(a.id)) return; NOTIFY.seen.add(a.id);
      if(a.severity==='info') return;
      try{ const n=new Notification('🚨 智研指挥中心 · '+(a.severity==='crit'?'严重':'预警'),
        { body:a.message||'链路告警', tag:'xrd-al-'+a.id, icon:'/icon.svg' });
        n.onclick=()=>{ window.focus(); go('ops'); n.close(); }; }catch(e){} });
  }catch(e){}
}
setInterval(notifyCheck, 30000);

/* ---- H7 通知中心 (告警 · 事件 · 大事记) + favicon 角标 ---- */
let NC_OPEN=false, NC_TAB='live';
const NC_MILES=[
  {d:'2026-06-11', t:'平台公网上线', s:'Cloudflare + 香港 VPS Caddy + frp 隧道, 三系统子域 + SSO 单点登录'},
  {d:'2026-06-12', t:'三机真机部署 + 统一门户', s:'车载脑 cockpit_bridge / 主机械臂 arm01 真机 / apex 指挥中心一处统管'},
  {d:'2026-06-13', t:'工业级全栈 + UI 质感升级', s:'historian/ISA-18.2 告警/ISA-88 工单/SLO + 数字孪生 + 演示就绪预检 + 运维副驾'},
];
function ncToggle(){
  NC_OPEN=!NC_OPEN;
  document.getElementById('ncMask').classList.toggle('show', NC_OPEN);
  if(NC_OPEN) ncLoad();
}
function ncTab(t){ NC_TAB=t; NC_DIAG=false;
  document.querySelectorAll('.nc-tab').forEach(x=>x.classList.toggle('on', x.dataset.t===t));
  ncLoad();
}
let NC_DIAG=false;
function ncDiagToggle(){ NC_DIAG=!NC_DIAG; ncLoad(); }
async function ncDiagnose(){
  const body=document.getElementById('ncBody'), foot=document.getElementById('ncFoot');
  try{
    const d=await fetch('/api/diagnose',{cache:'no-store'}).then(r=>r.json());
    const ds=d.diagnoses||[];
    let head='<button class="nc-diagbtn on" onclick="ncDiagToggle()">🔍 AI 诊断 (返回列表)</button>';
    if(!ds.length){ body.innerHTML=head+'<div class="oe-empty" style="padding:36px 0">无活动告警可诊断 — 全链路健康 ✓</div>';
      foot.textContent='AI 诊断 · 0 条'; return; }
    body.innerHTML=head+ds.map(o=>'<div class="nc-diag '+(o.severity||'info')+'">'+
      '<div class="ndg-h"><span class="nc-sev"></span><b>'+esc(o.message)+'</b></div>'+
      '<div class="ndg-cause"><span>根因</span>'+esc(o.cause)+'</div>'+
      '<div class="ndg-steps"><span>处置</span><ol>'+o.steps.map(s=>'<li>'+esc(s)+'</li>').join('')+'</ol></div>'+
      '<div class="ndg-rel">↳ 关联检查: '+esc(o.related)+'</div></div>').join('');
    foot.textContent='AI 诊断 · '+ds.length+' 条 (crit '+(d.summary.crit||0)+' · 设计内降级 '+(d.summary.design_intended||0)+')';
  }catch(e){ body.innerHTML='<div class="oe-empty">诊断失败</div>'; }
}
function ncTime(ts){ const t=new Date(ts*1000);
  return t.toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'})+' '+t.toLocaleTimeString('zh-CN',{hour12:false}); }
async function ncLoad(){
  const body=document.getElementById('ncBody'), foot=document.getElementById('ncFoot');
  if(NC_TAB==='miles'){
    body.innerHTML='<div class="nc-timeline">'+NC_MILES.slice().reverse().map(m=>
      '<div class="nc-mile"><div class="nc-mdot"></div><div class="nc-mc"><div class="nc-md">'+esc(m.d)+'</div>'+
      '<b>'+esc(m.t)+'</b><span>'+esc(m.s)+'</span></div></div>').join('')+'</div>';
    foot.textContent=NC_MILES.length+' 个里程碑 · 平台演进史';
    return;
  }
  try{
    if(NC_TAB==='alarm'){
      if(NC_DIAG){ ncDiagnose(); return; }
      const d=await fetch('/api/alarms',{cache:'no-store'}).then(r=>r.json());
      const act=d.active||[];
      const diagBtn='<button class="nc-diagbtn" onclick="ncDiagToggle()">🔍 AI 诊断 (根因 + 处置)</button>';
      if(!act.length) body.innerHTML=diagBtn+'<div class="oe-empty" style="padding:36px 0">无活动告警 — 全链路健康 ✓</div>';
      else body.innerHTML=diagBtn+act.map(a=>'<div class="nc-it '+(a.severity||'info')+'">'+
        '<span class="nc-sev"></span><div class="nc-itc"><b>'+esc(a.message)+'</b>'+
        '<span class="nc-meta">'+ncTime(a.ts_raised)+' · '+esc(a.rule)+(a.ts_ack?' · 已确认':'')+'</span></div>'+
        (a.ts_ack?'':'<button class="nc-ack" onclick="ackAlarm('+a.id+');setTimeout(ncLoad,400)">确认</button>')+'</div>').join('');
      foot.textContent=act.length+' 条活动告警 · crit '+(d.counts.crit||0)+' / warn '+(d.counts.warn||0);
    } else {
      const d=await fetch('/api/events?hours=72&limit=60',{cache:'no-store'}).then(r=>r.json());
      const ev=d.events||[];
      if(!ev.length) body.innerHTML='<div class="oe-empty" style="padding:40px 0">暂无事件 — 三机链路稳定</div>';
      else body.innerHTML=ev.map(e=>'<div class="nc-it '+(e.severity||'info')+'">'+
        '<span class="nc-sev"></span><div class="nc-itc"><b>'+esc(e.message)+'</b>'+
        '<span class="nc-meta">'+ncTime(e.ts)+(e.sys?(' · '+(EV_SYS[e.sys]||e.sys)):'')+'</span></div></div>').join('');
      foot.textContent=ev.length+' 条事件 · 近 72 小时 (historian)';
    }
  }catch(e){ body.innerHTML='<div class="oe-empty">加载失败</div>'; }
}
// favicon 动态角标: 有未确认 crit/warn 时在图标右上画红/黄点
let _favBase=null;
function setFaviconBadge(counts){
  try{
    const c=document.createElement('canvas'); c.width=c.height=64; const x=c.getContext('2d');
    // 底: 紫蓝圆 + 六边形 (与品牌一致)
    const g=x.createLinearGradient(0,0,64,64); g.addColorStop(0,'#7c3aed'); g.addColorStop(1,'#2563eb');
    x.fillStyle=g; x.beginPath(); x.arc(32,32,30,0,7); x.fill();
    x.fillStyle='#fff'; x.font='bold 30px sans-serif'; x.textAlign='center'; x.textBaseline='middle';
    x.fillText('⬢',32,34);
    const n=(counts&&counts.unacked)||0;
    if(n){ x.fillStyle=(counts.crit?'#e11d48':'#d97706'); x.beginPath(); x.arc(50,14,14,0,7); x.fill();
      x.fillStyle='#fff'; x.font='bold 20px sans-serif'; x.fillText(n>9?'9+':String(n),50,15); }
    let link=document.querySelector('link[rel="icon"]');
    if(!link){ link=document.createElement('link'); link.rel='icon'; document.head.appendChild(link); }
    link.type='image/png'; link.href=c.toDataURL('image/png');
  }catch(e){}
}

/* ---- fleet 状态 ---- */
const _miss={lab:0,car:0,arm:0};
async function pollFleet(){
  try{
    const r=await fetch('/api/fleet',{cache:'no-store'}); const d=await r.json();
    ['lab','car','arm'].forEach(k=>{
      const s=d[k]||{};
      if(s.online){ _miss[k]=0; } else { _miss[k]++; }
      const on = s.online || _miss[k]<2;
      const nd=document.getElementById('d-'+k); if(nd){ nd.className='dot '+(on?'on':'off'); }
      const sd=document.getElementById('s-'+k); if(sd){ sd.className='dot '+(on?'on':'off'); }
      const tx=document.getElementById('t-'+k);
      if(tx){ tx.textContent = on ? '真机在线' : '镜像演示 (设备未上电)'; tx.style.color = on?'#059669':'#8593ad'; }
      const cc=document.getElementById('c-'+k);
      if(cc && s.online){ cc.innerHTML=''; const m=s.metrics||{};
        Object.entries(m).forEach(([kk,vv])=>{ const e=document.createElement('span'); e.className='chip'; e.textContent=kk+': '+vv; cc.appendChild(e); }); }
      else if(cc && !on){ cc.innerHTML=''; }
    });
    const ts=document.getElementById('fleetTs');
    const n=['lab','car','arm'].filter(k=>d[k]&&d[k].online).length;
    ts.textContent = `三机在线 ${n}/3 · ${new Date().toLocaleTimeString('zh-CN')}`;
  }catch(e){}
}
setInterval(pollFleet, 5000); pollFleet();

/* ---- Site9 R2: public status ---- */
function psClass(status){
  const s=String(status||'unknown').toLowerCase();
  if(s==='operational') return 'ok';
  if(s==='mirror'||s==='replay'||s==='degraded') return 'warn';
  if(s==='offline') return 'crit';
  return 'unknown';
}
function uiLang(){
  try{ return localStorage.getItem('xrd_lang')||'zh'; }catch(e){ return 'zh'; }
}
function uiText(zh,en){ return uiLang()==='en'?en:zh; }
function uiTerm(v){
  const s=String(v==null?'':v);
  if(uiLang()==='en') return s;
  const k=s.toLowerCase();
  const m={
    'unknown':'未知','source unknown':'来源未知','read-only':'只读','safe explanation':'安全解释',
    'open':'进行中','closed':'已关闭','done':'已完成','cancelled':'已取消','pending':'待处理',
    'pass':'通过','fail':'失败','review':'复核','required':'必需','active':'活动','available':'可用',
    'enabled':'启用','disabled':'禁用','blocked':'阻塞','route':'路线','ego':'本车','shadow':'影子规划',
    'clear':'无阻塞','direct':'直连','fallback':'兜底','explicit':'显式标注','not public':'非公开',
    'stopped':'已停止','records':'记录','platform':'平台','event':'事件','info':'信息','warn':'警告','crit':'严重',
    'offline':'离线','mirror':'镜像','replay':'回放','live':'真机','history':'历史','stale':'陈旧','mock':'模拟','documented/replay':'文档/回放',
    'live/mirror/replay':'真机/镜像/回放','schedule':'排程','page/api/source':'页面/API/来源',
    'pass/fail':'通过/失败','labelled':'已标注','3/5/8 min':'3/5/8 分钟',
    'benchmarked':'已对标','official refs':'官方参照','not hidden':'未隐藏','public safe':'公开安全',
    'source':'来源','source labels':'来源标签','overall score':'总评分','benchmark dimensions':'对标维度',
    'score gates':'评分门禁','final gate':'最终门禁','work orders':'工单','wo_log events':'工单日志事件',
    'open tasks':'进行中任务','scheduled resources':'已排程资源','dependency blockers':'依赖阻塞',
    'active alarms':'活动告警','tasks':'任务','replay':'回放','twin':'孪生','assets':'资产','observability':'可观测性',
    'page':'页面','api':'API','open wo':'打开工单','open material':'打开材料','inspect':'查看',
    'serving source':'serving 来源','safety gate':'安全门禁','public fsd endpoints':'公开 FSD 端点',
    'public control policy':'公网控制策略','bev occupancy':'BEV 占据网格','policy tokens':'策略 Token',
    'future bev rollout':'未来 BEV 滚动预测','read-only endpoints':'只读端点',
    'sensor / model stack':'传感器 / 模型栈','public safety boundary':'公网安全边界',
    'no public gate reason exposed.':'未公开安全门禁原因。',
    'read-only / no chassis velocity command':'只读 / 不下发底盘速度命令',
    'vehicle brain endpoint is offline; public console holds all motion.':'车载脑端点离线；公网控制台保持所有运动锁定。',
    'free occupancy cell':'空闲占据格','sample corridor':'样品通道','current ego cell':'当前本车格',
    'predicted route':'预测路线','dock goal':'对接目标','probabilistic shadow':'概率影子区',
    'bench edge clearance':'台面边缘间隙','hot-zone boundary':'热区边界','ld14 sweep':'LD14 扫描','astra depth frustum':'Astra 深度视锥',
    'hold current corridor':'保持当前通道','slow approach to sample dock':'低速接近样品对接点',
    'yield near arm sweep zone':'机械臂扫掠区附近让行','dock only after safety gate pass':'仅在安全门禁通过后对接',
    'nominal':'正常','shadow monitored':'影子规划监控','operator review':'操作员复核','hard gate':'硬门禁',
    'bev free-space corridor is public-display safe':'BEV 自由空间通道可安全公开展示',
    'unknown cells stay in review instead of becoming motion permission':'未知栅格保持复核态，不转换为运动许可',
    'dual-arm sweep zone is represented as a protected boundary':'双臂扫掠区以受保护边界呈现',
    'safety gate remains the final read-only decision shown publicly':'安全门禁保持为公网展示的最终只读决策',
    'public site never publishes robot motion commands':'公网永不发布机器人运动命令',
    '2d obstacle belt':'2D 障碍带','near-field occupancy':'近场占据','visual context / anomaly hints':'视觉上下文 / 异常提示',
    'future bev rollout':'未来 BEV 滚动预测','edge anomaly score, measured about 1.7ms on x5':'边缘异常分数，X5 实测约 1.7ms',
    'authoritative chassis control':'底盘权威控制','serving':'服务中','summarized':'摘要态','architecture metric':'架构指标',
    'not controlled by public site':'不由公网页面控制','documented':'文档化','policy':'策略',
    'this is lab-fsd v2 for a laboratory robot; it borrows world-model interface patterns but does not claim to replicate tesla fsd.':'这是面向实验室机器人的 Lab-FSD v2；借鉴世界模型界面范式，但不声称复刻 Tesla FSD。',
    'the public page is read-only and never emits chassis velocity, arm motion, lift, actuator, magnet, or servo commands.':'公网页面只读，绝不发出底盘速度、机械臂、升降台、电推杆、电磁铁或舵机命令。',
    'nav2/mppi and embedded safety layers remain the chassis-control authority.':'Nav2/MPPI 与嵌入式安全层仍是底盘控制权威。',
    'every panel is labelled live, mock, replay, offline, documented, or policy.':'每个面板都标注真机、模拟、回放、离线、文档或策略来源。',
    'sample':'样品','sample path':'样品路径','sample state':'样品状态','trace / object':'trace / 对象',
    'formula':'配方','fault injection':'故障注入','sample flow map':'样品流地图','event stack':'事件栈',
    'public command':'公网命令','normal':'正常','normal replay':'正常回放','stage':'阶段','system':'系统',
    'zone':'区域','recovery':'恢复','replay data table':'回放数据表','no traces':'暂无追踪',
    'not linked':'未关联','request':'请求','public metrics unavailable':'公开指标不可用',
    'fleet cockpit unavailable':'机群驾驶舱不可用','tasks unavailable':'任务不可用','no tasks':'暂无任务',
    'status api unreachable; showing local shell only.':'状态 API 不可达；仅显示本地外壳。',
    'device offline':'设备离线','sample missing':'样品缺失','route blocked':'路线阻塞','arm station busy':'机械臂工位忙碌','characterization mismatch':'表征失配',
    'vehicle or workstation heartbeat drops to mirror/offline':'车载脑或工位心跳降为镜像/离线',
    'hold sample state, switch public display to replay, require operator restore before execution resumes':'保持样品状态，将公网展示切到回放，执行恢复前需要操作员恢复',
    'vision/weight evidence does not confirm powder or bottle at the expected station':'视觉/重量证据未确认粉末或瓶子位于预期工位',
    'stop actuator sequence, ask operator to reseat sample, keep work order open with evidence note':'停止执行器序列，请操作员重新放置样品，工单保持打开并附证据备注',
    'bev occupancy marks the sample corridor as blocked':'BEV 占据标记样品通道被阻塞',
    'shadow planner yields; nav2/mppi remains authority and no public command is emitted':'影子规划让行；Nav2/MPPI 保持权威控制，公网不发命令',
    'dual-arm sweep zone or interlock state is not clear':'双臂扫掠区或互锁状态未清空',
    'queue waits at the station boundary and replays the latest safe arm pose':'队列在工位边界等待，并回放最近安全机械臂姿态',
    'xrd/pl observation diverges from the prediction evidence band':'XRD/PL 观测偏离预测证据波段',
    'close with mismatch note, attach actual lambda/xrd evidence, create next-round revise candidate':'用失配备注收单，附实测 lambda/XRD 证据，并创建下一轮 REVISE 候选',
    'work_order':'工单','candidate_material':'候选材料','public material schema replay':'公开材料 schema 回放',
    'sinter':'烧结','furnace':'炉端','documented replay':'文档化回放','lab-fsd replay':'Lab-FSD 回放',
    'work_order + wo_log':'工单 + 工单日志','ai brain rack':'AI 脑机架','car dock':'车载停靠点',
    'arm workstation':'机械臂工位','vps / cloudflare':'VPS / Cloudflare','embodied brain':'具身脑',
    'arm02 standby':'arm02 待机','xrd line':'XRD 线','pl line':'PL 线','vehicle brain':'车载脑',
    'ai candidate design':'AI 候选设计','ai prediction verdict':'AI 预测判读','queue and approval':'队列与审批',
    'vehicle route to station':'车载路线到工位','arm01 pick and fixture':'arm01 取放与固定','arm02 powder handling':'arm02 粉末处理',
    'lift / actuator / servo / magnet':'升降台 / 电推杆 / 舵机 / 电磁铁','sintering and furnace wait':'烧结与炉端等待',
    'xrd characterization':'XRD 表征','pl feedback and failure loop':'PL 回填与失败回流',
    'ai brain x5':'AI 脑 X5','stm32f407 layer':'STM32F407 层','xrd line':'XRD 线','pl line + ai brain':'PL 线 + AI 脑',
    'formula object + target band':'配方对象 + 目标波段','trace_id or material object':'trace_id 或材料对象',
    'ts/mlip/failure flags + r1 verdict':'TS/MLIP/失败标志 + R1 判读','prediction trace':'预测 trace',
    'work order enters staged lifecycle':'工单进入分阶段生命周期','sample corridor, bev occupancy, route hold':'样品通道、BEV 占据、路线保持',
    'bottle / bowl / holder gripper event':'瓶 / 碗 / 支架夹爪事件','joint pose + interlock':'关节姿态 + 互锁',
    'grind, guide, fill, transfer assist':'研磨、导向、灌装、转移辅助','joint pose + operation log':'关节姿态 + 操作日志',
    'lift stage, actuator extend/retract, servo angle, magnet hold':'升降台、电推杆伸缩、舵机角度、电磁铁保持',
    'ascii protocol replay':'ASCII 协议回放','temperature profile and hot-zone lock':'温度曲线与热区锁',
    'protocol template':'协议模板','pattern capture and phase check':'谱图采集与物相检查','xrd evidence object':'XRD 证据对象',
    'lambda_obs, mismatch, next candidate':'lambda_obs、失配、下一候选','actual feedback':'实测回填',
    'candidate object selected':'候选对象已选择','prediction verdict and uncertainty attached':'预测判读与不确定性已附加',
    'work order stage record created':'工单阶段记录已创建','route replay and safety gate shown':'路线回放与安全门禁已展示',
    'fixture and bottle/powder pickup pose':'夹具与瓶/粉取放姿态','grind / guide / fill assist pose':'研磨 / 导向 / 灌装辅助姿态',
    'lift stage + actuator + servo + electromagnet sequence':'升降台 + 电推杆 + 舵机 + 电磁铁序列',
    'sintering protocol and hot-zone lock':'烧结协议与热区锁','phase check object generated':'物相检查对象已生成',
    'lambda_obs backfill and failure learning':'lambda_obs 回填与失败学习','fault injection':'故障注入',
    'lift stage':'升降台','linear actuator':'电推杆','servo yaw':'舵机偏航','electromagnet':'电磁铁',
    'tim8_ch4 pul + pd7 dir + pd10 en':'TIM8_CH4 PUL + PD7 DIR + PD10 EN',
    'pc13/pc0 relay profile':'PC13/PC0 继电器曲线','pb8 tim4_ch3 50hz pwm':'PB8 TIM4_CH3 50Hz PWM',
    'pe0 hold/release state':'PE0 保持/释放状态',
    'replay is explanatory and read-only; it does not trigger car, arm, lift, magnet, actuator, or servo commands.':'回放仅用于解释且只读；不会触发车、臂、升降台、电磁铁、电推杆或舵机命令。',
    'live, mock, replay, stale, offline, and documented sources stay labelled at panel level.':'真机、模拟、回放、陈旧、离线和文档来源在面板级保持标注。',
    'fault injection changes the public replay narrative only; it is not injected into real robot services.':'故障注入只改变公网回放叙事，不注入真实机器人服务。',
    'no injected fault. use the fault chips above to show how the system holds state and recovers without public robot commands.':'未注入故障。可用上方故障按钮展示系统如何保持状态并恢复，且不开放公网机器人命令。',
    'public object not found':'未找到公开对象','no public object matched this id.':'没有匹配这个 ID 的公开对象。',
    'detail api is unavailable.':'详情 API 当前不可用。','not_found':'未找到','lookup':'查询',
    'no public material or prediction object matched this id.':'没有匹配这个 ID 的公开材料或预测对象。',
    'no public detail record':'没有公开详情记录',
    'try opening a row from materials explorer, or check whether the trace_id is present in public history.':'请从材料图鉴打开一行，或检查 trace_id 是否存在于公开历史中。',
    'back to explorer':'返回材料图鉴','unknown material':'未知材料','verdict':'判读','state':'状态','version':'版本',
    'cite this prediction':'引用这条预测','unavailable':'不可用','provenance':'来源溯源','evidence score':'证据评分',
    'no blocker':'无阻塞','no active public task in this lane.':'本泳道暂无公开任务。',
    'no public work orders or material objects yet.':'暂无公开工单或材料对象。',
    'candidate material object':'候选材料对象','not queued':'未入队',
    'not yet queued as a work order':'尚未创建为工单',
    'open material evidence, then create a work order when execution is needed':'先查看材料证据，需要执行时再创建工单',
    'public rows':'公开行','lambda pending':'lambda 待估计','data source':'数据源',
    'materials object schema':'材料对象结构','public source labels are preserved.':'公开来源标签已保留。',
    'no matched public material rows':'没有匹配的公开材料行','no matched candidate cards':'没有匹配候选卡片',
    'atlas failed to load':'图鉴加载失败','material filter':'材料筛选','prediction detail':'预测详情',
    'material detail':'材料详情','loading public research object...':'正在加载公开科研对象...',
    'missing material or trace id.':'缺少材料或 trace id。',
    'public rows are explicitly labelled live, mirror, history, curated, down/offline. the ui must never turn a stale row into a live claim.':'公开行会明确标注真机、镜像、历史、策展、down/offline；UI 绝不把陈旧行渲染成真机主张。',
    'formula, host, dopant, site, trace_id and work_order are normalized into one citable row.':'化学式、宿主、掺杂、位点、trace_id 和工单被规范成一条可引用记录。',
    'mattergen candidates are summarized with mattersim/chgnet stability fields when present.':'MatterGen 候选在存在时汇总 MatterSim/CHGNet 稳定性字段。',
    'lambda_em and band use the differentiable tanabe-sugano optical proxy or labelled observed history.':'lambda_em 和波段来自可微 Tanabe-Sugano 光学代理或已标注的实测历史。',
    'rows preserve live/mirror/history/curated labels; no offline row is rendered as live.':'行级数据保留真机/镜像/历史/策展标签；离线行不会被渲染成真机。',
    'formula object':'配方对象','mlip stability':'MLIP 稳定性','ts optical proxy':'TS 光学代理','evidence boundary':'证据边界',
    'ai brain':'AI 脑','vehicle brain':'车载脑','dual-arm station':'双臂工位','characterization':'表征',
    'browser + ai brain':'浏览器 + AI 脑','ai brain x5':'AI 脑 X5','command center':'指挥中心',
    'vehicle + dual arms':'车载脑 + 双臂','researcher':'研究员',
    'trace_id + verdict':'trace_id + 判读','wo_log + operator':'工单日志 + 操作员','lambda_obs + close_note':'实测 lambda + 收单备注',
    'formula design agent':'配方设计 Agent','synthesis prediction agent':'合成预测 Agent',
    'batch lifecycle agent':'批次生命周期 Agent','embodied execution agent':'具身执行 Agent',
    'characterization feedback agent':'表征回流 Agent',
    'formula / dopant / target wavelength':'配方 / 掺杂 / 目标波长','candidate material object':'候选材料对象',
    'formula object':'配方对象','trace_id, verdict, ci, sintering hints':'trace_id、判读、CI、烧结建议',
    'prediction trace + work order':'预测 trace + 工单','stage transitions and audit log':'阶段流转与审计日志',
    'approved work order':'已批准工单','sample movement and station completion evidence':'样品移动与工位完成证据',
    'xrd/pl result and lambda_obs':'XRD/PL 结果与 lambda_obs','closed work order and next-round learning signal':'已关闭工单与下一轮学习信号',
    'read-only public query; no private prompt exposure':'公开只读查询；不暴露私有 prompt',
    'hard priors can override llm text':'硬先验可覆盖 LLM 文本',
    'state changes are role-gated and logged':'状态变更受角色保护并写入日志',
    'public site never publishes unsafe chassis velocity or arm commands':'公网不发布不安全底盘速度或机械臂命令',
    'measured/history rows are labelled':'实测/历史行均保留标签',
    'nir phosphor screen':'NIR 荧光粉筛选','failure-loop run':'失败回流轮次','defense demo path':'答辩演示路径',
    'select host/dopant -> predict go/revise -> create batch -> execute sample -> backfill pl/xrd':'选择 host/掺杂 -> 预测 GO/REVISE -> 创建批次 -> 执行样品 -> 回填 PL/XRD',
    'open revise rows -> inspect flags -> alter site/concentration -> requeue -> compare evidence_score':'打开 REVISE 行 -> 检查 flag -> 调整位点/浓度 -> 重新入队 -> 对比证据评分',
    'open brain explain -> filter atlas -> open material detail -> open work order -> show audit export':'打开 AI 脑解释 -> 筛选图鉴 -> 打开材料详情 -> 打开工单 -> 展示审计导出',
    'public studio is read-oriented; write apis remain role-gated.':'公开 Studio 以只读为主；写 API 仍受角色保护。',
    'every stage must link to trace_id, work_order, wo_log, material object, or labelled replay/curated evidence.':'每个阶段必须链接到 trace_id、工单、工单日志、材料对象或已标注的回放/策展证据。',
    'robot execution is represented as lifecycle state and evidence, not as unsafe public controls.':'机器人执行以生命周期状态和证据呈现，不提供不安全公网控制。',
    'design / predict':'设计 / 预测','embodied execution':'具身执行','furnace / xrd / pl':'炉 / XRD / PL','feedback loop':'反馈闭环',
    'llm/bpu prediction, graphrag, audit trace':'LLM/BPU 预测、GraphRAG、审计 trace',
    'lab-fsd replay, nav2/mppi authority, sample route':'Lab-FSD 回放、Nav2/MPPI 权威控制、样品路线',
    'dual mycobot station, interlock, sample handling':'双 myCobot 工位、互锁、样品处理',
    'secondary mycobot station':'副 myCobot 工位','available for replay; physical execution requires operator readiness':'可用于回放；物理执行需要操作员确认就绪',
    'tim8 stepper lift stage replay / documented readiness':'TIM8 步进升降台回放 / 文档化就绪',
    'pe0 hold/release state replay':'PE0 吸附/释放状态回放','relay extend/retract replay':'继电器伸出/缩回回放',
    'hot-zone protocol and risk lock':'热区 protocol 与风险锁','phase check evidence object':'物相检查证据对象',
    'lambda_obs feedback evidence':'lambda_obs 回流证据','sso, caddy, service worker, release rollback':'SSO、Caddy、Service Worker、发布回滚',
    'public display only':'公开只读展示','no live execution request queued':'暂无真机执行请求入队',
    'operator approval before physical run':'物理运行前需要操作员批准','manual approval':'人工批准',
    'operator/member':'操作员/队员','physical execution is never started from public page':'物理执行绝不会从公开页面启动',
    'risk lock':'风险锁','safety policy':'安全策略','hot-zone, arm sweep, and public command boundaries':'热区、机械臂扫掠区和公网命令边界',
    'rollback/cancel':'回滚/取消','role-gated api':'角色保护 API','work order can be cancelled or replayed, not remotely driven':'工单可取消或回放，但不可远程驱动',
    'cloud command is a public scheduling and evidence view; it does not expose dangerous remote control.':'云端指挥是公开排程与证据视图，不暴露危险远程控制。',
    'equipment reservation and fault locks are simulated/read-only until an authenticated operator acts inside protected systems.':'设备预约与故障锁在公开端为模拟/只读，需认证操作员在受保护系统内执行。',
    'every queue card links toward tasks, replay, twin, assets, or observability instead of hiding blockers.':'每张队列卡都会链接到任务、回放、孪生、资产或可观测性，不隐藏阻塞项。',
    '3 minute script':'3 分钟脚本','5 minute script':'5 分钟脚本','8 minute script':'8 分钟脚本',
    'state the problem, the closed loop, and the proof surface':'说明问题、闭环和证据界面',
    'walk a judge through the strongest evidence path':'带评委走完最强证据路径',
    'complete a full defense narrative with offline fallback':'完成含离线兜底的完整答辩叙事',
    'problem':'问题','system':'系统','differentiator':'差异点','boundary':'边界','open the os':'打开系统',
    'ai proof':'AI 证据','embodied proof':'具身证据','cloud lab proof':'云实验室证据','audit proof':'审计证据',
    'why it matters':'为什么重要','scientific stack':'科学栈','ai/edge stack':'AI/边缘栈','embodied autonomy':'具身自主',
    'lifecycle execution':'生命周期执行','operations':'运维','fallback':'兜底',
    'nir phosphor discovery is slow because design, synthesis, xrd, pl and feedback are separated.':'NIR 荧光粉发现慢，是因为设计、合成、XRD、PL 与反馈彼此割裂。',
    'the public site shows one software-defined lab loop: ai prediction, embodied execution replay, characterization and feedback.':'公网展示一条软件定义实验室闭环：AI 预测、具身执行回放、表征与反馈。',
    'ai brain x5, materials atlas, lab-fsd v2, replay and cloud command are linked by trace/work_order evidence.':'AI 脑 X5、材料图鉴、Lab-FSD v2、回放和云端指挥都由 trace/工单证据串联。',
    'offline or replay sources stay labelled; public pages never expose robot or actuator controls.':'离线或回放来源保持标注；公开页面绝不暴露机器人或执行器控制。',
    'start at the homepage, then open defense mode as the evidence map.':'从首页开始，再打开答辩防御作为证据地图。',
    'show brain explain and atlas rows with ci, source labels and export endpoints.':'展示 AI 脑解释与带 CI、来源标签、导出端点的图鉴行。',
    'show lab-fsd read-only world model and experiment replay with fault injection.':'展示 Lab-FSD 只读世界模型和带故障注入的实验回放。',
    'show command center resource board, blockers, approvals and public guardrails.':'展示指挥中心资源看板、阻塞项、审批和公网安全边界。',
    'open traces, hardening and export links to demonstrate reproducibility and public safety.':'打开链路、加固检查和导出链接，证明可复现与公网安全。',
    'explain 15 months to 15 minutes as a closed-loop materials automation objective, not a marketing slogan.':'把 15 个月到 15 分钟解释为材料自动化闭环目标，而不是营销口号。',
    'ts inverse, conformal ci, mlip/cache labels and graphrag are presented as inspectable evidence, not hidden claims.':'TS 反向、Conformal CI、MLIP/cache 标签和 GraphRAG 都以可检查证据呈现，而不是隐藏口号。',
    '9 local llm paths, bpu slot evidence and fly-mb reasoning are visible through public summaries.':'9 路本地 LLM、BPU 槽证据和 Fly-MB 推理通过公开摘要可见。',
    'lab-fsd v2 is a read-only world-model console; nav2/mppi remains the physical control authority.':'Lab-FSD v2 是只读世界模型控制台；Nav2/MPPI 仍是物理控制权威。',
    'replay and studio connect formula, trace_id, work_order, stages, faults, actuators and recovery notes.':'回放和 Studio 串联配方、trace_id、工单、阶段、故障、执行器和恢复说明。',
    'command, observability, traces and hardening prove the system can be operated, debugged and defended.':'指挥、可观测性、链路和加固检查证明系统可运维、可调试、可答辩。',
    'if hardware is offline, labelled replay/history still completes the explanation without pretending to be live.':'硬件离线时，带标签的回放/历史仍能完成解释，不伪装成 live。',
    'ai brain x5 prediction stack':'AI 脑 X5 预测栈','9 local llm / 5 bpu slot evidence':'9 路本地 LLM / 5 BPU 槽证据',
    'fly-mb memory brain explainability':'Fly-MB 记忆脑可解释性','ts inverse / conformal ci scientific proof':'TS 反向 / Conformal CI 科学证据',
    'materials atlas searchable object model':'材料图鉴可检索对象模型','lab-fsd v2 world model and safety gate':'Lab-FSD v2 世界模型与安全门禁',
    'digital twin / experiment replay lifecycle':'数字孪生 / 实验回放生命周期','dual-arm station and sample flow evidence':'双臂工位与样品流证据',
    'cloud lab command center scheduling':'云实验室指挥中心排程','liquid glass visual system':'液态玻璃视觉系统',
    'observability, traces, exports and hardening':'可观测性、链路、导出与加固',
    'api + page':'API + 页面','api + source file':'API + 源文件','page + reasoning trace':'页面 + 推理 trace',
    'method + export':'方法 + 导出','page + export':'页面 + 导出','page + api':'页面 + API','screenshot + css':'截图 + CSS',
    'pages + apis':'页面 + API','public summaries only; no raw prompts or secrets':'仅公开摘要；不暴露原始 prompt 或密钥',
    'latency and slot labels are evidence, not public model control':'延迟和槽位标签是证据，不是公网模型控制',
    'compressed explanation, not private chain-of-thought':'压缩解释，不是私有思维链',
    'uncertainty shown; unsupported constants stay labelled':'展示不确定性；不支持的常数保持标注',
    'rows keep live/mirror/history/source labels':'行保留 live/mirror/history/source 标签',
    'read-only; no navigation velocity or physical control':'只读；无导航速度或物理控制',
    'fault injection affects narrative only':'故障注入只影响叙事',
    'no public arm, lift, magnet, linear actuator or yaw command':'无公开机械臂、升降台、电磁铁、电推杆或舵机命令',
    'public display only; operator approval remains protected':'仅公开展示；操作员审批仍受保护',
    'visual layer supports evidence; it is not used as proof by itself':'视觉层支撑证据展示，但不单独作为证明',
    'public-safe logs only; secrets and private prompts excluded':'仅公开安全日志；排除密钥与私有 prompt',
    'live / mirror demo path':'真机 / 镜像演示路径','offline hardware demo path':'离线硬件演示路径',
    'open defense':'打开答辩防御','materials proof':'材料证据','command proof':'指挥证据','trace proof':'链路证据',
    'state boundary':'状态边界','replay sample flow':'回放样品流','fsd read-only':'FSD 只读','evidence exports':'证据导出','hardening':'加固检查',
    'can explain the core loop in 3 minutes':'可在 3 分钟内讲清核心闭环',
    'every major claim has a page or api evidence link':'每个主要主张都有页面或 API 证据链接',
    'offline hardware still has a labelled defense path':'离线硬件仍有带标签的答辩路径',
    'replay/mock/mirror are not described as live':'回放/mock/镜像不会被描述成 live',
    'no public dangerous controls':'无公网危险控制','exportable objects exist for materials and traces':'材料和链路对象可导出',
    'problem':'问题','closed loop':'闭环','scientific proof':'科学证据','evidence links':'证据链接','safety boundary':'安全边界',
    'defense mode is an evidence index and script surface; it does not add public write operations.':'答辩防御是证据索引和脚本界面，不增加公网写操作。',
    'claims without evidence links must be treated as plan, replay, documented boundary, or offline history.':'没有证据链接的主张必须标为计划、回放、文档边界或离线历史。',
    'robot navigation and actuator control remain outside public pages.':'机器人导航和执行器控制保持在公开页面之外。',
    'visual / interaction system':'视觉 / 交互系统','scientific database and evidence object':'科学数据库与证据对象',
    'materials api interoperability':'材料 API 互操作性','cloud laboratory operating system':'云实验室操作系统',
    'embodied autonomy and safety case':'具身自主与安全案例','observability and operations':'可观测性与运维',
    'mobile, accessibility and performance':'移动端、可访问性与性能',
    'layered material, clear hierarchy, restrained motion, content-first controls.':'分层材质、清晰层级、克制动效、内容优先控件。',
    'searchable scientific object, method transparency, downloadable evidence and uncertainty.':'可检索科学对象、方法透明、可下载证据和不确定性。',
    'structured api, documented schema, filters, exports and reproducible examples.':'结构化 API、文档化 schema、筛选、导出和可复现实例。',
    'single workspace for experiment design, execution, analysis, resource scheduling and auditability.':'一个工作台覆盖实验设计、执行、分析、资源排程和审计。',
    'safety framework, measurable evidence, simulation/replay and explicit operational boundary.':'安全框架、可量化证据、仿真/回放和明确运行边界。',
    'metrics, logs, traces, status, incidents, performance signals and correlation paths.':'指标、日志、链路、状态、事故、性能信号和关联路径。',
    'mobile-first layout, no layout shift, reduced motion, keyboard focus and inspectable performance.':'移动优先布局、无布局漂移、减弱动效、键盘焦点和可检查性能。',
    'not a native os material api; it is a web implementation scoped to this project.':'不是原生 OS 材质 API；这是本项目范围内的 Web 实现。',
    'dataset size is project-scale, not a global public database.':'数据集规模是项目级，不是全球公共数据库。',
    'api is intentionally public-safe and read-focused; protected write apis remain behind sso.':'API 刻意保持公网安全和只读为主；受保护写 API 仍在 SSO 后。',
    'public page is a command-center evidence surface, not a public actuator console.':'公开页面是指挥中心证据面，不是公开执行器控制台。',
    'it borrows the safety-case pattern for a lab robot; it does not claim road-scale av capability.':'它借鉴实验室机器人安全案例范式，不声称道路级自动驾驶能力。',
    'telemetry is lightweight and project-local, not a full saas observability backend.':'遥测是轻量项目本地实现，不是完整 SaaS 可观测后端。',
    'no real-user core web vitals backend yet; current evidence is synthetic browser verification.':'暂未接入真实用户 Core Web Vitals 后端；当前证据来自合成浏览器验证。',
    'average benchmark score >= 4.5':'平均对标分 >= 4.5','no dimension below 4.0':'无维度低于 4.0',
    'no public unsafe control token':'无公网危险控制 token','live / mirror / replay / mock / offline labels preserved':'保留 live / mirror / replay / mock / offline 标签',
    'api docs and curl examples available':'API 文档和 curl 示例可用','read-only evidence surface':'只读证据界面',
    'status taxonomy visible':'状态分类可见','/api/docs + /api/openapi.json':'/api/docs + /api/openapi.json',
    'materials explorer supports filtering, sorting and public exports':'材料图鉴支持筛选、排序和公开导出',
    'emerald cloud lab command center':'Emerald Cloud Lab 指挥中心',
    'agent studio connects formula, prediction, queue, execution, xrd/pl and feedback':'科学 Agent 工作室串联配方、预测、队列、执行、XRD/PL 与回流',
    'cloud command shows lanes, resources, calendar, blockers, approvals and guardrails':'云端指挥展示任务泳道、资源、日历、阻塞项、审批和安全边界',
    'defense mode states physical control stays outside the public site':'答辩防御明确物理控制不在公网展示站内',
    'add real-user web vitals collection after public domain traffic is stable.':'公开域名流量稳定后接入真实用户 Web Vitals 采集。',
    'expand materials atlas with more source-labelled rows and doi-backed methods.':'扩充材料图鉴，加入更多带来源标签和 DOI 方法的行。',
    'add a public read-only evidence bundle download for offline judging.':'增加公开只读证据包下载，便于离线评审。',
    'the site reaches top-tier benchmark parity for a competition/research showcase within a public-safe scope; it does not claim the commercial scale of apple, alphafold db, materials project, emerald cloud lab, waymo, datadog or grafana.':'本站在公开安全范围内达到竞赛/科研展示站的顶级横向对标标准；不声称具备 Apple、AlphaFold DB、Materials Project、Emerald Cloud Lab、Waymo、Datadog 或 Grafana 的商业规模。'
  };
  return m[k] || m[s] || s;
}
function uiEsc(v){ return esc(uiTerm(v)); }
function psStatusLabel(status){
  const s=String(status||'unknown').toLowerCase();
  if(uiLang()==='en') return status||'Unknown';
  return {operational:'正常',degraded:'降级',mirror:'镜像',replay:'回放',offline:'离线',unknown:'未知'}[s] || status || '未知';
}
function psTime(ts){
  if(!ts) return '—';
  try{ return new Date(ts*1000).toLocaleString('zh-CN',{hour12:false}); }catch(e){ return '—'; }
}
function psBar(arr){
  arr=arr||[];
  return '<div class="ps-bars">'+arr.map(b=>'<span class="'+psClass(b.status)+'" title="'+uiEsc(b.status)+' · '+psTime(b.from)+'"></span>').join('')+'</div>';
}
function psMiniBars(d,key){
  const bars=((((d||{}).availability||{})[key]||{})['24h']||[]).slice(-12);
  if(!bars.length) return '<div class="scene-mini-bars empty"><span></span><span></span><span></span><span></span></div>';
  return '<div class="scene-mini-bars">'+bars.map(b=>'<span class="'+psClass(b.status)+'" title="'+uiEsc((b.status||'unknown')+' · '+psTime(b.from))+'"></span>').join('')+'</div>';
}
function sceneTelemetryEnsure(){
  const host=document.querySelector('#overview .hero-scene'); if(!host) return null;
  let box=document.getElementById('sceneTelemetry');
  if(!box){
    box=document.createElement('div');
    box.id='sceneTelemetry';
    box.className='scene-telemetry unknown';
    host.appendChild(box);
  }
  return box;
}
function sceneTelemetryComponent(d,key,alts){
  const comps=(d&&d.components)||[], keys=[key].concat(alts||[]);
  return comps.find(c=>keys.includes(c.key)) || comps.find(c=>keys.some(k=>String(c.name||'').toLowerCase().includes(k)));
}
function sceneTelemetrySourceLabel(c){
  const src=String((c&&c.source)||'unknown').toLowerCase();
  return {live:'真机',stale:'陈旧',mirror:'镜像',replay:'回放',offline:'离线',unknown:'未知'}[src] || src;
}
function sceneNodeOpen(focus,page){
  try{ if(window.focus3D) focus3D(focus); }catch(e){}
  if(page) setTimeout(()=>go(page), 150);
}
function homeNodeList(d){
  const pick=(key,alts)=>sceneTelemetryComponent(d,key,alts||[]);
  const pred=pick('prediction',['predict','api'])||{};
  const atlasStatus=(pred.status||'Operational');
  return [
    {key:'lab', label:uiText('AI 脑 X5','AI brain X5'), focus:'lab', page:'brain', sub:uiText('9 LLM / 5 BPU slots','9 LLM / 5 BPU slots'), comp:pick('lab',['ai brain'])},
    {key:'car', label:uiText('车载脑','Vehicle brain'), focus:'car', page:'fsd', sub:'LD14 / Astra / Nav2', comp:pick('car',['car brain'])},
    {key:'arm', label:uiText('双臂工位','Dual-arm station'), focus:'arm', page:'replay', sub:'arm01 / arm02 / F407', comp:pick('arm',['arm workstation','arm01'])},
    {key:'prediction', label:uiText('AI 预测 API','Prediction API'), focus:'lab', page:'brain', sub:'TS / MLIP / CI', comp:pred},
    {key:'atlas', label:uiText('材料证据库','Materials atlas'), focus:'xrd', page:'atlas', sub:uiText('Atlas / export / citation','Atlas / export / citation'), comp:{status:atlasStatus, source:'mirror', detail:'curated public dataset'}},
    {key:'sample', label:uiText('样品路径','Sample path'), focus:'sample', page:'replay', sub:'trace / work_order', comp:{status:'Replay', source:'replay', detail:'read-only sample-flow replay'}},
    {key:'vps', label:uiText('公网 VPS / SSO','Public VPS / SSO'), focus:'vps', page:'status', sub:'Cloudflare / Caddy / SW', comp:pick('vps',['command center','auth'])||pick('static',[])},
    {key:'static', label:uiText('发布缓存层','Release cache'), focus:'vps', page:'release', sub:'Service Worker / rollback', comp:pick('static',['service worker'])}
  ];
}
function componentDetail(c){
  if(!c) return uiText('source unknown','source unknown');
  const source=sceneTelemetrySourceLabel(c);
  const latency=(c.latency_ms!=null)?(Math.round(c.latency_ms)+'ms'):'';
  return [source,latency,c.source_detail||''].filter(Boolean).join(' · ') || source;
}
function homeOsUpdate(d){
  const s=(d&&d.summary)||{}, cls=psClass(s.status), rel=s.release||'site27-highlight-claims-20260709';
  const dot=document.getElementById('homeOsDot'); if(dot) dot.className='os-live-dot '+cls;
  const relEl=document.getElementById('homeOsRelease'); if(relEl) relEl.textContent=rel+' · '+uiText('公网只读','read-only public surface');
  const nodes=homeNodeList(d);
  const matrix=document.getElementById('homeStatusMatrix');
  if(matrix) matrix.innerHTML=nodes.slice(0,6).map(n=>{
    const c=n.comp||{}, st=psClass(c.status), status=psStatusLabel(c.status), detail=componentDetail(c);
    return '<button type="button" class="home-status-row '+st+'" onclick="sceneNodeOpen(&quot;'+esc(n.focus)+'&quot;,&quot;'+esc(n.page)+'&quot;)">'+
      '<span></span><b>'+uiEsc(n.label)+'</b><i>'+status+'</i><em>'+uiEsc(detail)+'</em></button>';
  }).join('');
  const swName='cmdcenter-shell-v73-site27-highlight-claims';
  const rb=document.getElementById('homeReleaseBoard');
  if(rb) rb.innerHTML=[
    ['Release',rel],
    ['SW Cache',swName],
    [uiText('Method Boundary','Method Boundary'),uiText('公开面 GET/HEAD/OPTIONS · 写操作需 SSO/RBAC','Public GET/HEAD/OPTIONS · writes require SSO/RBAC')],
    [uiText('Boundary','Boundary'),uiText('公网只读 · 不暴露机器人执行控制','Read-only public surface · no public robot actuation')],
    [uiText('Fallback','Fallback'),uiText('真机离线时显示镜像 / 回放 / 离线标签','Mirror / replay / offline labels when devices are unavailable')]
  ].map(x=>'<div><span>'+uiEsc(x[0])+'</span><b>'+uiEsc(x[1])+'</b></div>').join('');
  const ev=document.getElementById('homeEventStream');
  const events=(d&&d.events)||[];
  if(ev) ev.innerHTML=(events.length?events.slice(0,4).map(e=>{
    const sev=e.severity==='crit'?'crit':e.severity==='warn'?'warn':'info';
    const kind=e.kind||e.severity||'event';
    return '<div class="home-event-row '+sev+'"><span>'+psTime(e.ts)+'</span><em>'+uiEsc(kind)+'</em><b>'+uiEsc(e.sys||'platform')+'</b><p>'+uiEsc(e.message||'')+'</p><i>'+uiText('source: public_status · boundary: read-only','source: public_status · boundary: read-only')+'</i></div>';
  }).join(''):'<div class="home-event-row info"><span>local</span><em>evidence</em><b>'+uiText('暂无近期事故','No recent incidents')+'</b><p>'+uiText('公网只读孪生继续显示镜像、回放和状态来源。','The read-only public twin keeps mirror, replay and source labels visible.')+'</p><i>source: local fallback · boundary: no public control</i></div>');
}
function sceneTelemetryUpdate(d){
  const box=sceneTelemetryEnsure(); if(!box) return;
  const s=(d&&d.summary)||{}, cls=psClass(s.status);
  box.className='scene-telemetry '+cls;
  const item=(n)=>{
    const c=n.comp||{}, st=psClass(c.status), source=sceneTelemetrySourceLabel(c);
    const status=psStatusLabel(c&&c.status);
    const detail=(c&&c.latency_ms!=null)?(Math.round(c.latency_ms)+'ms'):(c&&c.source_detail)||n.sub||source;
    return '<button type="button" class="scene-tel-node '+st+'" onclick="sceneNodeOpen(&quot;'+esc(n.focus)+'&quot;,&quot;'+esc(n.page)+'&quot;)">'+
      '<i></i><span>'+uiEsc(n.label)+'</span><b>'+status+'</b><em>'+uiEsc(source+' · '+detail)+'</em>'+psMiniBars(d,n.key)+'</button>';
  };
  const nodes=homeNodeList(d);
  box.innerHTML=
    '<div class="scene-tel-head"><span>'+uiText('实验室数字孪生','Lab digital twin')+'</span><b>'+psStatusLabel(s.status)+'</b><em>'+uiEsc(s.release||'')+'</em></div>'+
    '<div class="scene-tel-grid">'+
      nodes.map(item).join('')+
    '</div>';
}
function psUpdateHome(d){
  const s=(d&&d.summary)||{}, comps=(d&&d.components)||[];
  const strip=document.getElementById('statusStripHome'); if(!strip) return;
  const cls=psClass(s.status);
  strip.className='status-strip '+cls;
  const dot=strip.querySelector('.ss-dot'); if(dot) dot.className='ss-dot '+cls;
  const title=document.getElementById('ssTitle'); if(title) title.textContent=uiText('状态: ','Status: ')+psStatusLabel(s.status);
  const live=comps.filter(c=>c.source==='live').length, mirror=comps.filter(c=>c.source==='mirror').length, off=comps.filter(c=>c.source==='offline').length;
  const body=document.getElementById('ssBody'); if(body) body.textContent=uiText('在线 ','live ')+live+' · '+uiText('镜像 ','mirror ')+mirror+' · '+uiText('离线 ','offline ')+off+' · '+(s.release||'');
  homeOsUpdate(d);
  sceneTelemetryUpdate(d);
}
async function loadResearchPassport(){
  const grid=document.getElementById('researchPassportGrid');
  const cite=document.getElementById('researchPassportCite');
  if(!grid && !cite) return;
  let d=null;
  try{ d=await fetch('/api/research_passport',{cache:'no-store'}).then(r=>r.json()); }catch(e){ d=null; }
  if(!d) return;
  if(grid){
    grid.innerHTML=(d.passport_cards||[]).map(card=>'<article class="rp-card-'+uiEsc(card.key||'')+'">'+
      '<span>'+uiEsc(card.label||'')+'</span>'+
      '<b>'+uiEsc(card.title||'')+'</b>'+
      '<p>'+uiEsc(card.detail||'')+'</p>'+
      '<i>'+uiEsc(card.value||'')+'</i>'+
      '<div class="rp-links">'+((card.evidence||[]).slice(0,3).map(h=>'<a href="'+uiEsc(h)+'" target="_blank" rel="noopener">'+uiEsc(h)+'</a>').join(''))+'</div>'+
    '</article>').join('');
  }
  if(cite){
    const c=d.citation||{}, counts=d.counts||{};
    cite.innerHTML='<b>Citation</b><span>'+uiEsc(c.how_to_cite||'')+'</span>'+
      '<em>'+uiEsc((counts.material_rows||0)+' materials · '+(counts.public_get_surfaces||0)+' public GET surfaces · '+(counts.components||0)+' status components')+'</em>'+
      '<div><a href="/api/evidence_bundle.txt" target="_blank" rel="noopener">TXT</a><a href="/api/evidence_bundle.json" target="_blank" rel="noopener">JSON</a><a href="/api/public_manifest" target="_blank" rel="noopener">Manifest</a></div>';
  }
}
function psRender(d){
  if(!d) return;
  psUpdateHome(d);
  const s=d.summary||{}, comps=d.components||[], av=d.availability||{}, evs=d.events||[];
  const ts=document.getElementById('psTs'); if(ts) ts.textContent=psStatusLabel(s.status)+' · '+(s.release||'')+' · '+new Date((d.ts||Date.now()/1000)*1000).toLocaleTimeString('zh-CN');
  const sum=document.getElementById('psSummary');
  if(sum) sum.innerHTML='<div class="ps-hero '+psClass(s.status)+'"><div><span>'+uiText('总体','Overall')+'</span><b>'+psStatusLabel(s.status)+'</b><p>'+uiEsc(s.note||'')+'</p></div>'+
    '<div class="ps-legend"><i class="ok"></i>'+uiText('真机','live')+' <i class="warn"></i>'+uiText('镜像/回放/陈旧','mirror/replay/stale')+' <i class="crit"></i>'+uiText('离线','offline')+' <i class="unknown"></i>'+uiText('未知','unknown')+'</div></div>';
  const grid=document.getElementById('psComponents');
  if(grid) grid.innerHTML=comps.map(c=>'<div class="ps-card '+psClass(c.status)+'">'+
    '<div class="ps-card-h"><b>'+uiEsc(c.name)+'</b><span>'+psStatusLabel(c.status)+'</span></div>'+
    '<div class="ps-source">'+uiEsc(c.source||'unknown')+(c.latency_ms!=null?' · '+esc(c.latency_ms)+'ms':'')+'</div>'+
    '<p>'+uiEsc(c.detail||'')+'</p></div>').join('') || '<div class="oe-empty">'+uiText('暂无组件','No components')+'</div>';
  const ab=document.getElementById('psAvailability');
  if(ab) ab.innerHTML=['lab','car','arm'].map(k=>{
    const label={lab:uiText('AI 脑','AI brain'),car:uiText('车载脑','Car brain'),arm:uiText('双臂工位','Arm workstation')}[k]||k, row=av[k]||{};
    return '<div class="ps-av"><div class="ps-av-h"><b>'+label+'</b><span>24h</span></div>'+psBar(row['24h'])+
      '<div class="ps-av-h small"><span>7d</span></div>'+psBar(row['7d'])+'</div>';
  }).join('');
  const eb=document.getElementById('psEvents');
  if(eb) eb.innerHTML=evs.length?evs.map(e=>'<div class="ps-event '+(e.severity==='crit'?'crit':e.severity==='warn'?'warn':'info')+'">'+
    '<span>'+psTime(e.ts)+'</span><b>'+uiEsc(e.sys||'platform')+'</b><i>'+uiEsc(e.kind||e.severity||'event')+'</i><p>'+uiEsc(e.message||'')+'</p></div>').join('') :
    '<div class="oe-empty">'+uiText('暂无近期事故。设备离线时，网站会在可用处保持镜像/回放。','No recent incidents. If devices are offline, the site remains on mirror/replay where available.')+'</div>';
}
async function loadPublicStatus(){
  let d=null;
  try{ d=await fetch('/api/public_status',{cache:'no-store'}).then(r=>r.json()); }
  catch(e){
    d={summary:{status:'Unknown',release:'offline',note:'Status API unreachable; showing local shell only.'},components:[],availability:{},events:[]};
  }
  psRender(d);
}
setInterval(()=>{ if(cur==='home'||cur==='status') loadPublicStatus(); }, 15000);
loadPublicStatus();
loadResearchPassport();

/* ---- 平台 KPI (真实值, 取自 lab serving 端; 轮询 + SSE 双路喂) ---- */
const KPI_FALLBACK={ source:'mirror', kpi:{
  ci_coverage_pct:91.7, ci_narrowing_pct:26.4, audit_valid:649, audit_total:649,
  audit_intact:true, predictions:1286
}};
function applyKpi(d){
  d=d||{};
  const k=Object.assign({}, KPI_FALLBACK.kpi, d.kpi||{});
  const source=d.source||KPI_FALLBACK.source;
  const set=(id,v)=>{ const e=document.getElementById(id); if(e) e.innerHTML=v; };
  const tw=(id,v,dec,suf)=>{ const e=document.getElementById(id);
    if(e&&v!=null) tweenNum(e, +v, x=>(dec?x.toFixed(dec):Math.round(x))+(suf||'')); };
  tw('kpi-cov', k.ci_coverage_pct, (k.ci_coverage_pct%1?1:0), '<span class="ku">%</span>');
  tw('kpi-narrow', k.ci_narrowing_pct, 1, '<span class="ku">%</span>');
  if(k.audit_valid!=null) set('kpi-audit', k.audit_valid+'/'+k.audit_total+(k.audit_intact?' <span class="ku" style="color:#059669">✓</span>':''));
  tw('kpi-pred', k.predictions, 0, '');
  const src=document.getElementById('kpiSrc');
  if(src){ const real=source==='real'; src.className='kpi-src '+(real?'real':'mirror');
    src.textContent = real?'● 实时 · 真机直连':'◐ 实时 · 镜像演示';
    src.title = real?'数据来自在线设备':'设备离线或接口不可达时显示 VPS 镜像演示口径'; }
}
async function pollKpi(){
  try{ applyKpi(await fetch('/api/kpi',{cache:'no-store'}).then(r=>r.json())); }
  catch(e){ applyKpi(KPI_FALLBACK); }
}
setInterval(pollKpi, 12000); pollKpi();

/* ---- P1: historian 事件流 + SSE 实时推送 ---- */
const EV_SYS={lab:'AI 脑',car:'车载脑',arm:'机械臂'};
const _evSeen=new Set();
function evRow(e){
  const d=document.createElement('div'); d.className='ev '+(e.severity==='crit'?'crit':e.severity==='warn'?'warn':'info');
  const t=new Date(e.ts*1000);
  const stamp=t.toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'})+' '+t.toLocaleTimeString('zh-CN',{hour12:false});
  d.innerHTML='<span class="sev"></span><span class="t"></span><span class="sy '+(e.sys||'')+'"></span><span class="msg"></span>';
  d.querySelector('.t').textContent=stamp;
  d.querySelector('.sy').textContent=EV_SYS[e.sys]||e.sys||'平台';
  d.querySelector('.msg').textContent=e.message||'';
  return d;
}
function pushEvents(evs, live){
  const feed=document.getElementById('evFeed'); if(!feed) return;
  evs.forEach(e=>{
    if(_evSeen.has(e.id)) return; _evSeen.add(e.id);
    feed.insertBefore(evRow(e), feed.firstChild);   // 升序逐条 prepend → 最新在顶
    if(live && (e.severity==='crit'||e.severity==='warn'))
      toast((e.severity==='crit'?'🔴 ':'🟡 ')+(e.message||''));
  });
  const empty=feed.querySelector('.oe-empty');
  if(empty && feed.children.length>1) empty.remove();
  while(feed.children.length>120) feed.removeChild(feed.lastChild);
}
async function loadEvents(){
  try{
    const d=await fetch('/api/events?hours=72&limit=60',{cache:'no-store'}).then(r=>r.json());
    pushEvents((d.events||[]).slice().reverse(), false);  // API 倒序 → 反转成升序再 prepend
  }catch(e){}
}
let _es=null;
function startStream(){
  if(!window.EventSource) return;
  try{
    _es=new EventSource('/api/stream');
    _es.onmessage=(ev)=>{ try{
      const d=JSON.parse(ev.data);
      if(d.events){ pushEvents(d.events, true); if(cur==='ops') loadAlarms(); }
      if(d.kpi) applyKpi(d.kpi);
      if(d.alarms){ applyBell(d.alarms); if(NOTIFY.on && (d.alarms.crit||d.alarms.warn)) notifyCheck(); }
    }catch(e){} };
    _es.onerror=()=>{ try{_es.close();}catch(e){} _es=null; setTimeout(startStream, 10000); };
  }catch(e){}
}
loadEvents(); startStream();

/* ---- P2: 告警中心 (规则引擎升警 / 人工 ack / 自动销警) ---- */
const SEV_TXT={crit:'CRIT',warn:'WARN',info:'INFO'};
function fmtDur(s){ if(s<90) return Math.round(s)+'s'; if(s<5400) return Math.round(s/60)+' 分';
  if(s<172800) return (s/3600).toFixed(1)+' 时'; return Math.round(s/86400)+' 天'; }
function alRow(a, hist){
  const d=document.createElement('div'); d.className='al '+(a.severity||'info')+(hist?' hist':'');
  const t=new Date(a.ts_raised*1000);
  const stamp=t.toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'})+' '+t.toLocaleTimeString('zh-CN',{hour12:false});
  const dur=hist ? fmtDur((a.ts_cleared||a.ts_raised)-a.ts_raised) : fmtDur(Date.now()/1000-a.ts_raised);
  d.innerHTML='<span class="sevtag"></span><span class="amsg"></span><span class="ainfo"></span>';
  d.querySelector('.sevtag').textContent=SEV_TXT[a.severity]||a.severity;
  d.querySelector('.amsg').textContent=a.message||'';
  d.querySelector('.ainfo').innerHTML=stamp+'<br>'+(hist?('持续 '+dur+' · 已恢复'):('已持续 '+dur));
  if(!hist){
    if(a.ts_ack){ const s=document.createElement('span'); s.className='acked'; s.textContent='✓ '+(a.ack_by||'')+' 已确认'; d.appendChild(s); }
    else{ const b=document.createElement('button'); b.className='ackbtn'; b.textContent='确认'; b.onclick=()=>ackAlarm(a.id); d.appendChild(b); }
  } else if(a.ts_ack){ const s=document.createElement('span'); s.className='acked'; s.textContent='✓ '+(a.ack_by||''); d.appendChild(s); }
  return d;
}
function applyBell(c){
  setFaviconBadge(c);
  if(NC_OPEN) ncLoad();
  const b=document.getElementById('bellCnt'); if(!b) return;
  const n=c.total||0;
  if(!n){ b.style.display='none'; return; }
  b.style.display='flex'; b.textContent=n>99?'99+':n;
  b.className='bcnt'+(c.crit?' crit':c.warn?' warn':'');
}
async function loadAlarms(){
  try{
    const d=await fetch('/api/alarms',{cache:'no-store'}).then(r=>r.json());
    const list=document.getElementById('alarmList'), hist=document.getElementById('alarmHist');
    if(list){ list.innerHTML='';
      if(!(d.active||[]).length) list.innerHTML='<div class="oe-empty">无活动告警 — 全链路健康 ✓</div>';
      (d.active||[]).forEach(a=>list.appendChild(alRow(a,false))); }
    if(hist){ hist.innerHTML='';
      if(!(d.history||[]).length) hist.innerHTML='<div class="oe-empty">暂无历史</div>';
      (d.history||[]).forEach(a=>hist.appendChild(alRow(a,true))); }
    const mc=document.getElementById('mailCh');
    if(mc){ mc.className='mail-ch'+(d.mail_channel?' on':'');
      mc.textContent=d.mail_channel?'✉ 邮件通道: 已启用 (crit 自动发信)':'✉ 邮件通道: 未配置'; }
    applyBell(d.counts||{});
  }catch(e){}
}
async function ackAlarm(id){
  try{
    const r=await fetch('/api/alarms/'+id+'/ack',{method:'POST'});
    if(r.status===403){ toast('评委账号为只读演示, 无法确认告警'); return; }
    if(r.ok){ toast('✓ 已确认'); loadAlarms(); } else toast('确认失败 ('+r.status+')');
  }catch(e){ toast('确认失败'); }
}
loadAlarms();
notifySync(); if(NOTIFY.on){ NOTIFY._init=false; notifyCheck(); }

/* ---- P3: SLO 可用性 (historian 真数据) ---- */
const LAT_COL={lab:'#7c3aed',car:'#0891b2',arm:'#d97706'};
function fillSegs(el, segs){
  el.innerHTML='';
  segs.forEach(s=>{ const d=document.createElement('i'); d.className='seg '+s; el.appendChild(d); });
}
async function drawLatency(k, svg, lbl){
  try{
    const d=await fetch('/api/history?sys='+k+'&hours=24',{cache:'no-store'}).then(r=>r.json());
    const pts=(d.points||[]).map(p=>({ts:p.ts, ms:(p.real>0&&p.real_ms!=null)?p.real_ms:p.mirror_ms}))
      .filter(p=>p.ms!=null);
    if(pts.length<2){ if(lbl) lbl.textContent='样本不足 — historian 持续累积中 (30s/样)'; return; }
    const t0=pts[0].ts, t1=pts[pts.length-1].ts||(t0+1);
    const mx=Math.max(10, ...pts.map(p=>p.ms));
    const xy=pts.map(p=>(((p.ts-t0)/(t1-t0))*600).toFixed(1)+','+(58-(p.ms/mx*50)).toFixed(1)).join(' ');
    const col=LAT_COL[k]||'#2563eb';
    const last=pts[pts.length-1];
    svg.innerHTML='<polyline points="'+xy+'" fill="none" stroke="'+col+'" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'+
      '<circle cx="600" cy="'+(58-(last.ms/mx*50)).toFixed(1)+'" r="3.5" fill="'+col+'"/>';
    if(lbl) lbl.textContent='当前服务路径延迟 · 24h · 峰值 '+Math.round(mx)+'ms · 现值 '+Math.round(last.ms)+'ms · '+pts.length+' 桶';
  }catch(e){}
}
async function loadSlo(){
  try{
    const d=await fetch('/api/uptime',{cache:'no-store'}).then(r=>r.json());
    const host=document.getElementById('sloGrid'); if(!host) return; host.innerHTML='';
    ['lab','car','arm'].forEach(k=>{
      const m=OPS_META[k];
      const w24=(d.windows['24h']||{})[k]||{}, w7=(d.windows['7d']||{})[k]||{};
      const pc=v=>v==null?'—':v+'%';
      const card=document.createElement('div'); card.className='slo';
      card.innerHTML='<div class="sh"><span>'+m.ic+'</span><b>'+m.nm+'</b>'+
        '<span class="sv">UI 可用 <b>'+pc(w24.availability_pct)+'</b> /24h</span>'+
        '<span class="sv">真机在线 <b>'+pc(w24.real_pct)+'</b> /24h</span>'+
        '<span class="sv">7d UI 可用 <b>'+pc(w7.availability_pct)+'</b></span></div>'+
        '<div class="sb" data-w="24"></div><div class="sbl"><span>24h 前</span><span>现在</span></div>'+
        '<div class="sb" data-w="7d"></div><div class="sbl"><span>7 天前</span><span>现在</span></div>'+
        '<svg class="lat" viewBox="0 0 604 64" preserveAspectRatio="none"></svg><div class="latl">延迟曲线加载中…</div>';
      fillSegs(card.querySelectorAll('.sb')[0], w24.segments||[]);
      fillSegs(card.querySelectorAll('.sb')[1], w7.segments||[]);
      host.appendChild(card);
      drawLatency(k, card.querySelector('.lat'), card.querySelector('.latl'));
    });
  }catch(e){}
}
setInterval(()=>{ if(cur==='ops') loadSlo(); }, 60000);

/* ---- P4: 资产数字孪生 ---- */
const AG_SERV={real:['● 真机在线','real'],mirror:['◐ 镜像演示','mirror'],down:['○ 离线','down']};
function assetFocus(k){ go('home',{after:()=>{ if(window.focus3D) window.focus3D(k); }}); }
async function loadAssets(){
  try{
    const d=await fetch('/api/assets',{cache:'no-store'}).then(r=>r.json());
    const grid=document.getElementById('assetGrid'); if(!grid) return; grid.innerHTML='';
    const sel=document.getElementById('mAsset'); let opts='';
    let nAssets=0;
    (d.groups||[]).forEach(g=>{
      const card=document.createElement('div'); card.className='ag';
      const sv=(g.key==='vps')?'real':(g.serving||'down');
      const [stx,scl]=AG_SERV[sv]||AG_SERV.down;
      const can3d=['lab','car','arm'].includes(g.key);
      card.innerHTML='<div class="gh"><span class="ic">'+(g.icon||'')+'</span><b>'+g.name+'</b>'+
        '<span class="gserv '+scl+'">'+(g.key==='vps'?'● VPS 在线':stx)+'</span>'+
        (can3d?'<span class="g3d" data-k="'+g.key+'">⬢ 3D</span>':'')+'</div>'+
        '<div class="ghost">'+g.host+' · <b>'+g.ip+'</b></div>';
      const box=document.createElement('div');
      (g.children||[]).forEach(c=>{
        nAssets++;
        const row=document.createElement('div'); row.className='ac';
        let st, scls;
        if(c.status==='follow'){ st=stx.replace('● ','').replace('◐ ','').replace('○ ',''); scls=sv==='real'?'ok':sv==='mirror'?'dim':'warn'; }
        else if(c.status==='active'){ st='active'; scls='ok'; }
        else{ st=c.status; scls='warn'; }
        row.innerHTML='<span class="nm"></span><span class="kind"></span><span class="spec"></span>'+
          (c.maint_n?'<span class="mn">🔧 '+c.maint_n+'</span>':'')+'<span class="st '+scls+'"></span>';
        row.querySelector('.nm').textContent=c.name;
        row.querySelector('.kind').textContent=c.kind;
        row.querySelector('.spec').textContent=c.spec;
        row.querySelector('.st').textContent=st;
        box.appendChild(row);
        opts+='<option value="'+c.id+'">'+(g.icon||'')+' '+c.name+'</option>';
      });
      card.appendChild(box);
      const g3=card.querySelector('.g3d'); if(g3) g3.onclick=()=>assetFocus(g3.dataset.k);
      grid.appendChild(card);
    });
    if(sel && opts) sel.innerHTML=opts;
    const ts=document.getElementById('assetsTs');
    if(ts) ts.textContent=(d.groups||[]).length+' 资产组 · '+nAssets+' 资产 · '+new Date().toLocaleTimeString('zh-CN');
    loadMaint();
  }catch(e){}
}
async function loadMaint(){
  try{
    const d=await fetch('/api/maintenance?limit=40',{cache:'no-store'}).then(r=>r.json());
    const list=document.getElementById('maintList'); if(!list) return; list.innerHTML='';
    if(!(d.entries||[]).length){ list.innerHTML='<div class="oe-empty">暂无维保记录 — 用上方表单添加第一条</div>'; return; }
    (d.entries||[]).forEach(m=>{
      const t=new Date(m.ts*1000);
      const row=document.createElement('div'); row.className='ev info';
      row.innerHTML='<span class="sev"></span><span class="t"></span><span class="sy"></span><span class="msg"></span>';
      row.querySelector('.t').textContent=t.toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'})+' '+t.toLocaleTimeString('zh-CN',{hour12:false});
      row.querySelector('.sy').textContent=m.asset;
      row.querySelector('.msg').textContent=(m.author?('['+m.author+'] '):'')+m.note;
      list.appendChild(row);
    });
  }catch(e){}
}
async function addMaint(){
  const asset=(document.getElementById('mAsset')||{}).value;
  const inp=document.getElementById('mNote');
  const note=(inp&&inp.value||'').trim();
  if(!asset||!note){ toast('请选资产并填写维保内容'); return; }
  try{
    const r=await fetch('/api/maintenance',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({asset,note})});
    if(r.status===403){ toast('评委账号为只读演示, 无法记录维保'); return; }
    if(r.ok){ toast('✓ 已记录'); inp.value=''; loadAssets(); } else toast('记录失败 ('+r.status+')');
  }catch(e){ toast('记录失败'); }
}

/* ---- H2 演示就绪预检 (Preflight) ---- */
const PF_SEV={crit:['crit','🔴'],warn:['warn','🟡'],info:['info','🔵'],ok:['ok','🟢'],
  accepted:['info','🔵'],open:['warn','🟡']};
const PF_GICO={'三机服务':'🛰','镜像兜底':'🪞','告警':'🚨','数据底座':'🗄','KPI 真值':'📊','备份':'💾','安全':'🛡'};
async function loadPreflight(){
  const vEl=document.getElementById('pfVerdict'), cEl=document.getElementById('pfChecks');
  if(cEl && !cEl.querySelector('.pf-grp')) cEl.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    const d=await fetch('/api/preflight',{cache:'no-store'}).then(r=>r.json());
    if(d.error){ if(vEl) vEl.innerHTML='<div class="oe-empty">预检失败</div>'; return; }
    const s=d.summary||{};
    // —— GO/NO-GO 横幅 ——
    if(vEl){
      const go=s.go;
      vEl.className='pf-verdict '+(go?(s.warn?'warn':'go'):'nogo');
      vEl.innerHTML=
        '<div class="pf-vbig">'+(go?'✓':'✕')+'<span>'+(s.verdict||(go?'GO':'NO-GO'))+'</span></div>'+
        '<div class="pf-vsub">'+(go
          ? (s.warn?('可开演, 有 '+s.warn+' 项注意'):'全部就绪, 可以开演')
          : ('有 '+s.crit+' 项阻断必须先处理'))+'</div>'+
        '<div class="pf-vcounts">'+
          '<span class="pfc crit">🔴 '+(s.crit||0)+' 阻断</span>'+
          '<span class="pfc warn">🟡 '+(s.warn||0)+' 注意</span>'+
          '<span class="pfc info">🔵 '+(s.info||0)+' 降级</span>'+
          '<span class="pfc ok">🟢 '+(s.ok||0)+' 就绪</span>'+
          '<span class="pfc tot">'+(s.total||0)+' 项 · '+new Date().toLocaleTimeString('zh-CN')+'</span>'+
        '</div>';
    }
    // —— 清单 (按 group 归组, 组内已按严重度排序) ——
    if(cEl){
      const groups={}, order=[];
      (d.checks||[]).forEach(c=>{ if(!groups[c.group]){groups[c.group]=[];order.push(c.group);} groups[c.group].push(c); });
      let html='';
      order.forEach(g=>{
        html+='<div class="pf-grp"><div class="pf-gh">'+(PF_GICO[g]||'•')+' '+esc(g)+'</div>';
        groups[g].forEach(c=>{
          const [cls,ico]=PF_SEV[c.status]||PF_SEV.info;
          html+='<div class="pf-row '+cls+'"><span class="pf-ico">'+ico+'</span>'+
            '<div class="pf-rt"><b>'+esc(c.label)+'</b><span>'+esc(c.detail)+'</span>'+
            (c.fix?'<code class="pf-fix">↳ '+esc(c.fix)+'</code>':'')+'</div></div>';
        });
        html+='</div>';
      });
      cEl.innerHTML=html||'<div class="oe-empty">无检查项</div>';
    }
    // —— 在险册 ——
    const rEl=document.getElementById('pfRisks');
    if(rEl){
      const RL={crit:['crit','🔴 严重'],open:['warn','🟡 待办'],accepted:['info','🔵 已接受']};
      if(!(d.risks||[]).length) rEl.innerHTML='<div class="oe-empty">无在册风险 ✓</div>';
      else rEl.innerHTML=(d.risks||[]).map(r=>{
        const [cls,lab]=RL[r.level]||RL.open;
        return '<div class="pf-risk '+cls+'"><div class="pf-rkh"><b>'+esc(r.title)+'</b>'+
          '<span class="pf-rlvl '+cls+'">'+lab+'</span></div>'+
          '<div class="pf-rkd">'+esc(r.detail)+'</div>'+
          '<div class="pf-rkm">缓解: '+esc(r.mitigation)+'</div></div>';
      }).join('');
    }
    // —— 备份心跳 ——
    const bEl=document.getElementById('pfBackup'), bk=d.backup||{};
    if(bEl){
      if(bk.configured){
        const cls=bk.fresh?'ok':'warn';
        bEl.innerHTML='<div class="pf-bk '+cls+'"><div class="pf-bkv">'+
          (bk.age_h!=null?bk.age_h+'h 前':'—')+'</div>'+
          '<div class="pf-bkl">上次备份 · 保留 '+(bk.kept_days||'?')+' 天</div>'+
          (bk.detail?'<div class="pf-bkd">'+esc(bk.detail)+'</div>':'')+'</div>';
      }else{
        const lv=bk.historian_liveness_h;
        bEl.innerHTML='<div class="pf-bk info"><div class="pf-bkv" style="font-size:1rem">心跳未上报</div>'+
          (lv!=null?'<div class="pf-bkl">historian 活性 '+lv+'h (数据面在跑)</div>':'')+
          '<div class="pf-bkd">'+esc(bk.detail||'')+'</div></div>';
      }
    }
    const ts=document.getElementById('pfTs');
    if(ts) ts.textContent=(s.go?'就绪 GO':'阻断 NO-GO')+' · '+(s.total||0)+' 项体检 · '+new Date().toLocaleTimeString('zh-CN');
    pfPipApply(s);
  }catch(e){ if(vEl) vEl.innerHTML='<div class="oe-empty">预检请求失败</div>'; }
}
function pfPipApply(s){
  const pip=document.getElementById('pfPip');
  if(pip){
    if(s && s.crit){ pip.style.display='block'; pip.className='pf-pip crit'; }
    else if(s && s.warn){ pip.style.display='block'; pip.className='pf-pip warn'; }
    else pip.style.display='none';
  }
  const mp=document.getElementById('mmPip');
  if(mp) mp.className='mm-pip'+(s&&s.crit?' crit':s&&s.warn?' warn':'');
}
/* ---- H23 离线降级指示 ---- */
function offApply(){
  const bar=document.getElementById('offbar'); if(!bar) return;
  const off=!navigator.onLine; bar.classList.toggle('show', off);
  document.body.classList.toggle('is-offline', off);
}
window.addEventListener('online', ()=>{ offApply(); toast('✓ 网络已恢复','ok'); try{ pollFleet();pollKpi(); }catch(e){} });
window.addEventListener('offline', ()=>{ offApply(); });
offApply();

/* ---- 顶栏"更多"菜单 ---- */
function moreToggle(e){ if(e) e.stopPropagation();
  const m=document.getElementById('moreMenu'); const open=m.classList.toggle('show');
  document.getElementById('btnMore').classList.toggle('on', open);
  document.body.classList.toggle('more-open', open);
  // 同步通知开关文案
  const n=document.getElementById('mmNotify'); if(n) n.textContent=(NOTIFY.on?'🔕 关闭桌面通知':'📣 开启桌面通知');
}
function moreClose(){ const m=document.getElementById('moreMenu'); if(m) m.classList.remove('show');
  const b=document.getElementById('btnMore'); if(b) b.classList.remove('on'); document.body.classList.remove('more-open'); }
function moreGo(fn){ moreClose(); try{ fn(); }catch(e){} }
document.addEventListener('click', e=>{ if(!e.target.closest('.more-wrap')) moreClose(); });
// 后台轻量轮询: 给顶栏 ✅ 按钮一个红/黄角标 (NO-GO/注意), 不打扰当前页
async function pfPipPoll(){
  try{ const d=await fetch('/api/preflight',{cache:'no-store'}).then(r=>r.json()); if(d.summary) pfPipApply(d.summary); }catch(e){}
}
pfPipPoll(); setInterval(pfPipPoll, 90000);

/* ---- G4 Scientific Agent Studio ---- */
let _agentStudio=null;
async function loadAgentStudio(){
  const host=document.getElementById('studioShell');
  if(host) host.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    _agentStudio=await fetch('/api/agent_studio',{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _agentStudio={summary:{},stages:[],agents:[],objects:[],protocols:[],guardrails:[String(e)]};
  }
  studioRender();
}
function studioRender(){
  const d=_agentStudio||{}, host=document.getElementById('studioShell');
  if(!host) return;
  const s=d.summary||{};
  const ts=document.getElementById('studioTs');
  if(ts) ts.textContent=uiTerm(d.source||'source unknown')+' · '+(d.release||'')+' · '+new Date().toLocaleTimeString('zh-CN');
  host.innerHTML='<div class="studio-kpis">'+
    '<div><b>'+esc(s.workorders??0)+'</b><span>'+uiText('工单','work orders')+'</span></div>'+
    '<div><b>'+esc(s.open??0)+'</b><span>'+uiText('进行中','open')+'</span></div>'+
    '<div><b>'+esc(s.done??0)+'</b><span>'+uiText('已关闭','closed')+'</span></div>'+
    '<div><b>'+esc(s.audit_events??0)+'</b><span>'+uiText('审计事件','wo_log events')+'</span></div></div>'+
    '<div class="studio-flow">'+(d.stages||[]).map(x=>'<div><span>'+esc(String(x.idx+1).padStart(2,'0'))+'</span><b>'+esc(x.name||'')+'</b><i>'+uiEsc(x.owner||'')+'</i><em>'+esc(x.count??0)+' '+uiText('活动','active')+' · '+uiEsc(x.evidence||'')+'</em></div>').join('')+'</div>'+
    '<div class="studio-grid"><section><h2>'+uiText('Agent 合约','Agent contracts')+'</h2><div class="studio-agents">'+(d.agents||[]).map(a=>'<article><span>'+uiEsc(a.owner||'')+'</span><b>'+uiEsc(a.title||'')+'</b><p><strong>'+uiText('输入','Input')+'</strong> '+uiEsc(a.input||'')+'</p><p><strong>'+uiText('输出','Output')+'</strong> '+uiEsc(a.output||'')+'</p><div>'+((a.tools||[]).map(t=>'<code>'+esc(t)+'</code>').join(''))+'</div><i>'+uiEsc(a.guardrail||'')+'</i></article>').join('')+'</div></section>'+
    '<aside><h2>'+uiText('最近实验对象','Recent experiment objects')+'</h2><div class="studio-objects">'+((d.objects||[]).slice(0,10).map(studioObjectHtml).join('')||'<div class="studio-empty">'+uiText('暂无公开工单或材料对象。','No public work orders or material objects yet.')+'</div>')+'</div></aside></div>'+
    '<div class="studio-bottom"><div><h2>'+uiText('协议模板','Protocol templates')+'</h2>'+((d.protocols||[]).map(p=>'<article><b>'+uiEsc(p.name||'')+'</b><p>'+uiEsc((p.steps||[]).join(' -> '))+'</p></article>').join(''))+'</div><div><h2>'+uiText('安全边界','Guardrails')+'</h2>'+((d.guardrails||[]).map(g=>'<p>'+uiEsc(g)+'</p>').join(''))+'</div></div>';
}
function studioObjectHtml(o){
  const id=Number(o.id||0);
  const mid=String(o.object_id||o.trace_id||o.formula||o.code||'');
  const label=id?uiText('打开工单','open WO'):uiText('打开材料','open material');
  return '<div><b>'+esc(o.code||o.formula||'object')+'</b><span>'+esc(o.formula||'')+' · '+uiEsc(o.verdict||o.state||'pending')+'</span><em>'+uiEsc(o.stage_name||'')+'</em><button data-wo="'+esc(id)+'" data-mid="'+esc(mid)+'" onclick="studioOpenObject(this)">'+label+'</button></div>';
}
function studioOpenObject(btn){
  const id=Number(btn&&btn.dataset?btn.dataset.wo:0);
  if(id){ studioOpenWo(id); return; }
  const mid=btn&&btn.dataset?String(btn.dataset.mid||''):'';
  if(mid){ detailOpen('material', mid); return; }
  go('atlas');
}
function studioOpenWo(id){
  id=Number(id||0);
  if(!id){ go('mq'); return; }
  go('mq',{after:()=>woDetail(id)});
}

/* ---- G5 Lab-FSD v2 World Model Console ---- */
let _fsdConsole=null;
async function loadFsdConsole(){
  const host=document.getElementById('fsdShell');
  if(host) host.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    _fsdConsole=await fetch('/api/lab_fsd_console',{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _fsdConsole={summary:{serving:'offline',safety_gate:'blocked',public_control:'read-only'},occupancy:{cols:0,rows:0,cells:[]},future:[],policy_tokens:[],systems:[],endpoints:[],boundaries:[String(e)]};
  }
  fsdRender();
}
function fsdClass(v){ return String(v||'unknown').toLowerCase().replace(/[^a-z0-9_-]/g,'')||'unknown'; }
function fsdPct(v){ v=Number(v||0); return isFinite(v)?Math.round(v*100)+'%':'—'; }
function fsdSource(src){ return '<span class="fsd-src s-'+fsdClass(src)+'">'+uiEsc(src||'unknown')+'</span>'; }
function fsdRender(){
  const d=_fsdConsole||{}, host=document.getElementById('fsdShell');
  if(!host) return;
  const s=d.summary||{}, gate=d.safety_gate||{}, occ=d.occupancy||{}, cells=occ.cells||[];
  const ts=document.getElementById('fsdTs');
  if(ts) ts.textContent=uiText('来源','source')+' '+uiTerm(d.source||'unknown')+' · '+(d.release||'')+' · '+new Date().toLocaleTimeString('zh-CN');
  host.innerHTML=
    '<div class="fsd-kpis">'+
      '<div><b>'+uiEsc(s.serving||'unknown')+'</b><span>'+uiText('服务来源','serving source')+'</span></div>'+
      '<div class="gate-'+fsdClass(s.safety_gate)+'"><b>'+uiEsc(s.safety_gate||'review')+'</b><span>'+uiText('安全门禁','safety gate')+'</span></div>'+
      '<div><b>'+esc(s.active_endpoint_count??0)+'</b><span>'+uiText('公开 FSD 端点','public FSD endpoints')+'</span></div>'+
      '<div><b>'+uiEsc(s.public_control||'read-only')+'</b><span>'+uiText('公网控制策略','public control policy')+'</span></div>'+
    '</div>'+
    '<div class="fsd-main"><section class="fsd-bev"><div class="fsd-head"><h2>'+uiText('BEV 占据网格','BEV Occupancy')+'</h2>'+fsdSource(s.world_model_source||d.source)+'</div>'+
      '<div class="fsd-gridmap" style="--fsd-cols:'+esc(occ.cols||9)+'">'+
        cells.map(c=>'<span class="fsd-cell c-'+fsdClass(c.level)+'" title="'+uiEsc((c.label||'')+' · '+(c.source||''))+'"><i>'+uiEsc(c.short||'')+'</i></span>').join('')+
      '</div><div class="fsd-legend"><span><i class="c-route"></i>'+uiText('路线','route')+'</span><span><i class="c-ego"></i>'+uiText('本车','ego')+'</span><span><i class="c-shadow"></i>'+uiText('影子规划','shadow')+'</span><span><i class="c-blocked"></i>'+uiText('阻塞','blocked')+'</span></div></section>'+
      '<aside class="fsd-side"><div class="fsd-head"><h2>'+uiText('安全门禁','Safety Gate')+'</h2>'+fsdSource(gate.source||d.source)+'</div><div class="fsd-gate gate-'+fsdClass(gate.state)+'"><b>'+uiEsc(gate.state||'review')+'</b><p>'+uiEsc(gate.reason||'No public gate reason exposed.')+'</p></div>'+
      '<div class="fsd-head small"><h2>'+uiText('策略 Token','Policy Tokens')+'</h2></div><div class="fsd-tokens">'+((d.policy_tokens||[]).map(fsdTokenHtml).join(''))+'</div></aside></div>'+
    '<div class="fsd-lower"><section><div class="fsd-head"><h2>'+uiText('未来 BEV 滚动预测','Future BEV Rollout')+'</h2>'+fsdSource(d.source)+'</div><div class="fsd-future">'+((d.future||[]).map(f=>'<article><span>'+esc(f.t||'')+'</span><b>'+uiEsc(f.intent||'')+'</b><em>'+uiEsc(f.risk||'')+'</em><p>'+uiEsc(f.pose||'')+'</p></article>').join(''))+'</div></section>'+
      '<section><div class="fsd-head"><h2>'+uiText('只读端点','Read-only Endpoints')+'</h2>'+fsdSource(s.serving)+'</div><div class="fsd-endpoints">'+((d.endpoints||[]).map(fsdEndpointHtml).join(''))+'</div></section></div>'+
    '<div class="fsd-lower"><section><div class="fsd-head"><h2>'+uiText('传感器 / 模型栈','Sensor / Model Stack')+'</h2></div><div class="fsd-systems">'+((d.systems||[]).map(x=>'<article><b>'+uiEsc(x.name||'')+'</b><span>'+uiEsc(x.role||'')+'</span><em>'+uiEsc(x.state||'')+' · '+uiEsc(x.source||'')+'</em></article>').join(''))+'</div></section>'+
      '<section><div class="fsd-head"><h2>'+uiText('公网安全边界','Public Safety Boundary')+'</h2></div><div class="fsd-boundary">'+((d.boundaries||[]).map(x=>'<p>'+uiEsc(x)+'</p>').join(''))+'</div></section></div>';
}
function fsdTokenHtml(t){
  return '<article><div><b>'+uiEsc(t.token||'')+'</b>'+fsdSource(t.source||'')+'</div><meter min="0" max="100" value="'+esc(Math.round(Number(t.confidence||0)*100))+'"></meter><span>'+fsdPct(t.confidence)+' · '+uiEsc(t.reason||'')+'</span></article>';
}
function fsdEndpointHtml(e){
  const brief=e.brief||{};
  const bits=Object.keys(brief).slice(0,4).map(k=>'<code>'+esc(k)+': '+esc(Array.isArray(brief[k])?brief[k].join(','):brief[k])+'</code>').join('');
  return '<article class="'+(e.ok?'ok':'off')+'"><b>'+esc(e.path||'')+'</b><span>HTTP '+esc(e.code||0)+' · '+uiEsc(e.source||'offline')+'</span><div>'+bits+'</div></article>';
}

/* ---- G6 Autonomous Experiment Replay ---- */
let _replayData=null, _replayPct=58, _replayFault='', _replayTimer=null;
async function loadExperimentReplay(){
  const host=document.getElementById('replayShell');
  if(host && !_replayData) host.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    const qs='?pct='+encodeURIComponent(_replayPct)+'&fault='+encodeURIComponent(_replayFault||'');
    _replayData=await fetch('/api/experiment_replay'+qs,{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _replayData={trace:{},stages:[],events:[],faults:[],actuators:[],guardrails:[String(e)],twin:{map:{},sample:{}}};
  }
  replayRender();
}
function replayScrub(v){
  _replayPct=Math.max(0,Math.min(100,Number(v)||0));
  const lab=document.getElementById('replayPctLabel'); if(lab) lab.textContent=Math.round(_replayPct)+'%';
  clearTimeout(_replayTimer); _replayTimer=setTimeout(loadExperimentReplay,130);
}
function replayFault(f){
  _replayFault=String(f||'');
  document.querySelectorAll('#replayFaults button').forEach(b=>b.classList.toggle('on',(b.dataset.f||'')===_replayFault));
  loadExperimentReplay();
}
function replayRender(){
  const d=_replayData||{}, host=document.getElementById('replayShell');
  if(!host) return;
  const ts=document.getElementById('replayTs');
  if(ts) ts.textContent=uiText('回放','replay')+' '+(d.progress_pct??_replayPct)+'% · '+(d.release||'')+' · '+new Date().toLocaleTimeString('zh-CN');
  const pct=document.getElementById('replayPct'); if(pct) pct.value=String(d.progress_pct??_replayPct);
  const lab=document.getElementById('replayPctLabel'); if(lab) lab.textContent=Math.round(Number(d.progress_pct??_replayPct))+'%';
  const fb=document.getElementById('replayFaults');
  if(fb && d.faults){
    fb.innerHTML='<button class="'+(!_replayFault?'on':'')+'" data-f="" onclick="replayFault(this.dataset.f)">'+uiText('正常','normal')+'</button>'+
      d.faults.map(f=>'<button class="'+((_replayFault===f.key)?'on':'')+'" data-f="'+esc(f.key)+'" onclick="replayFault(this.dataset.f)">'+uiEsc(f.label||f.key)+'</button>').join('');
  }
  const t=d.trace||{}, twin=d.twin||{}, sample=twin.sample||{};
  host.innerHTML=
    '<div class="replay-kpis">'+
      '<div><b>'+uiEsc(t.trace_id||t.code||'replay')+'</b><span>'+uiText('追踪 / 对象','trace / object')+'</span></div>'+
      '<div><b>'+uiEsc(t.formula||'formula')+'</b><span>'+uiEsc((t.dopant||'')+(t.site?' @ '+t.site:''))+'</span></div>'+
      '<div><b>'+uiEsc(d.active_fault?d.active_fault.label:'normal')+'</b><span>'+uiText('故障注入','fault injection')+'</span></div>'+
      '<div><b>'+uiEsc(sample.stage||'sample path')+'</b><span>'+uiText('样品状态','sample state')+'</span></div>'+
    '</div>'+
    '<div class="replay-stages">'+((d.stages||[]).map(replayStageHtml).join(''))+'</div>'+
    '<div class="replay-main"><section><div class="replay-head"><h2>'+uiText('样品流地图','Sample Flow Map')+'</h2>'+fsdSource((twin.source_label||{}).car||'replay')+'</div>'+replayMapHtml(twin)+
      '<div class="replay-actuators">'+((d.actuators||[]).map(a=>'<article><b>'+uiEsc(a.label||'')+'</b><span>'+uiEsc(a.evidence||'')+'</span><em>'+uiEsc(a.state||'')+' · '+uiText('公网命令','public command')+' '+uiEsc(a.public_command||'disabled')+'</em></article>').join(''))+'</div></section>'+
      '<aside><div class="replay-head"><h2>'+uiText('事件栈','Event Stack')+'</h2>'+fsdSource(d.mode||'replay')+'</div><div class="replay-events">'+((d.events||[]).map(replayEventHtml).join(''))+'</div></aside></div>'+
    '<div class="replay-bottom"><section><div class="replay-head"><h2>'+uiText('故障注入','Fault Injection')+'</h2></div>'+replayFaultPanel(d)+'</section>'+
      '<section><div class="replay-head"><h2>'+uiText('回放数据表','Replay Data Table')+'</h2></div>'+replayTable(d)+'</section></div>'+
    '<div class="replay-guardrails">'+((d.guardrails||[]).map(g=>'<p>'+uiEsc(g)+'</p>').join(''))+'</div>';
}
function replayStageHtml(s){
  return '<article class="st-'+fsdClass(s.state)+'"><span>'+esc(String((s.idx||0)+1).padStart(2,'0'))+'</span><b>'+uiEsc(s.title||'')+'</b><em>'+uiEsc(s.owner||'')+'</em><p>'+uiEsc(s.action||'')+'</p><i>'+uiEsc(s.evidence||'')+'</i></article>';
}
function replayEventHtml(e){
  return '<article class="'+fsdClass(e.severity||'info')+'"><span>'+esc(e.t||'')+'</span><b>'+uiEsc(e.system||'')+'</b><p>'+uiEsc(e.event||'')+'</p><em>'+uiEsc(e.source||'')+(e.recovery?' · '+uiText('恢复','recovery')+': '+uiEsc(e.recovery):'')+'</em></article>';
}
function replayMapHtml(twin){
  const m=(twin||{}).map||{}, nodes=m.nodes||[], route=m.route||[], zones=m.zones||[], sample=(twin||{}).sample||{};
  return '<div class="replay-map">'+
    zones.map(z=>'<span class="rz '+fsdClass(z.risk)+'" style="left:'+esc(z.x||0)+'%;top:'+esc(z.y||0)+'%;width:'+esc(z.w||10)+'%;height:'+esc(z.h||10)+'%" title="'+uiEsc(z.label||'zone')+'"></span>').join('')+
    route.map((r,i)=>'<span class="rr" style="left:'+esc(r.x||0)+'%;top:'+esc(r.y||0)+'%"><i>'+esc(i+1)+'</i></span>').join('')+
    nodes.map(n=>'<span class="rn" style="left:'+esc(n.x||0)+'%;top:'+esc(n.y||0)+'%"><b>'+uiEsc(n.label||n.id||'')+'</b></span>').join('')+
    '<span class="rsample" style="left:'+esc(sample.x||0)+'%;top:'+esc(sample.y||0)+'%"><i>'+uiText('样品','sample')+'</i></span>'+
    '</div>';
}
function replayFaultPanel(d){
  const af=d.active_fault;
  if(!af) return '<div class="replay-fault-panel ok"><b>'+uiText('正常回放','normal replay')+'</b><p>'+uiText('未注入故障。可用上方故障按钮展示系统如何保持状态并恢复，且不开放公网机器人命令。','No injected fault. Use the fault chips above to show how the system holds state and recovers without public robot commands.')+'</p></div>';
  return '<div class="replay-fault-panel '+fsdClass(af.severity)+'"><b>'+uiEsc(af.label||af.key)+'</b><p>'+uiEsc(af.symptom||'')+'</p><em>'+uiEsc(af.recovery||'')+'</em></div>';
}
function replayTable(d){
  return '<div class="replay-table"><table><thead><tr><th>t</th><th>'+uiText('阶段','stage')+'</th><th>'+uiText('系统','system')+'</th><th>'+uiText('来源','source')+'</th></tr></thead><tbody>'+
    ((d.events||[]).map(e=>'<tr><td>'+esc(e.t||'')+'</td><td>'+uiEsc(e.stage||'')+'</td><td>'+uiEsc(e.system||'')+'</td><td>'+uiEsc(e.source||'')+'</td></tr>').join(''))+
    '</tbody></table></div>';
}

/* ---- G7 Cloud Lab Command Center ---- */
let _cloudCommand=null;
async function loadCloudCommand(){
  const host=document.getElementById('commandShell');
  if(host) host.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    _cloudCommand=await fetch('/api/cloud_command_center',{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _cloudCommand={summary:{},lanes:[],resources:[],calendar:[],blockers:[],approvals:[],guardrails:[String(e)]};
  }
  commandRender();
}
function commandRender(){
  const d=_cloudCommand||{}, host=document.getElementById('commandShell');
  if(!host) return;
  const s=d.summary||{};
  const ts=document.getElementById('commandTs');
  if(ts) ts.textContent=(d.release||'')+' · '+new Date().toLocaleTimeString('zh-CN');
  host.innerHTML=
    '<div class="command-kpis">'+
      '<div><b>'+esc(s.open_tasks??0)+'</b><span>'+uiText('进行中任务','open tasks')+'</span></div>'+
      '<div><b>'+esc(s.resources??0)+'</b><span>'+uiText('已排程资源','scheduled resources')+'</span></div>'+
      '<div><b>'+esc(s.blockers??0)+'</b><span>'+uiText('依赖阻塞','dependency blockers')+'</span></div>'+
      '<div><b>'+esc(s.alarms??0)+'</b><span>'+uiText('活动告警','active alarms')+'</span></div>'+
    '</div>'+
    '<div class="command-lanes">'+((d.lanes||[]).map(commandLaneHtml).join(''))+'</div>'+
    '<div class="command-grid"><section><div class="command-head"><h2>'+uiText('资源看板','Resource Board')+'</h2>'+fsdSource('live/mirror/replay')+'</div><div class="command-resources">'+((d.resources||[]).map(commandResourceHtml).join(''))+'</div></section>'+
      '<aside><div class="command-head"><h2>'+uiText('资源日历','Resource Calendar')+'</h2>'+fsdSource('schedule')+'</div><div class="command-calendar">'+((d.calendar||[]).map(c=>'<article><span>'+esc(c.t||'')+'</span><b>'+uiEsc(c.resource||'')+'</b><p>'+uiEsc(c.slot||'')+'</p><em>'+uiEsc(c.state||'')+'</em></article>').join(''))+'</div></aside></div>'+
    '<div class="command-bottom"><section><div class="command-head"><h2>'+uiText('阻塞项 / 人工决策','Blockers / Human Decisions')+'</h2></div><div class="command-blockers">'+((d.blockers||[]).map(b=>'<article><b>'+esc(b.task||'')+'</b><p>'+uiEsc(b.blocker||'')+'</p><em>'+uiEsc(b.decision||'')+'</em><button onclick="go(&quot;tasks&quot;)">'+uiText('打开任务','open tasks')+'</button></article>').join(''))+'</div></section>'+
      '<section><div class="command-head"><h2>'+uiText('审批 / 风险锁','Approvals / Risk Locks')+'</h2></div><div class="command-approvals">'+((d.approvals||[]).map(a=>'<article><b>'+uiEsc(a.gate||'')+'</b><span>'+uiEsc(a.state||'')+' · '+uiEsc(a.who||'')+'</span><p>'+uiEsc(a.reason||'')+'</p></article>').join(''))+'</div></section></div>'+
    '<div class="command-bottom"><section><div class="command-head"><h2>'+uiText('样品 / 资产位置','Sample / Asset Locations')+'</h2></div><div class="command-locations">'+((d.sample_locations||[]).map(l=>'<article><b>'+esc(l.label||'')+'</b><span>'+uiEsc(l.source||'')+'</span><em>x '+esc(l.x??'')+' · y '+esc(l.y??'')+'</em></article>').join(''))+'</div></section>'+
      '<section><div class="command-head"><h2>'+uiText('资产分组 / 维护','Asset Groups / Maintenance')+'</h2></div><div class="command-assets">'+((d.asset_groups||[]).map(a=>'<article><b>'+esc(a.name||a.key||'')+'</b><span>'+esc(a.children||0)+' '+uiText('个资产','assets')+'</span></article>').join(''))+((d.maintenance||[]).slice(0,4).map(m=>'<article><b>'+esc(m.asset||'')+'</b><span>'+esc(m.note||'')+'</span></article>').join(''))+'</div></section></div>'+
    '<div class="command-actions"><button onclick="go(&quot;tasks&quot;)">'+uiText('任务','Tasks')+'</button><button onclick="go(&quot;replay&quot;)">'+uiText('回放','Replay')+'</button><button onclick="go(&quot;twin&quot;)">'+uiText('孪生','Twin')+'</button><button onclick="go(&quot;assets&quot;)">'+uiText('资产','Assets')+'</button><button onclick="go(&quot;obs&quot;)">'+uiText('可观测性','Observability')+'</button></div>'+
    '<div class="command-guardrails">'+((d.guardrails||[]).map(g=>'<p>'+uiEsc(g)+'</p>').join(''))+'</div>';
}
function commandLaneHtml(l){
  const items=l.items||[];
  return '<section><div class="command-head"><h2>'+uiEsc(l.label||'')+'</h2><span>'+esc(l.count||0)+' '+uiText('任务','tasks')+'</span></div>'+
    '<div class="command-tasklist">'+(items.length?items.map(commandTaskHtml).join(''):'<div class="command-empty">'+uiText('本泳道暂无公开任务。','No active public task in this lane.')+'</div>')+'</div></section>';
}
function commandTaskHtml(t){
  return '<article><b>'+esc(t.code||t.formula||'task')+'</b><span>'+esc(t.formula||'')+' · '+esc(t.stage_name||'')+'</span><p>'+uiEsc(t.next_action||'')+'</p><em>'+uiEsc(t.blocker||'no blocker')+'</em><button onclick="go(&quot;tasks&quot;)">'+uiText('查看','inspect')+'</button></article>';
}
function commandResourceHtml(r){
  return '<article><div><b>'+uiEsc(r.label||r.key||'')+'</b>'+fsdSource(r.source||r.serving||'unknown')+'</div><p>'+uiEsc(r.capability||'')+'</p><em>'+uiEsc(r.reservation||'')+'</em></article>';
}

/* ---- G8 Evidence / Defense Mode Pro ---- */
let _defenseMode=null;
async function loadDefenseMode(){
  const host=document.getElementById('defenseShell');
  if(host) host.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    _defenseMode=await fetch('/api/defense_mode',{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _defenseMode={summary:{},scripts:[],evidence:[],demo_paths:[],checklist:[],guardrails:[String(e)]};
  }
  defenseRender();
}
function defenseOpen(h){
  h=String(h||'');
  const map={'/':'home','/defense':'defense','/benchmark':'benchmark','/brain':'brain','/atlas':'atlas','/fsd':'fsd','/replay':'replay','/command':'command','/traces':'traces','/status':'status','/sec':'sec','/observability':'obs'};
  if(map[h]){ go(map[h]); return; }
  if(h.startsWith('/api/')||h==='/style.css'){ location.href=h; return; }
  location.href=h||'/';
}
function defenseRender(){
  const d=_defenseMode||{}, host=document.getElementById('defenseShell');
  if(!host) return;
  const s=d.summary||{};
  const ts=document.getElementById('defenseTs');
  if(ts) ts.textContent=(d.release||'')+' · '+new Date().toLocaleTimeString('zh-CN');
  host.innerHTML=
    '<div class="defense-kpis">'+
      '<div><b>'+esc(s.scripts??0)+'</b><span>'+uiText('脚本','scripts')+'</span></div>'+
      '<div><b>'+esc(s.evidence??0)+'</b><span>'+uiText('证据卡','evidence cards')+'</span></div>'+
      '<div><b>'+esc(s.demo_paths??0)+'</b><span>'+uiText('演示路径','demo paths')+'</span></div>'+
      '<div><b>'+esc(s.checklist??0)+'</b><span>'+uiText('评委检查','judge checks')+'</span></div>'+
    '</div>'+
    '<div class="defense-scripts">'+((d.scripts||[]).map(defenseScriptHtml).join(''))+'</div>'+
    '<div class="defense-grid"><section><div class="defense-head"><h2>'+uiText('证据矩阵','Evidence Matrix')+'</h2>'+fsdSource('page/API/source')+'</div><div class="defense-evidence">'+((d.evidence||[]).map(defenseEvidenceHtml).join(''))+'</div></section>'+
      '<aside><div class="defense-head"><h2>'+uiText('评委检查清单','Judge Checklist')+'</h2>'+fsdSource('pass/fail')+'</div><div class="defense-checks">'+((d.checklist||[]).map(c=>'<article class="'+fsdClass(c.state)+'"><b>'+uiEsc(c.item||'')+'</b><span>'+uiEsc(c.state||'')+'</span><p>'+uiEsc(c.evidence||'')+'</p></article>').join(''))+'</div></aside></div>'+
    '<div class="defense-bottom"><section><div class="defense-head"><h2>'+uiText('真机 / 离线演示路径','Live / Offline Demo Paths')+'</h2>'+fsdSource('labelled')+'</div><div class="defense-paths">'+((d.demo_paths||[]).map(defensePathHtml).join(''))+'</div></section>'+
      '<section><div class="defense-head"><h2>'+uiText('评委短名单','Judge Shortlist')+'</h2>'+fsdSource('3/5/8 min')+'</div><div class="defense-shortlist">'+((d.judge_shortlist||[]).map(x=>'<span>'+uiEsc(x)+'</span>').join(''))+'</div></section></div>'+
    '<div class="defense-guardrails">'+((d.guardrails||[]).map(g=>'<p>'+uiEsc(g)+'</p>').join(''))+'</div>';
}
function defenseScriptHtml(sc){
  return '<section><div class="defense-script-head"><span>'+esc(sc.duration||'')+'</span><b>'+uiEsc(sc.label||'')+'</b><p>'+uiEsc(sc.goal||'')+'</p></div><div class="defense-beats">'+
    ((sc.beats||[]).map(b=>'<article><span>'+esc(b.t||'')+'</span><b>'+uiEsc(b.title||'')+'</b><p>'+uiEsc(b.body||'')+'</p></article>').join(''))+'</div></section>';
}
function defenseEvidenceHtml(e){
  return '<article><div class="defense-card-head"><b>'+uiEsc(e.claim||e.key||'')+'</b>'+fsdSource(e.source||'evidence')+'</div><span>'+uiEsc(e.type||'')+'</span><p>'+uiEsc(e.boundary||'')+'</p><div class="defense-card-actions">'+
    '<button data-href="'+esc(e.href||'')+'" onclick="defenseOpen(this.dataset.href)">'+uiText('页面','page')+'</button>'+
    '<button data-href="'+esc(e.api||'')+'" onclick="defenseOpen(this.dataset.href)">API</button></div></article>';
}
function defensePathHtml(p){
  return '<article><div class="defense-card-head"><b>'+uiEsc(p.label||p.key||'')+'</b>'+fsdSource(p.source||'path')+'</div><div class="defense-path-steps">'+
    ((p.steps||[]).map((s,i)=>'<button data-href="'+esc(s.href||'')+'" onclick="defenseOpen(this.dataset.href)"><span>'+esc(i+1)+'</span><b>'+uiEsc(s.title||'')+'</b><em>'+uiEsc(s.evidence||'')+'</em></button>').join(''))+'</div></article>';
}

/* ---- G11 Global Benchmark ---- */
let _globalBenchmark=null;
async function loadGlobalBenchmark(){
  const host=document.getElementById('benchmarkShell');
  if(host && !_globalBenchmark) host.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    _globalBenchmark=await fetch('/api/global_benchmark',{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _globalBenchmark={dimensions:[],gates:[],sources:[],next_bets:[],honest_boundary:String(e),overall_score:0,gate_state:'unknown'};
  }
  benchmarkRender();
}
function benchmarkOpen(h){
  h=String(h||'');
  if(!h) return;
  if(/^https?:\/\//.test(h)){ window.open(h,'_blank','noopener'); return; }
  defenseOpen(h);
}
function benchmarkRender(){
  const d=_globalBenchmark||{}, host=document.getElementById('benchmarkShell');
  if(!host) return;
  const ts=document.getElementById('benchmarkTs');
  if(ts) ts.textContent=(d.release||'')+' · '+uiText('评分','score')+' '+esc(d.overall_score??'—')+' · '+new Date().toLocaleTimeString('zh-CN');
  host.innerHTML=
    '<div class="benchmark-kpis">'+
      '<div><b>'+esc(d.overall_score??'—')+'</b><span>'+uiText('总评分','overall score')+'</span></div>'+
      '<div><b>'+esc(d.dimension_count??((d.dimensions||[]).length))+'</b><span>'+uiText('对标维度','benchmark dimensions')+'</span></div>'+
      '<div><b>'+esc((d.gates||[]).filter(g=>g.state==='pass').length)+'/'+esc((d.gates||[]).length)+'</b><span>'+uiText('评分门禁','score gates')+'</span></div>'+
      '<div><b>'+uiEsc(d.gate_state||'unknown')+'</b><span>'+uiText('最终门禁','final gate')+'</span></div>'+
    '</div>'+
    '<div class="benchmark-grid"><section><div class="benchmark-head"><h2>'+uiText('顶级站横向矩阵','Top-Site Parity Matrix')+'</h2>'+fsdSource('benchmarked')+'</div>'+
      '<div class="benchmark-dims">'+((d.dimensions||[]).map(benchmarkDimHtml).join(''))+'</div></section>'+
      '<aside><div class="benchmark-head"><h2>'+uiText('评分门禁','Score Gates')+'</h2>'+fsdSource('pass/fail')+'</div><div class="benchmark-gates">'+((d.gates||[]).map(benchmarkGateHtml).join(''))+'</div>'+
      '<div class="benchmark-head"><h2>'+uiText('外部来源','External Sources')+'</h2>'+fsdSource('official refs')+'</div><div class="benchmark-sources">'+((d.sources||[]).map(s=>'<button data-href="'+esc(s.url||'')+'" onclick="benchmarkOpen(this.dataset.href)">'+esc(s.label||'source')+'</button>').join(''))+'</div></aside></div>'+
    '<div class="benchmark-bottom"><section><div class="benchmark-head"><h2>'+uiText('剩余诚实差距','Remaining Honest Gaps')+'</h2>'+fsdSource('not hidden')+'</div><div class="benchmark-bets">'+((d.next_bets||[]).map(x=>'<p>'+uiEsc(x)+'</p>').join(''))+'</div></section>'+
      '<section><div class="benchmark-head"><h2>'+uiText('边界','Boundary')+'</h2>'+fsdSource('public safe')+'</div><div class="benchmark-boundary">'+uiEsc(d.honest_boundary||'')+'</div></section></div>';
}
function benchmarkDimHtml(x){
  const surfaces=(x.our_surface||[]).map(h=>'<button data-href="'+esc(h)+'" onclick="benchmarkOpen(this.dataset.href)">'+esc(h)+'</button>').join('');
  const evidence=(x.evidence||[]).map(e=>'<li>'+uiEsc(e)+'</li>').join('');
  return '<article class="'+fsdClass(x.state||'pass')+'">'+
    '<div class="benchmark-card-head"><span>'+uiEsc(x.label||x.key||'')+'</span><b>'+esc(x.score??'—')+'</b></div>'+
    '<div class="benchmark-ref"><button data-href="'+esc(x.external_url||'')+'" onclick="benchmarkOpen(this.dataset.href)">'+esc(x.benchmark||'benchmark')+'</button></div>'+
    '<p>'+uiEsc(x.top_trait||'')+'</p><ul>'+evidence+'</ul>'+
    '<div class="benchmark-surfaces">'+surfaces+'</div>'+
    '<em>'+uiEsc(x.gap||'')+'</em></article>';
}
function benchmarkGateHtml(g){
  return '<article class="'+fsdClass(g.state||'unknown')+'"><b>'+uiEsc(g.label||g.key||'')+'</b><span>'+uiEsc(g.state||'')+'</span><p>'+uiEsc(g.value??'')+'</p></article>';
}

/* ---- H4 材料图鉴 / Atlas ---- */
let _atlas=null, _atF={verdict:'',band:'',q:''};
const AT_BAND={nir_i_low:['#22d3ee','NIR-I 短波 <750'],nir_i:['#8b5cf6','NIR-I 750-1000'],
  nir_ii:['#ec4899','NIR-II 1000-1350'],nir_ii_long:['#f43f5e','NIR-II 长波 >1350'],unknown:['#94a3b8','未估计']};
async function loadAtlas(){
  if(_atlas){ atlasRender(); }
  try{
    _atlas=await fetch('/api/atlas',{cache:'no-store'}).then(r=>r.json());
    atlasRender();
  }catch(e){ const g=document.getElementById('atGrid'); if(g) g.innerHTML='<div class="oe-empty">图鉴加载失败</div>'; }
}
function atlasRender(){
  if(!_atlas) return;
  const s=_atlas.summary||{}, items=_atlas.items||[];
  // 统计条 + λ 直方图
  const st=document.getElementById('atStats');
  if(st){
    const vd=s.verdict||{}; const go=vd.GO||0, rev=vd.REVISE||0;
    // λ 直方图 (10nm 桶)
    const lams=items.map(i=>i.lambda_em).filter(x=>x!=null);
    let hist='';
    if(lams.length){
      const lo=Math.floor(Math.min(...lams)/20)*20, hi=Math.ceil(Math.max(...lams)/20)*20;
      const nb=Math.max(1,Math.round((hi-lo)/20)); const buck=new Array(nb).fill(0);
      lams.forEach(l=>{ let b=Math.min(nb-1,Math.floor((l-lo)/20)); buck[b]++; });
      const mx=Math.max(...buck);
      hist='<div class="at-hist"><div class="at-hh">发射波长分布 (λ_em, 20nm/桶 · '+lams.length+' 个有效估计)</div><div class="at-bars">'+
        buck.map((c,i)=>'<div class="at-bar" title="'+(lo+i*20)+'-'+(lo+i*20+20)+'nm: '+c+'"><div class="at-bf" style="height:'+(c/mx*100)+'%"></div></div>').join('')+
        '</div><div class="at-axis"><span>'+lo+'nm</span><span>'+hi+'nm</span></div></div>';
    }
    st.innerHTML='<div class="at-cards">'+
      '<div class="at-sc"><div class="v">'+(s.total||0)+'</div><div class="l">候选总数</div></div>'+
      '<div class="at-sc go"><div class="v">'+go+'</div><div class="l">GO 可做</div></div>'+
      '<div class="at-sc rev"><div class="v">'+rev+'</div><div class="l">REVISE 待调</div></div>'+
      '<div class="at-sc"><div class="v">'+(s.lambda_min!=null?(s.lambda_min+'–'+s.lambda_max):'—')+'</div><div class="l">λ_em 范围 (nm)</div></div>'+
      '<div class="at-sc"><div class="v">'+(_atlas.source==='real'?'真机':'镜像真算')+'</div><div class="l">数据源</div></div>'+
      '</div>'+hist;
  }
  // 过滤器
  const fl=document.getElementById('atFilters');
  if(fl){
    const vd=s.verdict||{}, bd=s.band||{};
    let h='<span class="at-fl-lbl">'+uiText('判读','verdict')+'</span>';
    h+='<button class="at-chip'+(_atF.verdict===''?' on':'')+'" onclick="atFilter(\'verdict\',\'\')">全部 '+(s.total||0)+'</button>';
    Object.keys(vd).forEach(v=>{ h+='<button class="at-chip '+(v==='GO'?'go':'rev')+(_atF.verdict===v?' on':'')+'" onclick="atFilter(\'verdict\',\''+v+'\')">'+v+' '+vd[v]+'</button>'; });
    h+='<span class="at-fl-lbl" style="margin-left:10px">波段</span>';
    h+='<button class="at-chip'+(_atF.band===''?' on':'')+'" onclick="atFilter(\'band\',\'\')">全部</button>';
    Object.keys(bd).forEach(b=>{ const m=AT_BAND[b]||AT_BAND.unknown;
      h+='<button class="at-chip'+(_atF.band===b?' on':'')+'" onclick="atFilter(\'band\',\''+b+'\')" style="--bc:'+m[0]+'"><i style="background:'+m[0]+'"></i>'+m[1].split(' ')[0]+' '+bd[b]+'</button>'; });
    fl.innerHTML=h;
  }
  // 卡片网格 (过滤 + 限 120 张防爆)
  const g=document.getElementById('atGrid'); if(!g) return;
  const q=(_atF.q||'').trim().toLowerCase();
  let rows=items.filter(it=>(!_atF.verdict||it.verdict===_atF.verdict)&&(!_atF.band||it.band===_atF.band)&&
    (!q||[it.formula,it.trace,it.source,it.verdict,it.site,it.round].join(' ').toLowerCase().includes(q)));
  const shown=rows.slice(0,120);
  g.innerHTML=shown.map(it=>{
    const m=AT_BAND[it.band]||AT_BAND.unknown;
    const vcls=it.verdict==='GO'?'go':it.verdict==='REVISE'?'rev':'';
    return '<div class="at-card '+vcls+'">'+
      '<div class="at-fm">'+esc(it.formula)+'</div>'+
      '<div class="at-lam" style="--lc:'+m[0]+'">'+(it.lambda_em!=null?(it.lambda_em+' nm'):'λ 未估计')+'</div>'+
      '<div class="at-tags">'+
        '<span class="at-vd '+vcls+'">'+esc(it.verdict)+'</span>'+
        (it.site?'<span class="at-tg">'+esc(it.site)+' 位</span>':'')+
        (it.stability_pct!=null?'<span class="at-tg">稳定 '+it.stability_pct+'%</span>':'')+
      '</div>'+
      '<div class="at-src">'+esc(it.source)+(it.round?(' · '+esc(it.round)):'')+(it.converged?' · ✓收敛':'')+'</div>'+
      (it.trace?'<div class="at-tr">'+esc(it.trace)+'</div>':'')+
      '</div>';
  }).join('')||'<div class="oe-empty">无匹配候选</div>';
  const ts=document.getElementById('atTs');
  if(ts) ts.textContent=(s.total||0)+' 候选 · 显示 '+shown.length+(rows.length>120?(' / '+rows.length+' (限 120)'):'')+' · '+
    (_atlas.source==='real'?'真机':'镜像真算')+' · '+new Date().toLocaleTimeString('zh-CN');
}
function atFilter(kind,val){ _atF[kind]=(_atF[kind]===val?'':val); atlasRender(); }

/* ---- Site9 R5 Materials Explorer ---- */
let _materials=null, _mxTimer=null;
const MX={q:'', sort:'lambda_em', dir:'asc', density:'comfortable',
  cols:{formula:true,host:false,dopant:true,site:true,verdict:true,lambda_em:true,confidence_interval:true,
    band:false,method:true,source:true,trace_id:true,work_order:true,stability_pct:false,state:false,created:false}};
const MX_LABEL={formula:'化学式',host:'宿主',dopant:'掺杂',site:'位点',verdict:'判读',lambda_em:'lambda_em',
  confidence_interval:'CI90',band:'波段',method:'方法',source:'来源',trace_id:'trace_id',
  work_order:'工单',stability_pct:'稳定性',state:'状态',created:'创建时间'};
const MX_COLS=Object.keys(MX.cols);
function atlasSourceLabel(){
  const src=(_atlas&&_atlas.source)||'unknown';
  if(src==='real') return 'live';
  if(src==='mirror') return 'mirror';
  if(src==='down') return 'offline';
  return src||'unknown';
}

loadAtlas = async function(){
  if(_atlas) atlasRender();
  try{
    _atlas=await fetch('/api/atlas',{cache:'no-store'}).then(r=>r.json());
    atlasRender();
  }catch(e){
    const g=document.getElementById('atGrid');
    if(g) g.innerHTML='<div class="oe-empty">'+uiText('图鉴加载失败','Atlas failed to load')+'</div>';
  }
  if(_atF.q && !MX.q) MX.q=_atF.q;
  const inp=document.getElementById('mxQ'); if(inp && MX.q && inp.value!==MX.q) inp.value=MX.q;
  mxFetch();
};

function mxQueryString(){
  const qs=new URLSearchParams();
  qs.set('limit','500'); qs.set('sort',MX.sort); qs.set('dir',MX.dir);
  if(MX.q) qs.set('q',MX.q);
  if(_atF.verdict) qs.set('verdict',_atF.verdict);
  if(_atF.band) qs.set('band',_atF.band);
  return qs.toString();
}
async function mxFetch(){
  try{
    _materials=await fetch('/api/materials/explorer?'+mxQueryString(),{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _materials={items:[],summary:{total:0},error:String(e)};
  }
  mxRender();
}
function mxApply(){
  const q=document.getElementById('mxQ'); const s=document.getElementById('mxSort');
  MX.q=(q?q.value:MX.q).trim(); MX.sort=s?s.value:MX.sort; _atF.q=MX.q;
  atlasRender();
  clearTimeout(_mxTimer); _mxTimer=setTimeout(mxFetch,160);
}
function mxKey(e){ if(e.key==='Enter'){ clearTimeout(_mxTimer); mxApply(); mxFetch(); } }
function mxExample(q){
  MX.q=q; _atF.q=q;
  const inp=document.getElementById('mxQ'); if(inp) inp.value=q;
  atlasRender(); mxFetch(); toast('Material filter: '+q,'info');
}
function mxFlipDir(){
  MX.dir=MX.dir==='asc'?'desc':'asc';
  const b=document.getElementById('mxDir'); if(b) b.textContent=MX.dir==='asc'?uiText('升序','Asc'):uiText('降序','Desc');
  mxFetch();
}
function mxToggleDensity(){
  MX.density=MX.density==='compact'?'comfortable':'compact';
  const b=document.getElementById('mxDensity'); if(b) b.textContent=MX.density==='compact'?uiText('舒适','Comfort'):uiText('紧凑','Compact');
  mxRender();
}
function mxToggleCols(){
  const c=document.getElementById('mxCols'); if(c) c.classList.toggle('show');
}
function mxToggleCol(k){
  if(!(k in MX.cols)) return;
  MX.cols[k]=!MX.cols[k];
  mxRender();
}
function mxSort(k){
  if(MX.sort===k) MX.dir=MX.dir==='asc'?'desc':'asc';
  else { MX.sort=k; MX.dir='asc'; }
  const s=document.getElementById('mxSort'); if(s) s.value=MX.sort;
  const b=document.getElementById('mxDir');
  if(b) b.textContent=MX.dir==='asc'?uiText('升序','Asc'):uiText('降序','Desc');
  mxFetch();
}
function mxExport(fmt){
  const path=fmt==='json'?'/api/materials/export.json':'/api/materials/export.csv';
  window.location.href=path+'?'+mxQueryString();
}

/* ---- G3 AI Brain / Fly-MB explainability ---- */
let _brainExplain=null;
async function loadBrainExplain(){
  const host=document.getElementById('brainExplain');
  if(host) host.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  try{
    _brainExplain=await fetch('/api/ai_brain/explain',{cache:'no-store'}).then(r=>r.json());
  }catch(e){
    _brainExplain={error:String(e), pipeline:[], signals:[], boundaries:[], summary:{}};
  }
  brainRender();
}
function brainRender(){
  const d=_brainExplain||{}, host=document.getElementById('brainExplain');
  if(!host) return;
  const sm=d.summary||{}, llm=sm.llm||{}, bpu=sm.bpu_slots||{};
  const ts=document.getElementById('brainTs');
  if(ts) ts.textContent=uiTerm(d.source||'unknown')+' · '+(d.release||'')+' · '+new Date().toLocaleTimeString('zh-CN');
  host.innerHTML='<div class="brain-status">'+
    '<div><b>'+esc((llm.online??'-')+'/'+(llm.total??'-'))+'</b><span>'+uiText('本地 LLM','local LLM')+'</span><i>'+uiEsc(llm.state||'unknown')+'</i></div>'+
    '<div><b>'+esc((bpu.available??'-')+'/'+(bpu.total??'-'))+'</b><span>'+uiText('BPU 槽','BPU slots')+'</span><i>'+uiEsc(bpu.state||'unknown')+'</i></div>'+
    '<div><b>'+uiText('只读','read-only')+'</b><span>'+uiText('公网边界','public boundary')+'</span><i>'+uiEsc(sm.public_boundary||'safe explanation')+'</i></div>'+
    '</div>'+
    '<div class="brain-pipeline">'+(d.pipeline||[]).map((p,i)=>'<div class="brain-step">'+
      '<div class="brain-num">'+String(i+1).padStart(2,'0')+'</div><div><span>'+uiEsc(p.kind||'stage')+'</span><b>'+uiEsc(p.title||'')+'</b><p>'+uiEsc(p.detail||'')+'</p></div></div>').join('')+'</div>'+
    '<div class="brain-bottom"><div class="brain-signals">'+(d.signals||[]).map(s=>'<div><span>'+uiEsc(s.label||'')+'</span><b>'+uiEsc(s.value||'')+'</b><i>'+uiEsc(s.source||'')+'</i></div>').join('')+'</div>'+
    '<div class="brain-boundary"><b>'+uiText('诚实边界','Honest boundaries')+'</b>'+((d.boundaries||[]).map(x=>'<p>'+uiEsc(x)+'</p>').join('')||'<p>'+uiText('暂无公开边界说明。','No public boundary notes.')+'</p>')+'</div></div>';
}
function mxDisplay(row,k){
  const v=row?row[k]:null;
  if(v==null || v==='') return '-';
  if(k==='lambda_em') return v+' nm';
  if(k==='confidence_interval') return (typeof v==='number')?'+/- '+v+' nm':String(v);
  if(k==='stability_pct') return v+'%';
  return uiEsc(v);
}
function mxDetailId(row){
  return row ? String(row.id||row.trace_id||row.trace||row.formula||'').trim() : '';
}
function mxCell(row,k){
  const text=mxDisplay(row,k);
  if(k==='formula'){
    const id=mxDetailId(row);
    if(id) return '<button class="mx-link" data-mid="'+esc(id)+'" onclick="detailOpen(&quot;material&quot;,this.dataset.mid)">'+text+'</button>';
  }
  if(k==='trace_id' && row && row.trace_id){
    return '<button class="mx-link trace" data-tid="'+esc(row.trace_id)+'" onclick="detailOpen(&quot;prediction&quot;,this.dataset.tid)">'+esc(row.trace_id)+'</button>';
  }
  return text;
}
function mxRender(){
  const rows=(_materials&&_materials.items)||[];
  const sum=(_materials&&_materials.summary)||{};
  const input=document.getElementById('mxQ');
  if(input && document.activeElement!==input && input.value!==MX.q) input.value=MX.q;
  const cols=document.getElementById('mxCols');
  if(cols){
    cols.innerHTML=MX_COLS.map(k=>'<label><input type="checkbox" '+(MX.cols[k]?'checked':'')+
      ' onchange="mxToggleCol(\''+k+'\')"> '+uiEsc(MX_LABEL[k]||k)+'</label>').join('');
  }
  const sm=document.getElementById('mxSummary');
  if(sm){
    const range=sum.lambda_min!=null?(sum.lambda_min+'-'+sum.lambda_max+' nm'):uiText('lambda 待估计','lambda pending');
    sm.innerHTML='<b>'+rows.length+'</b> '+uiText('公开行','public rows')+' · '+range+' · '+uiText('来源','source')+' '+
      Object.entries(sum.source||{}).map(([k,v])=>uiEsc(k)+': '+v).join(' / ');
  }
  const schema=document.getElementById('mxSchema');
  if(schema){
    const cards=(_materials&&_materials.method_cards)||[];
    const fields=(_materials&&_materials.schema)||[];
    schema.innerHTML='<div class="mx-schema-head"><b>'+uiText('材料对象结构','Materials object schema')+'</b><span>'+uiEsc((_materials&&_materials.source_policy)||'Public source labels are preserved.')+'</span></div>'+
      '<div class="mx-schema-grid">'+cards.map(c=>'<div><b>'+uiEsc(c.title)+'</b><p>'+uiEsc(c.detail)+'</p></div>').join('')+'</div>'+
      '<div class="mx-field-row">'+fields.map(f=>'<span title="'+uiEsc(f.meaning||'')+'">'+uiEsc(MX_LABEL[f.field]||f.field)+'</span>').join('')+'</div>';
  }
  const visible=MX_COLS.filter(k=>MX.cols[k]);
  const tb=document.getElementById('mxTable');
  if(tb){
    tb.className='mx-table '+(MX.density==='compact'?'compact':'');
    tb.innerHTML='<thead><tr>'+visible.map(k=>'<th onclick="mxSort(\''+k+'\')">'+uiEsc(MX_LABEL[k]||k)+
      (MX.sort===k?(' '+(MX.dir==='asc'?uiText('升序','asc'):uiText('降序','desc'))):'')+'</th>').join('')+'</tr></thead><tbody>'+
      rows.map(r=>'<tr>'+visible.map(k=>'<td data-l="'+uiEsc(MX_LABEL[k]||k)+'">'+mxCell(r,k)+'</td>').join('')+'</tr>').join('')+
      (rows.length?'':'<tr><td colspan="'+visible.length+'">'+uiText('没有匹配的公开材料行','No matched public material rows')+'</td></tr>')+'</tbody>';
  }
  const cards=document.getElementById('mxCards');
  if(cards){
    cards.innerHTML=rows.slice(0,80).map(r=>{
      const vcls=r.verdict==='GO'?'go':r.verdict==='REVISE'?'rev':'';
      const mid=mxDetailId(r);
      return '<div class="mx-card '+vcls+'"><div><button class="mx-card-title" data-mid="'+esc(mid)+'" onclick="detailOpen(&quot;material&quot;,this.dataset.mid)">'+uiEsc(r.formula||'unknown')+'</button><span>'+uiEsc(r.method||'')+'</span></div>'+
        '<div class="mx-kv"><span>'+uiText('发射峰','lambda')+'</span><b>'+mxDisplay(r,'lambda_em')+'</b></div>'+
        '<div class="mx-kv"><span>'+uiText('判读','verdict')+'</span><b>'+uiEsc(r.verdict||'PENDING')+'</b></div>'+
        '<div class="mx-mini">'+uiEsc([r.dopant,r.site,r.source,r.work_order||r.trace_id].filter(Boolean).join(' · '))+'</div></div>';
    }).join('') || '<div class="oe-empty">'+uiText('没有匹配的公开材料行','No matched public material rows')+'</div>';
  }
}

atlasRender = function(){
  if(!_atlas) return;
  const s=_atlas.summary||{}, items=_atlas.items||[];
  const st=document.getElementById('atStats');
  if(st){
    const vd=s.verdict||{}; const go=vd.GO||0, rev=vd.REVISE||0;
    const lams=items.map(i=>i.lambda_em).filter(x=>x!=null);
    let hist='';
    if(lams.length){
      const lo=Math.floor(Math.min(...lams)/20)*20, hi=Math.ceil(Math.max(...lams)/20)*20;
      const nb=Math.max(1,Math.round((hi-lo)/20)); const buck=new Array(nb).fill(0);
      lams.forEach(l=>{ let b=Math.min(nb-1,Math.floor((l-lo)/20)); buck[b]++; });
      const mx=Math.max(...buck);
      hist='<div class="at-hist"><div class="at-hh">'+uiText('发射波长分布','Emission distribution')+' (lambda_em, 20nm/桶 · '+lams.length+' '+uiText('行','rows')+')</div><div class="at-bars">'+
        buck.map((c,i)=>'<div class="at-bar" title="'+(lo+i*20)+'-'+(lo+i*20+20)+'nm: '+c+'"><div class="at-bf" style="height:'+(c/mx*100)+'%"></div></div>').join('')+
        '</div><div class="at-axis"><span>'+lo+'nm</span><span>'+hi+'nm</span></div></div>';
    }
    st.innerHTML='<div class="at-cards">'+
      '<div class="at-sc"><div class="v">'+(s.total||0)+'</div><div class="l">'+uiText('候选总数','Candidates')+'</div></div>'+
      '<div class="at-sc go"><div class="v">'+go+'</div><div class="l">GO</div></div>'+
      '<div class="at-sc rev"><div class="v">'+rev+'</div><div class="l">REVISE</div></div>'+
      '<div class="at-sc"><div class="v">'+(s.lambda_min!=null?(s.lambda_min+'-'+s.lambda_max):'—')+'</div><div class="l">'+uiText('lambda_em 范围','lambda_em range')+'</div></div>'+
      '<div class="at-sc"><div class="v">'+uiTerm(atlasSourceLabel())+'</div><div class="l">'+uiText('数据源','Data source')+'</div></div>'+
      '</div>'+hist;
  }
  const fl=document.getElementById('atFilters');
  if(fl){
    const vd=s.verdict||{}, bd=s.band||{};
    let h='<span class="at-fl-lbl">'+uiText('判读','verdict')+'</span>';
    h+='<button class="at-chip'+(_atF.verdict===''?' on':'')+'" onclick="atFilter(\'verdict\',\'\')">'+uiText('全部','All')+' '+(s.total||0)+'</button>';
    Object.keys(vd).forEach(v=>{ h+='<button class="at-chip '+(v==='GO'?'go':'rev')+(_atF.verdict===v?' on':'')+'" onclick="atFilter(\'verdict\',\''+v+'\')">'+v+' '+vd[v]+'</button>'; });
    h+='<span class="at-fl-lbl" style="margin-left:10px">'+uiText('波段','band')+'</span>';
    h+='<button class="at-chip'+(_atF.band===''?' on':'')+'" onclick="atFilter(\'band\',\'\')">'+uiText('全部','All')+'</button>';
    Object.keys(bd).forEach(b=>{ const m=AT_BAND[b]||AT_BAND.unknown;
      h+='<button class="at-chip'+(_atF.band===b?' on':'')+'" onclick="atFilter(\'band\',\''+b+'\')" style="--bc:'+m[0]+'"><i style="background:'+m[0]+'"></i>'+esc((m[1]||b).split(' ')[0])+' '+bd[b]+'</button>'; });
    fl.innerHTML=h;
  }
  const g=document.getElementById('atGrid'); if(!g) return;
  const q=(_atF.q||'').trim().toLowerCase();
  let rows=items.filter(it=>(!_atF.verdict||it.verdict===_atF.verdict)&&(!_atF.band||it.band===_atF.band)&&
    (!q||[it.formula,it.trace,it.source,it.verdict,it.site,it.round].join(' ').toLowerCase().includes(q)));
  const shown=rows.slice(0,120);
  g.innerHTML=shown.map(it=>{
    const m=AT_BAND[it.band]||AT_BAND.unknown;
    const vcls=it.verdict==='GO'?'go':it.verdict==='REVISE'?'rev':'';
    return '<div class="at-card '+vcls+'">'+
      '<div class="at-fm">'+esc(it.formula)+'</div>'+
      '<div class="at-lam" style="--lc:'+m[0]+'">'+(it.lambda_em!=null?(it.lambda_em+' nm'):uiText('lambda 待估计','lambda pending'))+'</div>'+
      '<div class="at-tags">'+
        '<span class="at-vd '+vcls+'">'+esc(it.verdict)+'</span>'+
        (it.site?'<span class="at-tg">'+esc(it.site)+'</span>':'')+
        (it.stability_pct!=null?'<span class="at-tg">'+uiText('稳定','stability')+' '+it.stability_pct+'%</span>':'')+
      '</div>'+
      '<div class="at-src">'+uiEsc(it.source)+(it.round?(' · '+esc(it.round)):'')+(it.converged?' · '+uiText('已收敛','converged'):'')+'</div>'+
      (it.trace?'<div class="at-tr">'+esc(it.trace)+'</div>':'')+
      '</div>';
  }).join('')||'<div class="oe-empty">'+uiText('无匹配候选','No matched candidate cards')+'</div>';
  const ts=document.getElementById('atTs');
  if(ts) ts.textContent=(s.total||0)+' '+uiText('候选','candidates')+' · '+uiText('显示','showing')+' '+shown.length+(rows.length>120?(' / '+rows.length+' '+uiText('(限 120)','(limited 120)')):'')+' · '+
    uiTerm(atlasSourceLabel())+' · '+new Date().toLocaleTimeString('zh-CN');
};
atFilter = function(kind,val){
  _atF[kind]=(_atF[kind]===val?'':val);
  atlasRender();
  clearTimeout(_mxTimer); _mxTimer=setTimeout(mxFetch,80);
};

function homeSearchKey(e){ if(e.key==='Enter') homeSearchGo(); }
function homeSearchExample(q){
  const inp=document.getElementById('homeSearchInput'); if(inp) inp.value=q;
  homeSearchGo();
}
function homeSearchGo(){
  const inp=document.getElementById('homeSearchInput');
  const q=(inp&&inp.value?inp.value:'').trim();
  if(!q){ toast(uiText('请输入化学式、trace_id、工单或关键词','Type a formula, trace_id, work order, or keyword'),'warn'); return; }
  const low=q.toLowerCase();
  if(/\btrace[_-]?[a-z0-9_-]{3,}\b/i.test(q)||low.includes('trace_id')){
    go('traces',{after:()=>traceFill(q)}); toast(uiText('链路搜索: ','Trace search: ')+q,'info'); return;
  }
  if(/\bwo[-_ ]?\d{2,}\b/i.test(q)||low.includes('work order')){
    go('tasks'); toast(uiText('工单搜索: ','Work order search: ')+q,'info'); return;
  }
  MX.q=q; _atF.q=q;
  go('atlas',{after:()=>mxExample(q)});
}

/* ---- Site9 R6: citable material / prediction detail ---- */
function encPath(s){ return encodeURIComponent(String(s||'')).replace(/%2F/g,'%2F'); }
function detailOpen(kind,id){
  DETAIL_KIND=kind||'material'; DETAIL_ID=String(id||'').trim(); DETAIL_TAB='structure'; DETAIL_DATA=null;
  if(DETAIL_ID){
    const path=(DETAIL_KIND==='prediction'?'/predictions/':'/materials/')+encPath(DETAIL_ID);
    try{ history.pushState({view:'detail',kind:DETAIL_KIND,id:DETAIL_ID},'',path); }catch(e){}
  }
  go('detail',{force:true});
}
async function loadDetail(){
  const title=document.getElementById('dtTitle'), sub=document.getElementById('dtSub'), body=document.getElementById('dtBody');
  if(title) title.textContent=DETAIL_KIND==='prediction'?uiText('预测详情','Prediction detail'):uiText('材料详情','Material detail');
  if(sub) sub.textContent=uiText('正在加载公开研究对象...','Loading public research object...');
  if(body) body.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>';
  if(!DETAIL_ID){ detailNotFound(uiText('缺少材料或 trace_id。','Missing material or trace id.')); return; }
  const path=(DETAIL_KIND==='prediction'?'/api/predictions/':'/api/materials/')+encPath(DETAIL_ID);
  try{
    const r=await fetch(path,{cache:'no-store'});
    const d=await r.json();
    if(!r.ok){ detailNotFound(d.message||uiText('没有匹配这个 ID 的公开对象。','No public object matched this id.')); return; }
    DETAIL_DATA=d; DETAIL_TAB='structure'; detailRender();
  }catch(e){ detailNotFound(uiText('详情 API 当前不可用。','Detail API is unavailable.')); }
}
function detailNotFound(msg){
  const title=document.getElementById('dtTitle'), sub=document.getElementById('dtSub'), badges=document.getElementById('dtBadges');
  const cite=document.getElementById('dtCite'), dl=document.getElementById('dtDownloads'), tabs=document.getElementById('dtTabs'), body=document.getElementById('dtBody');
  if(title) title.textContent=uiText('未找到公开对象','Public object not found');
  if(sub) sub.textContent=uiTerm(msg);
  if(badges) badges.innerHTML='<span class="dt-badge off">'+uiText('未找到','not_found')+'</span><span class="dt-badge">release site27-highlight-claims-20260709</span>';
  if(cite) cite.innerHTML='<b>'+uiText('查询','Lookup')+'</b><code>'+esc(DETAIL_ID||'-')+'</code><span>'+uiText('没有匹配这个 ID 的公开材料或预测对象。','No public material or prediction object matched this id.')+'</span>';
  if(dl) dl.innerHTML='';
  if(tabs) tabs.innerHTML='';
  if(body) body.innerHTML='<div class="dt-empty"><b>'+uiText('没有公开详情记录','No public detail record')+'</b><p>'+uiText('请从材料图鉴打开一行，或检查 trace_id 是否存在于公开历史中。','Try opening a row from Materials Explorer, or check whether the trace_id is present in public history.')+'</p><button onclick="go(&quot;atlas&quot;)">'+uiText('返回材料图鉴','Back to Explorer')+'</button></div>';
}
function detailSetTab(t){ DETAIL_TAB=t; detailRender(); }
function detailRender(){
  if(!DETAIL_DATA) return;
  const d=DETAIL_DATA, it=d.item||{}, cite=d.citation||{};
  const title=document.getElementById('dtTitle'), sub=document.getElementById('dtSub'), badges=document.getElementById('dtBadges');
  const c=document.getElementById('dtCite'), dl=document.getElementById('dtDownloads'), tabs=document.getElementById('dtTabs'), body=document.getElementById('dtBody');
  if(title) title.textContent=uiTerm(it.formula||'Unknown material');
  if(sub) sub.textContent=[it.host,it.dopant,it.site].filter(Boolean).map(uiTerm).join(' · ')+' · '+uiTerm(it.verdict||'PENDING')+' · '+(it.lambda_em!=null?it.lambda_em+' nm':uiText('lambda 待估计','lambda pending'));
  if(badges) badges.innerHTML=[
    ['verdict',it.verdict||'PENDING'],['source',it.source||'unknown'],['state',it.state||'unknown'],['version',d.release||'']
  ].map(x=>'<span class="dt-badge '+esc(String(x[1]).toLowerCase())+'">'+uiEsc(uiTerm(x[0])+': '+uiTerm(x[1]))+'</span>').join('');
  if(c) c.innerHTML='<b>'+uiText('引用这条预测','Cite this prediction')+'</b><code>'+esc(cite.id||it.id||'-')+'</code><span>'+uiEsc((cite.version||'')+' / '+(cite.method||'')+' / '+(cite.source||''))+'</span>';
  if(dl) dl.innerHTML=(d.downloads||[]).map(x=>x.available&&x.href?
    '<a class="dt-dl" href="'+esc(x.href)+'">'+uiEsc(x.label)+'</a>':
    '<span class="dt-dl off" title="'+uiEsc(x.reason||'Unavailable')+'">'+uiEsc(x.label)+' '+uiText('不可用','unavailable')+'</span>').join('');
  const keys=(d.tab_order&&d.tab_order.length?d.tab_order:Object.keys(d.tabs||{})).filter(k=>(d.tabs||{})[k]);
  if(tabs) tabs.innerHTML=keys.map(k=>'<button class="'+(DETAIL_TAB===k?'on':'')+'" data-dt-tab="'+esc(k)+'" onclick="detailSetTab(this.dataset.dtTab)">'+uiEsc(d.tabs[k].title||k)+'</button>').join('');
  const tab=(d.tabs||{})[DETAIL_TAB] || (d.tabs||{}).structure;
  if(body && tab){
    let extra='';
    if(DETAIL_TAB==='reasoning' && d.method_cards){
      extra+='<div class="dt-extra dt-methods">'+d.method_cards.map(c=>'<div><b>'+uiEsc(c.title)+'</b><p>'+uiEsc(c.detail)+'</p></div>').join('')+'</div>';
    }
    if(DETAIL_TAB==='repro' && d.api_examples){
      extra+='<div class="dt-extra dt-api">'+d.api_examples.map(x=>'<div><span>'+uiEsc(x.label)+'</span><code>'+esc(x.curl)+'</code></div>').join('')+'</div>';
    }
    if(DETAIL_TAB==='structure' && d.provenance){
      extra+='<div class="dt-extra dt-provenance"><div><b>'+uiText('来源溯源','Provenance')+'</b><p>'+uiEsc(d.provenance.row||'')+'</p></div><div><b>'+uiText('证据评分','Evidence score')+'</b><p>'+esc(d.provenance.evidence_score)+'/100 · '+uiEsc(d.provenance.uncertainty||'')+'</p></div></div>';
    }
    body.innerHTML='<div class="dt-grid">'+(tab.rows||[]).map(r=>'<div class="dt-row"><span>'+uiEsc(r[0])+'</span><b>'+uiEsc(r[1]==null||r[1]===''?'-':r[1])+'</b></div>').join('')+
      '</div><div class="dt-note">'+uiEsc(tab.note||'')+'</div>'+extra;
  }
}

function detailRouteFromLocation(fallback){
  let land=fallback||'home';
  try{
    const p=location.pathname.replace(/^\/+|\/+$/g,'');
    if(p){
      const parts=p.split('/');
      if(parts[0]==='materials' && parts[1]){
        DETAIL_KIND='material'; DETAIL_ID=decodeURIComponent(parts.slice(1).join('/')); DETAIL_TAB='structure'; DETAIL_DATA=null;
        return 'detail';
      }
      if(parts[0]==='predictions' && parts[1]){
        DETAIL_KIND='prediction'; DETAIL_ID=decodeURIComponent(parts.slice(1).join('/')); DETAIL_TAB='structure'; DETAIL_DATA=null;
        return 'detail';
      }
      const alias={observability:'obs', queue:'mq'};
      if(alias[p]) land=alias[p];
      if(PAGES_ALL.includes(p)) land=p;
    }
  }catch(e){}
  try{
    const sp=new URLSearchParams(location.search);
    const material=sp.get('material');
    const prediction=sp.get('prediction')||sp.get('trace')||sp.get('trace_id');
    const g=sp.get('go');
    if(material){
      DETAIL_KIND='material'; DETAIL_ID=material; DETAIL_TAB='structure'; DETAIL_DATA=null;
      return 'detail';
    }
    if(prediction){
      DETAIL_KIND='prediction'; DETAIL_ID=prediction; DETAIL_TAB='structure'; DETAIL_DATA=null;
      return 'detail';
    }
    if(g && PAGES_ALL.includes(g)) land=g;
  }catch(e){}
  return land;
}

window.addEventListener('popstate', ()=>{
  const land=detailRouteFromLocation(cur||'home');
  go(land,{force:true});
});

/* ---- H14 工程档案馆 (策展真实工程记录, 全可在代码/CLAUDE.md 核对) ---- */
let AR_TAB='adr';
const AR_ADR=[
  {n:'ADR-1', t:'PC 只做预处理, X5 跑全栈', d:'pymatgen/MACE/MatterGen/LoRA 训练全在 PC/5090, 只推文本产物 (CIF/JSON/GGUF/BPU bin) 到 X5。', why:'X5 无 GPU, ML 依赖太重; 分离后 PC 改代码自由, X5 只读文本产物不脆弱。'},
  {n:'ADR-2', t:'BPU 算子白名单', d:'轻量 BPU 模型限 Linear+ReLU+Dropout; Transformer 必须手写 opset 11 自动 decompose。', why:'Bayes-e v1.2.8 不支持 MobileNet/BN/fused ops; MobileNetV2 实测废弃改 YOLO。'},
  {n:'ADR-5', t:'BPU CMA 不释放 → subprocess per-forward', d:'Qwen3/R1-Distill 10-bin 超 CMA, 每 bin 独立子进程推理, 进程退出才真正释放显存。', why:'pyeasy_dnn del+gc.collect() 不释放 CMA, 多次 load 必 OOM; subprocess ~6s/bin 可接受。'},
  {n:'ADR-6', t:'BPU channel ≤8192 → down_proj 拆分', d:'R1-Distill intermediate 8960>8192, down_proj 拆 2×4480 sub-linear 用 SplitMLP 合成。', why:'Bayes-e 硬限, Round 9 踩到; GNN 类不受影响。'},
  {n:'ADR-7', t:'可微 TS 反向设计取代经验斜率', d:'autograd 反传过 6×6 d3 Tanabe-Sugano 矩阵 eigh, sigmoid-box 约束物理可行域。', why:'经验"-10nm/0.01Å"精度差; autograd 500 iter 4 NIR 目标 <2nm 误差。'},
  {n:'ADR-8', t:'R1 verdict 5 级硬先验规则', d:'强 PL 实测 > 价态失配 > 半径失配 > 跨 host UNKNOWN > R1 推理, 规则可推翻 BPU 虚拟推理。', why:'避免 R1 被 BPU 虚拟特征带偏; 实测胜预测。'},
  {n:'公网', t:'CF + 香港 VPS 双保险 + 镜像兜底', d:'访客→Cloudflare→VPS Caddy (active health 10s)→真机隧道 或 VPS 常驻镜像; 设备关机自动切镜像。', why:'VPS IP 443 被 GFW 污点→用户面走 CF 代理; 镜像跑真 predict_engine, 演示不依赖设备上电。'},
];
const AR_TRAP=[
  {t:'MCU GEMM 慢 = 逐 token 重读权重冲爆 D-cache', d:'AI-3 Transformer 1181→69.5ms (17×): 权重驻留批量 matmul (out 外层, 权重只读 1 次)。DWT 0xE0001004 分段剖析定位。'},
  {t:'BPU CMA 391MB 装不下 358MB 单 bin', d:'切 2 段 12+12 层 each 180MB 链式推理; Qwen2 553ms/forward, 手写 vs HF byte-level 对齐 max diff 0.0000。'},
  {t:'传感器读数精确为 0 → 先怀疑硬件', d:'ADXL345 振动死在 acRMS=0mg 一整会话, 真因模块/焊点坏 (重焊即 1112mg)。活传感器静止也有噪声, 读数精确=0 是失败默认值信号。'},
  {t:'NT35510 CASET/RASET 是 4 独立寄存器非 MIPI', d:'16-bit 模式 x0/x1/y0/y1 须 4 次 cmd+data 写 0x2A00-2A03; 按 MIPI 4-byte 只设了 x0_high → 结构化花屏。'},
  {t:'bit-bang 16-bit LCD + 30+ 杜邦线 = 信号完整性硬上限', d:'软件全做对仍 ~30% 帧花屏 cycling; 像素密集区花/稀疏区正常=物理串扰。换官方 FPC 屏 (TLI 30MHz 45.7FPS CPU 0 占用)。'},
  {t:'Caddy 探活抖动误切镜像', d:'health timeout 10s + health_fails 2 才下线; 设备端单线程 Werkzeug + frp 二跳, 探活用 subprocess curl 串行不并发 (urllib 对双跳隧道稳超时)。'},
  {t:'SQLite 双写锁', d:'historian 报警引擎须在采样事务 commit 后再开连接, 避免双写锁等待。审计链 _read_last_hash 4KB 截断 bug 致误报篡改 (38 分段≠篡改)。'},
];
const AR_ROAD=[
  {d:'Round 1-2', t:'脚手架 + 5 P0 创新', s:'3D 晶体 (pymatgen P1 预处理) + 可微 TS (λ_em MAE 6.2nm) + 1.5B SFT + GraphRAG 729 三元组 + CLIP R@1 47.9%'},
  {d:'Round 3-5', t:'双本地 LLM + 生成式发现 + 合成预测系统', s:'MatterGen 候选 + MatterSim/CHGNet 稳定性 + dashboard:8888 53 路由 + R1 5 级判决 + 持久化 SHA 链'},
  {d:'Round 6-7', t:'四硬核 + 竞争性创新', s:'MACE-MPA-0 + TS+Huang-Rhys + Self-Consistency/CoVe + Conformal CI + /bet /duel /landscape'},
  {d:'Round 8-9', t:'BPU 上 Transformer + 9 本地 LLM', s:'X5 BPU 24 层 Qwen2 真机实测 (553ms) + 5 BPU swap-load slot + 4 CPU llama-server'},
  {d:'2026-04', t:'具身脑全栈 + Round 4 BPU Sprint', s:'8 ROS2 包 + SmolVLM hybrid VLM + PP-OCRv4 全 BPU 6ms + MPPI cost MLP 880K traj/s + XFeat'},
  {d:'2026-05', t:'双臂工位 v4', s:'双 myCobot 280 + 10 stage 端到端剧本 + 4 快拆爪 + 防撞互锁 + WorkCockpit 驾驶舱'},
  {d:'2026-06', t:'公网平台 + 工业级全栈 + UI 升级轮', s:'CF+VPS 双保险 + SSO + historian/ISA-18.2 告警/ISA-88 工单/SLO + 数字孪生 + 指挥中心门户'},
];
function arTab(t){ AR_TAB=t;
  document.querySelectorAll('.ar-tab').forEach(x=>x.classList.toggle('on', x.dataset.t===t)); loadArchive(); }
function loadArchive(){
  const body=document.getElementById('arBody'), ts=document.getElementById('arTs'); if(!body) return;
  if(AR_TAB==='adr'){
    body.innerHTML='<div class="ar-grid">'+AR_ADR.map(a=>
      '<div class="ar-adr"><div class="ar-ah"><span class="ar-n">'+esc(a.n)+'</span><b>'+esc(a.t)+'</b></div>'+
      '<div class="ar-d">'+esc(a.d)+'</div><div class="ar-why"><span>理由</span>'+esc(a.why)+'</div></div>').join('')+'</div>';
    if(ts) ts.textContent=AR_ADR.length+' 条架构决策记录 (ADR)';
  } else if(AR_TAB==='trap'){
    body.innerHTML='<div class="ar-grid">'+AR_TRAP.map(a=>
      '<div class="ar-trap"><div class="ar-th">🐛 '+esc(a.t)+'</div><div class="ar-d">'+esc(a.d)+'</div></div>').join('')+'</div>';
    if(ts) ts.textContent=AR_TRAP.length+' 条硬核踩坑实录 (附修法)';
  } else {
    body.innerHTML='<div class="ar-road">'+AR_ROAD.map(r=>
      '<div class="ar-rd"><div class="ar-rdot"></div><div class="ar-rc"><div class="ar-rdate">'+esc(r.d)+'</div>'+
      '<b>'+esc(r.t)+'</b><span>'+esc(r.s)+'</span></div></div>').join('')+'</div>';
    if(ts) ts.textContent=AR_ROAD.length+' 个研发阶段 (Round 1-9 → 公网平台)';
  }
}

/* ---- H16 模型注册表 ---- */
const MD_TIER={llm:['#7c3aed','LLM'],bpu:['#06b6d4','BPU'],edge:['#d97706','边缘']};
async function loadModels(){
  try{
    const d=await fetch('/api/models',{cache:'no-store'}).then(r=>r.json());
    const s=d.summary||{};
    const st=document.getElementById('mdStats');
    if(st) st.innerHTML='<div class="at-cards">'+
      '<div class="at-sc"><div class="v">'+(s.total||0)+'</div><div class="l">模型总数</div></div>'+
      '<div class="at-sc go"><div class="v">'+(s.online||0)+'</div><div class="l">当前已加载</div></div>'+
      '<div class="at-sc"><div class="v">'+(d.tiers||[]).length+'</div><div class="l">层级</div></div>'+
      '<div class="at-sc"><div class="v">'+(d.source==='real'?'真机':'镜像')+'</div><div class="l">元数据源</div></div>'+
      '</div>';
    const body=document.getElementById('mdBody'); if(!body) return;
    body.innerHTML=(d.tiers||[]).map(t=>{
      const tc=MD_TIER[t.tier]||MD_TIER.bpu;
      const cards=(t.models||[]).map(m=>{
        const dot=m.online===true?'<span class="md-dot on" title="已加载">🟢</span>':
          m.online===false?'<span class="md-dot off" title="未加载 (设备离线)">⚪</span>':
          '<span class="md-dot na" title="策展清单">—</span>';
        return '<div class="md-card"><div class="md-top">'+dot+'<b>'+esc(m.label||m.name)+'</b></div>'+
          (m.spec?'<div class="md-spec">'+esc(m.spec)+'</div>':'')+
          (m.note?'<div class="md-note">'+esc(m.note)+'</div>':'')+'</div>';
      }).join('');
      return '<div class="md-tier"><div class="md-th"><span class="md-tag" style="--tc:'+tc[0]+'">'+tc[1]+'</span>'+
        '<b>'+esc(t.group)+'</b><span class="md-n">'+(t.models||[]).length+'</span>'+
        (t.live?'<span class="md-live">● 实时探活</span>':'<span class="md-cur">策展</span>')+'</div>'+
        (t.subtitle?'<div class="md-sub">'+esc(t.subtitle)+'</div>':'')+
        '<div class="md-grid">'+cards+'</div></div>';
    }).join('');
    const ts=document.getElementById('mdTs');
    if(ts) ts.textContent=(s.total||0)+' 模型 · '+(s.online||0)+' 已加载 · 元数据'+(d.source==='real'?'真机':'镜像')+' · '+new Date().toLocaleTimeString('zh-CN');
  }catch(e){ const b=document.getElementById('mdBody'); if(b) b.innerHTML='<div class="oe-empty">注册表加载失败</div>'; }
}

/* ---- H28 标准合规墙 (策展真实映射, 每条指向平台实现) ---- */
const STD_WALL=[
  {std:'ISA-18.2', area:'报警管理', impl:'告警生命周期: 条件成立升警 → 人工 ack 记操作人 → 恢复自动销警; crit/warn/info 分级 + 防抖 (延迟超阈 ×3 才升) + 邮件通道 (未配置诚实禁用)', go:'ops', goT:'运维 → 告警中心'},
  {std:'ISA-88', area:'批记录', impl:'批次工单生命周期: 创建即调 AI 真预测绑 trace_id/verdict → 取料 → 研磨灌装 → 烧结/表征 → 实测回填收单; 每步落 wo_log 记操作人, 批次档案可导出', go:'mq', goT:'批次工单'},
  {std:'21 CFR Part 11', area:'电子记录/审计追溯', impl:'预测记录 SHA-256 hash 链 (零篡改可实时重算 649/649) + 登录审计 logins.jsonl + RBAC (admin>member>judge) + 操作人留痕', go:'ops', goT:'运维 → 审计'},
  {std:'StatusPage SLO', area:'可用性公示', impl:'24h(48 段)/7d(56 段) 分段可用性 + 双口径诚实 (UI 可用% 含镜像兜底 ≠ 真机在线%), historian 真数据分桶', go:'ops', goT:'运维 → SLO'},
  {std:'Ignition Historian', area:'历史数据底座', impl:'SQLite WAL 单文件历史库 7 表 (状态/KPI/事件/告警/维保/工单/日志) + 30s 采样 + 分桶降采样 + SSE 实时推送 + 保留策略', go:'ops', goT:'运维 → 事件流'},
];
function loadStandards(){
  const body=document.getElementById('stBody'); if(!body) return;
  body.innerHTML='<div class="st-grid">'+STD_WALL.map(s=>
    '<div class="st-card"><div class="st-h"><span class="st-badge">'+esc(s.std)+'</span>'+
    '<span class="st-area">'+esc(s.area)+'</span><span class="st-chk">✓ 已落地</span></div>'+
    '<div class="st-impl">'+esc(s.impl)+'</div>'+
    '<button class="cp-act" onclick="go(\''+s.go+'\')">→ '+esc(s.goT)+'</button></div>').join('')+'</div>';
}

/* ---- H25 成本能效 (真实 BOM + 功耗规格) ---- */
function loadCost(){
  const body=document.getElementById('csBody'); if(!body) return;
  const HW=[
    ['AI 脑 RDK X5 (8G)','¥约 550','Bayes-e 10 TOPS · ARM64 4 核 · 7GB'],
    ['车载脑 RDK X5 (8G)','¥约 550','+ LD19/D300 雷达 + Astra Pro + 200W cam'],
    ['双 myCobot 280-Pi ×2','¥约 9000','6 DoF 桌面臂 · Pi 4B 2GB ×2'],
    ['双臂 v4 快拆套件','¥290','6 MG996R + 2 SG90 + 钕磁铁 + 3D 打印件'],
    ['STM32F407 底盘','¥约 200','4× 步进 + JY901S IMU + 电磁铁'],
  ];
  const OPS=[
    ['香港 VPS (腾讯轻量)','¥32 / 月','Caddy + frp 隧道 + 镜像 + 指挥中心'],
    ['域名 xiaomiju.xyz','¥约 20 / 年','Cloudflare 边缘代理 (免费档)'],
    ['公网总运营','≈ ¥34 / 月','双保险 (CF + VPS), 设备关机门户照常在'],
  ];
  const EFF=[
    ['BPU 边缘推理','10 TOPS @ ~5W','Qwen2 Transformer 553ms/forward · 本地零云费'],
    ['vs 云 GPU 推理','数百 W 级','需联网 + 按 token 计费 + 延迟受网络'],
    ['能效定位','边缘 ~2 TOPS/W','断网全本地可跑 (Agent 6 工具 + SOP 检索)'],
  ];
  const tbl=(rows,a,b,c)=>'<table class="cs-tbl"><tr><th>'+a+'</th><th>'+b+'</th><th>'+c+'</th></tr>'+
    rows.map(r=>'<tr><td>'+esc(r[0])+'</td><td class="cs-v">'+esc(r[1])+'</td><td class="cs-d">'+esc(r[2])+'</td></tr>').join('')+'</table>';
  body.innerHTML=
    '<div class="cs-sec"><h2>🛠 硬件采购 (一次性)</h2>'+tbl(HW,'项目','成本','规格')+'</div>'+
    '<div class="cs-sec"><h2>🌐 公网运营 (持续)</h2>'+tbl(OPS,'项目','成本','说明')+'</div>'+
    '<div class="cs-sec"><h2>⚡ 能效对比</h2>'+tbl(EFF,'方案','功耗','特征')+
    '<div class="cs-note">核心论点: 把 LLM/感知推理搬上边缘 BPU, 在 ~5W 功耗下做到 553ms/forward, 断网可跑、零云费、数据不出端 —— 这是云 GPU 方案给不了的成本与能效定位。</div></div>';
}

/* ---- H26 镜像保鲜台 ---- */
const MS_ICO={lab:'🧠',car:'🚗',arm:'🦾'};
async function loadMirrorSync(){
  const body=document.getElementById('msBody'); if(!body) return;
  let d; try{ d=await fetch('/api/mirror_sync',{cache:'no-store'}).then(r=>r.json()); }
  catch(e){ body.innerHTML='<div class="oe-empty">镜像状态获取失败</div>'; return; }
  const s=d.summary||{}; const ts=document.getElementById('msTs');
  if(ts) ts.textContent='陈旧 '+s.stale+' · 服务异常 '+s.svc_down+' · 真机在线 '+s.real_online+'/'+s.total;
  const canAdmin = window._role==='admin';
  let h='<div class="ms-bar"><div class="ms-sum">'+
    '<span class="ms-pill ok">服务正常 '+(s.total-s.svc_down)+'/'+s.total+'</span>'+
    '<span class="ms-pill '+(s.stale?'warn':'ok')+'">新鲜 '+(s.total-s.stale)+'/'+s.total+'</span>'+
    '<span class="ms-pill '+(s.real_online?'ok':'info')+'">真机在线 '+s.real_online+'/'+s.total+'</span></div>'+
    (canAdmin?'<button class="ms-sync" id="msSyncBtn" onclick="mirrorSyncNow()">🔄 立即同步检查</button>'
             :'<span class="ms-hint">同步检查需管理员</span>')+'</div>';
  h+='<div class="ms-grid">';
  (d.mirrors||[]).forEach(m=>{
    const st=m.status; const dot=st==='ok'?'ok':st==='warn'?'warn':'crit';
    h+='<div class="ms-card '+dot+'"><div class="ms-h"><span class="ms-ic">'+(MS_ICO[m.sys]||'•')+'</span>'+
      '<span class="ms-nm">'+esc(m.name)+'</span><span class="ms-dot '+dot+'"></span></div>'+
      '<div class="ms-rows">'+
        msRow('镜像服务', (m.svc_ok?'● active':'○ '+esc(m.svc)), m.svc_ok?'ok':'crit')+
        msRow('数据新鲜度', esc(m.data_age_txt)+(m.data_file?' · '+esc(m.data_file):''), m.stale?'warn':'ok')+
        msRow('真机直连', m.real_online?'● 在线 (可刷新)':'○ 离线 (镜像兜底)', m.real_online?'ok':'info')+
        msRow('上次同步检查', m.last_sync_ts?esc(m.last_sync_txt):'未记录', 'info')+
      '</div></div>';
  });
  h+='</div><div class="ms-note">'+esc(d.note||'')+'</div>';
  body.innerHTML=h;
}
function msRow(k,v,st){ return '<div class="ms-row"><span>'+k+'</span><b class="'+(st||'')+'">'+v+'</b></div>'; }
async function mirrorSyncNow(){
  const btn=document.getElementById('msSyncBtn'); if(btn){ btn.disabled=true; btn.textContent='同步检查中…'; }
  try{
    const r=await fetch('/api/mirror_sync',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(x=>x.json());
    if(r.error){ toast(r.error,'err'); }
    else { toast(r.message||'同步检查完成', r.synced&&r.synced.length?'ok':'info'); loadMirrorSync(); return; }
  }catch(e){ toast('同步检查失败','err'); }
  if(btn){ btn.disabled=false; btn.textContent='🔄 立即同步检查'; }
}

/* ---- H20 实测复现台 ---- */
let _rpItems=[], _rpBusy=false;
async function loadReproduce(){
  const body=document.getElementById('rpBody'); if(!body) return;
  let d; try{ d=await fetch('/api/reproduce',{cache:'no-store'}).then(r=>r.json()); }
  catch(e){ body.innerHTML='<div class="oe-empty">复现参考集获取失败</div>'; return; }
  _rpItems=d.items||[]; const ts=document.getElementById('rpTs');
  if(ts) ts.textContent=(d.source||'')+(d.items?(' · '+d.items.length+' 锚点'):'');
  body.innerHTML='<div class="rp-bar"><div id="rpSum"></div>'+
    '<button class="rp-all" id="rpAllBtn" onclick="reproRunAll()">▶ 全部复现</button></div>'+
    '<table class="rp-tbl"><thead><tr><th>配方</th><th>掺杂</th><th>实测 λ_em</th><th>预测 λ_em</th><th>误差</th><th>90% CI</th><th></th></tr></thead><tbody id="rpRows"></tbody></table>';
  rpRenderRows(); rpRenderSummary(d.summary||{});
}
function rpRenderSummary(s){
  const el=document.getElementById('rpSum'); if(!el) return;
  el.innerHTML='<span class="rp-pill">已复现 '+(s.done||0)+'/'+(s.total||0)+'</span>'+
    (s.mae!=null?'<span class="rp-pill ok">MAE '+s.mae+' nm</span>':'')+
    (s.done?'<span class="rp-pill ok">落 CI '+(s.within_ci||0)+'/'+s.done+'</span>':'')+
    (s.done?'<span class="rp-pill">≤50nm '+(s.within50||0)+'/'+s.done+'</span>':'');
}
function rpRenderRows(){
  const tb=document.getElementById('rpRows'); if(!tb) return; tb.innerHTML='';
  _rpItems.forEach(it=>{
    const p=it.pred; const tr=document.createElement('tr'); tr.id='rprow'+it.idx;
    const dop=esc(it.dopant_element+(it.dopant_pct?(' '+it.dopant_pct+'%'):'')+(it.dopant_site?(' @'+it.dopant_site):''));
    let pred='—', err='—', ci='—', btn='<button class="rp-run" onclick="reproRun('+it.idx+')">复现</button>';
    if(p&&p.pred_lambda!=null){
      pred='<b>'+p.pred_lambda+'</b> nm'+(p.src==='mirror'?' <span class="rp-src-tag">镜像</span>':'');
      const ae=Math.abs(p.err);
      err='<span class="'+(ae<=20?'rp-ok':ae<=50?'rp-warn':'rp-bad')+'">'+(p.err>0?'+':'')+p.err+' nm</span>';
      ci=p.within_ci?'<span class="rp-ok">✓ 命中</span>':'<span class="rp-bad">✗ 偏外</span>';
      btn='<button class="rp-run done" onclick="reproRun('+it.idx+')">重算</button>';
    }
    tr.innerHTML='<td class="rp-f">'+esc(it.formula)+'</td><td>'+dop+'</td>'+
      '<td><b>'+it.ref_lambda_em+'</b> nm</td><td>'+pred+'</td><td>'+err+'</td><td>'+ci+'</td><td>'+btn+'</td>';
    tb.appendChild(tr);
  });
}
async function reproRun(idx){
  const tr=document.getElementById('rprow'+idx); if(tr) tr.classList.add('rp-running');
  try{
    const r=await fetch('/api/reproduce',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({idx})}).then(x=>x.json());
    if(r.error){ toast(r.error,'err'); }
    else{ const it=_rpItems.find(x=>x.idx===idx); if(it) it.pred=r.pred; rpRenderRows(); rpRenderSummary(r.summary||{}); }
  }catch(e){ toast('复现失败','err'); }
  if(tr) tr.classList.remove('rp-running');
}
async function reproRunAll(){
  if(_rpBusy) return; _rpBusy=true;
  const btn=document.getElementById('rpAllBtn');
  const pending=_rpItems.filter(it=>!(it.pred&&it.pred.pred_lambda!=null));
  for(let i=0;i<pending.length;i++){
    if(btn) btn.textContent='复现中 '+(i+1)+'/'+pending.length+'…';
    await reproRun(pending[i].idx);
  }
  if(btn){ btn.textContent='▶ 全部复现'; }
  _rpBusy=false; toast('全部复现完成','ok');
}

/* ---- H17 AR 晶体查看器 (model-viewer 懒加载, 935KB 仅首次进 AR 页才拉) ---- */
let _arLoaded=false;
function loadAR(){
  const stat=document.getElementById('arStat');
  if(!_arLoaded){
    _arLoaded=true;
    const s=document.createElement('script'); s.type='module';
    s.src='/model-viewer.min.js?v=350';
    s.onload=()=>{ if(stat) stat.textContent='可拖拽 · 支持的手机可 AR'; };
    s.onerror=()=>{ if(stat) stat.textContent='3D 组件加载失败 (检查网络)'; };
    document.head.appendChild(s);
  }
  const mv=document.getElementById('arMV');
  if(mv){
    mv.addEventListener('load',()=>{ if(stat) stat.textContent='就绪 · 拖拽旋转 / 滚轮缩放'; },{once:true});
    mv.addEventListener('ar-status',e=>{ if(e.detail&&e.detail.status==='session-started'&&stat) stat.textContent='AR 进行中'; });
    if(customElements.get('model-viewer')&&stat) stat.textContent='就绪 · 拖拽旋转 / 滚轮缩放';
  }
}

/* ---- E1 可观测性中心 (手写交互式 SVG 折线图) ---- */
const OBS={metric:'latency',range:'6h',data:null,timer:null};
const OBS_RANGES=[['1h','1 小时'],['6h','6 小时'],['24h','24 小时'],['7d','7 天']];
function loadObs(){
  obsCtrls();
  obsFetchCockpit();
  obsFetch();
  obsStop();
  OBS.timer=setInterval(()=>{ if(document.getElementById('obsview').classList.contains('show')) obsFetch(); },30000);
  window.addEventListener('resize', obsRender);
}
function obsStop(){ if(OBS.timer){ clearInterval(OBS.timer); OBS.timer=null; } }
function obsCtrls(){
  const r=document.getElementById('obsRange');
  if(r) r.innerHTML=OBS_RANGES.map(([k,l])=>'<button class="obs-rb'+(k===OBS.range?' on':'')+'" onclick="obsSetRange(\''+k+'\')">'+l+'</button>').join('');
}
function obsSetRange(k){ OBS.range=k; obsCtrls(); obsFetch(); }
function obsSetMetric(k){ OBS.metric=k; obsFetch(); }
async function obsFetch(){
  try{ OBS.data=await fetch('/api/metrics?metric='+OBS.metric+'&range='+OBS.range,{cache:'no-store'}).then(r=>r.json()); obsRender(); }
  catch(e){ const c=document.getElementById('obsChart'); if(c) c.innerHTML='<div class="oe-empty">指标数据获取失败</div>'; }
}
function obsRender(){
  const d=OBS.data; if(!d) return;
  // metric tabs
  const mt=document.getElementById('obsMetrics');
  if(mt) mt.innerHTML=(d.metrics||[]).map(m=>'<button class="obs-mt'+(m.key===OBS.metric?' on':'')+'" onclick="obsSetMetric(\''+m.key+'\')">'+esc(m.title.split(' (')[0])+'</button>').join('');
  const ts=document.getElementById('obsTs'); if(ts) ts.textContent='historian · '+d.range+' · '+(d.summary.n||0)+' 点';
  // summary chips
  const s=d.summary||{}; const u=d.unit||'';
  const sm=document.getElementById('obsSum');
  if(sm) sm.innerHTML=['当前 '+obsFmt(s.last,u),'均值 '+obsFmt(s.avg,u),'峰值 '+obsFmt(s.max,u),'谷值 '+obsFmt(s.min,u)]
    .map(x=>'<span class="obs-chip">'+x+'</span>').join('');
  // legend
  const lg=document.getElementById('obsLegend');
  if(lg) lg.innerHTML=(d.series||[]).map(se=>'<span class="obs-lg"><i style="background:'+se.color+'"></i>'+esc(se.label)+'</span>').join('');
  obsDrawChart();
}
function obsFmt(v,u){ return v==null?'—':(v+(u||'')); }
function obsDrawChart(){
  const d=OBS.data, host=document.getElementById('obsChart'); if(!d||!host) return;
  const W=Math.max(320,host.clientWidth||720), H=340;
  const mL=46,mR=14,mT=14,mB=26, pw=W-mL-mR, ph=H-mT-mB;
  const t0=d.t0,t1=d.t1;
  let vmin=Infinity,vmax=-Infinity;
  (d.series||[]).forEach(s=>s.points.forEach(p=>{ if(p[1]!=null){ if(p[1]<vmin)vmin=p[1]; if(p[1]>vmax)vmax=p[1]; }}));
  if(!isFinite(vmin)){ host.innerHTML='<div class="oe-empty">该时间窗暂无采样点</div>'; return; }
  const pct=(d.unit==='%'); if(pct){ vmin=Math.min(vmin,0); vmax=Math.max(vmax,100); }
  if(vmin===vmax){ vmax=vmin+1; }
  const pad=(vmax-vmin)*0.12; vmax+=pad; vmin=Math.max(0,vmin-pad);
  const xOf=t=>mL+(t-t0)/(t1-t0)*pw, yOf=v=>mT+(1-(v-vmin)/(vmax-vmin))*ph;
  // grid + y labels
  let g=''; const ny=4;
  for(let i=0;i<=ny;i++){ const v=vmin+(vmax-vmin)*i/ny, y=yOf(v);
    g+='<line x1="'+mL+'" y1="'+y.toFixed(1)+'" x2="'+(W-mR)+'" y2="'+y.toFixed(1)+'" class="obs-grid"/>';
    g+='<text x="'+(mL-7)+'" y="'+(y+3).toFixed(1)+'" class="obs-yl">'+obsAxisNum(v)+'</text>'; }
  // x labels
  const nx=4; for(let i=0;i<=nx;i++){ const t=t0+(t1-t0)*i/nx, x=xOf(t);
    g+='<text x="'+x.toFixed(1)+'" y="'+(H-7)+'" class="obs-xl" text-anchor="'+(i===0?'start':i===nx?'end':'middle')+'">'+obsTimeLabel(t,d.range)+'</text>'; }
  // series polylines (skip null gaps)
  let paths='';
  (d.series||[]).forEach(s=>{
    let seg=[],segs=[];
    s.points.forEach(p=>{ if(p[1]==null){ if(seg.length){segs.push(seg);seg=[];} } else seg.push(xOf(p[0]).toFixed(1)+','+yOf(p[1]).toFixed(1)); });
    if(seg.length) segs.push(seg);
    segs.forEach(sg=>{ if(sg.length>1) paths+='<polyline points="'+sg.join(' ')+'" fill="none" stroke="'+s.color+'" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>';
      else if(sg.length===1){ const xy=sg[0].split(','); paths+='<circle cx="'+xy[0]+'" cy="'+xy[1]+'" r="2.4" fill="'+s.color+'"/>'; } });
  });
  host.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" width="100%" height="'+H+'" class="obs-svg" '+
    'onmousemove="obsHover(event)" onmouseleave="obsHoverOff()">'+
    '<rect x="'+mL+'" y="'+mT+'" width="'+pw+'" height="'+ph+'" class="obs-plot"/>'+g+paths+
    '<g id="obsCross"></g></svg><div class="obs-tip" id="obsTip" style="display:none"></div>';
  OBS._geo={W,H,mL,mR,mT,mB,pw,ph,t0,t1,xOf:[mL,t0,t1,pw],vmin,vmax};
}
function obsAxisNum(v){ return Math.abs(v)>=1000?(v/1000).toFixed(1)+'k':(Number.isInteger(v)?v:v.toFixed(0)); }
function obsTimeLabel(t,rng){ const d=new Date(t*1000); const p=n=>('0'+n).slice(-2);
  if(rng==='7d'||rng==='24h') return p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes());
  return p(d.getHours())+':'+p(d.getMinutes()); }
function obsHover(e){
  const d=OBS.data, g=OBS._geo; if(!d||!g) return;
  const svg=e.currentTarget, rect=svg.getBoundingClientRect();
  const sx=(e.clientX-rect.left)/rect.width*g.W;
  if(sx<g.mL||sx>g.W-g.mR){ obsHoverOff(); return; }
  const frac=(sx-g.mL)/g.pw, t=g.t0+frac*(g.t1-g.t0);
  // nearest bucket idx
  const n=(d.series[0]||{}).points.length||0; if(!n) return;
  let bi=Math.round((t-g.t0)/((g.t1-g.t0)/(n-1))); bi=Math.max(0,Math.min(n-1,bi));
  const bt=d.series[0].points[bi][0], bx=g.mL+(bt-g.t0)/(g.t1-g.t0)*g.pw;
  const yOf=v=>g.mT+(1-(v-g.vmin)/(g.vmax-g.vmin))*g.ph;
  let cross='<line x1="'+bx.toFixed(1)+'" y1="'+g.mT+'" x2="'+bx.toFixed(1)+'" y2="'+(g.mT+g.ph)+'" class="obs-cl"/>';
  let rows='';
  d.series.forEach(s=>{ const p=s.points[bi]; if(p&&p[1]!=null){
    cross+='<circle cx="'+bx.toFixed(1)+'" cy="'+yOf(p[1]).toFixed(1)+'" r="3.5" fill="'+s.color+'" stroke="#fff" stroke-width="1.5"/>';
    rows+='<div class="obs-tr"><i style="background:'+s.color+'"></i>'+esc(s.label)+'<b>'+obsFmt(p[1],d.unit)+'</b></div>'; }});
  const cg=document.getElementById('obsCross'); if(cg) cg.innerHTML=cross;
  const tip=document.getElementById('obsTip');
  if(tip){ tip.style.display='block'; tip.innerHTML='<div class="obs-tt">'+obsTimeLabel(bt,d.range)+'</div>'+rows;
    const left=Math.min(bx/g.W*rect.width+14, rect.width-150);
    tip.style.left=Math.max(4,left)+'px'; tip.style.top='14px'; }
}
function obsHoverOff(){ const c=document.getElementById('obsCross'); if(c) c.innerHTML=''; const t=document.getElementById('obsTip'); if(t) t.style.display='none'; }

/* ---- Site9 R8: Fleet / Tasks / Logs / Trace cockpit ---- */
function r8Age(s){ if(s==null) return uiText('从未','never'); if(s<60) return Math.max(0,Math.round(s))+uiText('秒前','s ago'); if(s<3600) return Math.round(s/60)+uiText('分钟前','m ago'); return Math.round(s/3600)+uiText('小时前','h ago'); }
function r8Val(v,u){ return v==null?'—':(v+(u||'')); }
function r8Cls(s){ return String(s||'unknown').toLowerCase().replace(/[^a-z0-9_-]+/g,'-'); }
function r8Src(s){ return '<span class="r8-src '+r8Cls(s)+'">'+uiEsc(s||'unknown')+'</span>'; }
function r8Spark(vals){
  vals=(vals||[]).slice(-24); if(!vals.length) return '<div class="r8-emptyline">'+uiText('暂无样本','no samples')+'</div>';
  const mx=Math.max(...vals,1), pts=vals.map((v,i)=>[i/(Math.max(vals.length-1,1))*96, 28-(Math.min(v,mx)/mx*22)]).map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
  return '<svg class="r8-spark" viewBox="0 0 100 32" aria-hidden="true"><polyline points="'+pts+'"/></svg>';
}
async function loadFleetConsole(){
  const b=document.getElementById('fleetBody'); if(b) b.innerHTML='<div class="skel" style="height:220px"></div>';
  let d; try{ d=await fetch('/api/fleet_cockpit',{cache:'no-store'}).then(r=>r.json()); }catch(e){ if(b) b.innerHTML=iEmpty(uiText('机群驾驶舱不可用','Fleet cockpit unavailable')); return; }
  const ts=document.getElementById('fleetTs'); if(ts) ts.textContent='release '+(d.release||'')+' · '+new Date((d.ts||Date.now()/1000)*1000).toLocaleTimeString('zh-CN');
  const s=d.summary||{};
  let h='<div class="r8-kpis">'+iKpi(uiText('真机','Live'),s.live||0,uiText('直连','direct'),s.live?'ok':'')+iKpi(uiText('镜像','Mirror'),s.mirror||0,uiText('兜底','fallback'),s.mirror?'warn':'')+iKpi(uiText('陈旧/离线','Stale/Offline'),(s.stale||0)+(s.offline||0),uiText('显式标注','explicit'),((s.stale||0)+(s.offline||0))?'warn':'ok')+iKpi(uiText('未知','Unknown'),s.unknown||0,uiText('非公开','not public'),s.unknown?'warn':'ok')+'</div>';
  h+='<div class="r8-grid">';
  (d.systems||[]).forEach(x=>{
    const mt=Object.entries(x.metrics||{}).map(([k,v])=>'<span class="r8-metric"><b>'+esc(k)+'</b>'+esc(v)+'</span>').join('');
    h+='<article class="r8-card">'+
      '<div class="r8-head"><div><b>'+uiEsc(x.name||x.key)+'</b><span>'+esc(x.key)+' · '+uiEsc(x.serving||'unknown')+'</span></div>'+r8Src(x.source)+'</div>'+
      '<div class="r8-row"><span>'+uiText('最后可见','Last seen')+'</span><b>'+r8Age(x.age_s)+'</b></div>'+
      '<div class="r8-row"><span>'+uiText('延迟','Latency')+'</span><b>'+r8Val(x.latency_ms,'ms')+'</b></div>'+
      '<div class="r8-row"><span>'+uiText('24h 可用性','24h availability')+'</span><b>'+r8Val(x.availability_24h,'%')+'</b></div>'+
      '<div class="r8-row"><span>'+uiText('镜像服务','Mirror service')+'</span><b>'+uiEsc(x.mirror_service||'unknown')+'</b></div>'+
      r8Spark(x.spark)+'<div class="r8-metrics">'+(mt||'<span class="r8-emptyline">'+uiText('公开指标不可用','public metrics unavailable')+'</span>')+'</div>'+
      '<p>'+uiEsc(x.next_action||'')+'</p></article>';
  });
  h+='</div>';
  if(d.events&&d.events.length) h+='<div class="r8-panel"><div class="r8-title">'+uiText('近期机群事件','Recent fleet events')+'</div>'+d.events.map(e=>'<div class="r8-li"><span>'+fmtClock(e.ts).slice(6)+'</span><b>'+uiEsc(e.sys||'platform')+'</b><em>'+uiEsc(e.severity||'info')+'</em><p>'+uiEsc(e.message||'')+'</p></div>').join('')+'</div>';
  h+='<div class="hl-foot">'+uiEsc(d.note||'')+'</div>';
  if(b) b.innerHTML=h;
}
async function loadTasksConsole(){
  const b=document.getElementById('tasksBody'); if(b) b.innerHTML='<div class="skel" style="height:220px"></div>';
  let d; try{ d=await fetch('/api/tasks_cockpit?limit=80',{cache:'no-store'}).then(r=>r.json()); }catch(e){ if(b) b.innerHTML=iEmpty(uiText('任务不可用','Tasks unavailable')); return; }
  const ts=document.getElementById('tasksTs'); if(ts) ts.textContent=uiText('任务','tasks')+' '+((d.counts||{}).total||0)+' · '+new Date((d.ts||Date.now()/1000)*1000).toLocaleTimeString('zh-CN');
  const c=d.counts||{}; let h='<div class="r8-kpis">'+iKpi(uiText('进行中','Open'),c.open||0,uiText('活动','active'),(c.open||0)?'warn':'ok')+iKpi(uiText('已完成','Done'),c.done||0,uiText('已关闭','closed'),'ok')+iKpi(uiText('已取消','Cancelled'),c.cancelled||0,uiText('已停止','stopped'),c.cancelled?'warn':'')+iKpi(uiText('总数','Total'),c.total||0,uiText('记录','records'))+'</div>';
  const rows=d.tasks||[];
  if(!rows.length){ h+=iEmpty(d.empty_state||'No tasks'); if(b) b.innerHTML=h; return; }
  h+='<div class="r8-task-list">';
  rows.forEach(t=>{
    const pct=Math.min(100,((t.stage||0)/5*100));
    const hist=(t.history||[]).slice(0,3).map(x=>'<div class="r8-hist"><span>'+fmtClock(x.ts).slice(6)+'</span><b>'+uiEsc(x.action)+'</b><i>'+uiEsc(x.detail||'')+'</i></div>').join('');
    h+='<article class="r8-task '+(t.manual_review?'review':'')+'">'+
      '<div class="r8-head"><div><b>'+esc(t.code||('WO#'+t.id))+'</b><span>'+uiEsc(t.formula||'')+' · '+uiEsc(t.dopant||'')+'</span></div><span class="r8-src '+r8Cls(t.state||'open')+'">'+uiEsc(t.state||'open')+'</span></div>'+
      '<div class="r8-taskbar"><i style="width:'+pct.toFixed(0)+'%"></i></div>'+
      '<div class="r8-row"><span>'+uiText('阶段','Stage')+'</span><b>'+uiEsc(t.stage_name||'—')+'</b></div>'+
      '<div class="r8-row"><span>Trace</span><b><code>'+esc(t.trace_id||'missing')+'</code></b></div>'+
      '<div class="r8-row"><span>'+uiText('判读 / 来源','Verdict / source')+'</span><b>'+uiEsc(t.verdict||'—')+' · '+uiEsc(t.source||'—')+'</b></div>'+
      '<div class="r8-row"><span>'+uiText('阻塞项','Blocker')+'</span><b>'+uiEsc(t.blocker||'clear')+'</b></div>'+
      '<p>'+uiText('下一步: ','Next: ')+uiEsc(t.next_action||'')+'</p>'+
      (t.trace_id?'<button class="i-mini" onclick="go(\'traces\',{after:()=>traceFill(\''+esc(t.trace_id)+'\')})">'+uiText('打开链路','Open trace')+'</button>':'')+
      '<div class="r8-history">'+(hist||'<span class="r8-emptyline">'+uiText('暂无近期历史','no recent history')+'</span>')+'</div></article>';
  });
  h+='</div>';
  if(b) b.innerHTML=h;
}
async function obsFetchCockpit(){
  let d; try{ d=await fetch('/api/observability_cockpit',{cache:'no-store'}).then(r=>r.json()); }catch(e){ return; }
  const host=document.getElementById('obsOps'); if(!host) return;
  host.innerHTML='<div class="r8-kpis">'+(d.cards||[]).map(c=>iKpi(c.label,r8Val(c.value,c.unit),c.source,c.state)).join('')+'</div><div class="hl-foot">'+esc(d.note||'')+'</div>';
}

/* ---- E2 服务依赖拓扑图 (分层 SVG 节点图 + 活动路径流动) ---- */
const TOPO={data:null,timer:null};
const TOPO_C={ok:'#10b981',info:'#2563eb',crit:'#f43f5e',idle:'#94a3b8'};
function loadTopology(){
  topoFetch(); topoStop();
  TOPO.timer=setInterval(()=>{ if(document.getElementById('topoview').classList.contains('show')) topoFetch(); },15000);
  window.addEventListener('resize', topoDraw);
}
function topoStop(){ if(TOPO.timer){ clearInterval(TOPO.timer); TOPO.timer=null; } }
async function topoFetch(){
  try{ TOPO.data=await fetch('/api/topology',{cache:'no-store'}).then(r=>r.json()); topoDraw(); }
  catch(e){ const g=document.getElementById('topoGraph'); if(g) g.innerHTML='<div class="oe-empty">拓扑获取失败</div>'; }
}
function topoDraw(){
  const d=TOPO.data, host=document.getElementById('topoGraph'); if(!d||!host) return;
  const v=document.getElementById('topoVerdict'); if(v) v.textContent=(d.summary&&d.summary.verdict)||'活体链路';
  const W=Math.max(560,host.clientWidth||880);
  const NW=152, NH=48, layers=5, colGap=(W-NW)/(layers-1);
  // group by layer
  const byL=[[],[],[],[],[]]; d.nodes.forEach(n=>byL[n.layer]&&byL[n.layer].push(n));
  const maxRows=Math.max(...byL.map(a=>a.length));
  const rowGap=78, H=Math.max(300, maxRows*rowGap+30);
  const pos={};
  byL.forEach((arr,li)=>{ const colX=li*colGap; const blockH=arr.length*rowGap;
    const y0=(H-blockH)/2+ (rowGap-NH)/2;
    arr.forEach((n,ri)=>{ pos[n.id]={x:colX,y:y0+ri*rowGap,cx:colX+NW/2,cy:y0+ri*rowGap+NH/2}; }); });
  // edges (bezier)
  let eS='';
  d.edges.forEach(e=>{ const a=pos[e.from],b=pos[e.to]; if(!a||!b) return;
    const x1=a.x+NW, y1=a.cy, x2=b.x, y2=b.cy, mx=(x1+x2)/2;
    const cls=e.active?'topo-edge act':'topo-edge';
    eS+='<path d="M'+x1+' '+y1+' C'+mx+' '+y1+','+mx+' '+y2+','+x2+' '+y2+'" class="'+cls+'"/>'; });
  // nodes
  let nS='';
  d.nodes.forEach(n=>{ const p=pos[n.id]; if(!p) return; const col=TOPO_C[n.status]||TOPO_C.idle;
    const ms=(n.ms!=null)?(n.ms+'ms'):'';
    nS+='<g class="topo-node" transform="translate('+p.x+','+p.y+')">'+
      '<rect width="'+NW+'" height="'+NH+'" rx="11" class="topo-rect" style="stroke:'+col+'"/>'+
      '<circle cx="13" cy="'+(NH/2)+'" r="5" fill="'+col+'"/>'+
      '<text x="26" y="'+(NH/2-3)+'" class="topo-nl">'+esc(n.label)+'</text>'+
      '<text x="26" y="'+(NH/2+12)+'" class="topo-ns">'+esc(n.sub||'')+(ms?(' · '+ms):'')+'</text>'+
      '</g>'; });
  host.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" width="100%" height="'+H+'" class="topo-svg">'+
    '<defs><marker id="taro" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">'+
    '<path d="M0 0 L6 3 L0 6 z" fill="#cbd5e1"/></marker></defs>'+eS+nS+'</svg>';
}

/* ---- E3 SLO 错误预算 burn-down ---- */
const SLO={win:'7d',target:99.5,scope:'lab',data:null};
const SLO_WINS=[['24h','24 小时'],['7d','7 天'],['30d','30 天']];
function loadSloBudget(){ sloCtrls(); sloFetch(); }
function sloCtrls(){
  const w=document.getElementById('sloWin');
  if(w) w.innerHTML=SLO_WINS.map(([k,l])=>'<button class="obs-rb'+(k===SLO.win?' on':'')+'" onclick="sloSetWin(\''+k+'\')">'+l+'</button>').join('');
}
function sloSetWin(k){ SLO.win=k; sloCtrls(); sloFetch(); }
function sloSetScope(k){ SLO.scope=k; sloFetch(); }
function sloSet(){ const t=parseFloat(document.getElementById('sloTarget').value); if(t>=90&&t<100){ SLO.target=t; sloFetch(); } }
async function sloFetch(){
  try{ SLO.data=await fetch('/api/slo_budget?window='+SLO.win+'&target='+SLO.target+'&scope='+SLO.scope,{cache:'no-store'}).then(r=>r.json()); sloRender(); }
  catch(e){ const c=document.getElementById('sloChart'); if(c) c.innerHTML='<div class="oe-empty">SLO 数据获取失败</div>'; }
}
function sloRender(){
  const d=SLO.data; if(!d) return;
  document.getElementById('sloTs').textContent='目标 '+d.target+'% · '+d.window+' · 真 samples';
  document.getElementById('sloNote').textContent=d.note||'';
  // scope tabs
  const st=document.getElementById('sloScope');
  if(st) st.innerHTML=(d.scopes||[]).map(s=>'<button class="obs-mt'+(s.key===d.scope?' on':'')+'" onclick="sloSetScope(\''+s.key+'\')">'+esc(s.label)+'</button>').join('');
  // cards
  const cc=document.getElementById('sloCards');
  if(cc) cc.innerHTML=(d.scopes||[]).map(s=>{
    const cls=s.status==='ok'?'ok':s.status==='warn'?'warn':'crit';
    const rem=Math.max(0,Math.min(100,s.remaining_pct==null?0:s.remaining_pct));
    return '<div class="slo-card '+cls+'"><div class="slo-cl">'+esc(s.label)+'<span class="slo-badge '+cls+'">'+(s.met?'达标':'未达标')+'</span></div>'+
      '<div class="slo-av">'+(s.availability_pct==null?'—':s.availability_pct+'%')+'<small>可用性 / 目标 '+s.target_pct+'%</small></div>'+
      '<div class="slo-bud"><div class="slo-budbar"><i style="width:'+rem+'%"></i></div>'+
      '<div class="slo-budl">剩余预算 '+(s.remaining_pct==null?'—':s.remaining_pct+'%')+' · 已耗 '+s.consumed_min+'/'+s.budget_total_min+' 分钟</div></div></div>';
  }).join('');
  // burndown chart (single series + 0 阈值)
  sloChart();
}
function sloChart(){
  const d=SLO.data, host=document.getElementById('sloChart'); if(!d||!host) return;
  const pts=d.burndown||[]; if(!pts.length){ host.innerHTML='<div class="oe-empty">无数据</div>'; return; }
  const W=Math.max(320,host.clientWidth||720),H=300,mL=44,mR=14,mT=14,mB=24,pw=W-mL-mR,ph=H-mT-mB;
  const t0=d.t0,t1=d.t1; let vmin=Math.min(0,...pts.map(p=>p[1])),vmax=Math.max(100,...pts.map(p=>p[1]));
  const pad=(vmax-vmin)*0.08; vmax+=pad; vmin-=pad;
  const xOf=t=>mL+(t-t0)/(t1-t0)*pw, yOf=v=>mT+(1-(v-vmin)/(vmax-vmin))*ph;
  let g='';for(let i=0;i<=4;i++){const v=vmin+(vmax-vmin)*i/4,y=yOf(v);
    g+='<line x1="'+mL+'" y1="'+y.toFixed(1)+'" x2="'+(W-mR)+'" y2="'+y.toFixed(1)+'" class="obs-grid"/>'+
    '<text x="'+(mL-7)+'" y="'+(y+3).toFixed(1)+'" class="obs-yl">'+Math.round(v)+'%</text>';}
  // 0% 违约线
  const yz=yOf(0); g+='<line x1="'+mL+'" y1="'+yz.toFixed(1)+'" x2="'+(W-mR)+'" y2="'+yz.toFixed(1)+'" class="slo-zero"/>'+
    '<text x="'+(W-mR)+'" y="'+(yz-4).toFixed(1)+'" class="slo-zl" text-anchor="end">违约线 0%</text>';
  for(let i=0;i<=4;i++){const t=t0+(t1-t0)*i/4,x=xOf(t);
    g+='<text x="'+x.toFixed(1)+'" y="'+(H-6)+'" class="obs-xl" text-anchor="'+(i===0?'start':i===4?'end':'middle')+'">'+obsTimeLabel(t,d.window)+'</text>';}
  const line=pts.map(p=>xOf(p[0]).toFixed(1)+','+yOf(p[1]).toFixed(1)).join(' ');
  const area=mL+','+yOf(vmin<0?0:vmin).toFixed(1)+' '+line+' '+(W-mR)+','+yOf(vmin<0?0:vmin).toFixed(1);
  const last=pts[pts.length-1][1], col=last>20?'#10b981':last>0?'#f59e0b':'#f43f5e';
  host.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" width="100%" height="'+H+'" class="obs-svg">'+g+
    '<polygon points="'+area+'" fill="'+col+'" opacity="0.08"/>'+
    '<polyline points="'+line+'" fill="none" stroke="'+col+'" stroke-width="2.4" stroke-linejoin="round"/></svg>';
}

/* ---- E4 事故复盘 ---- */
let _incOpen={};
async function loadIncidents(){
  const list=document.getElementById('incList'); if(!list) return;
  let d; try{ d=await fetch('/api/incidents',{cache:'no-store'}).then(r=>r.json()); }
  catch(e){ list.innerHTML='<div class="oe-empty">事故数据获取失败</div>'; return; }
  const s=d.summary||{}; document.getElementById('incTs').textContent=(s.total||0)+' 事故 · MTTR '+(s.mttr_txt||'—');
  const sm=document.getElementById('incSum');
  if(sm) sm.innerHTML=['共 '+(s.total||0)+' 起','进行中 '+(s.active||0),'设计内降级 '+(s.design_intended||0),'MTTR '+(s.mttr_txt||'—')]
    .map(x=>'<span class="obs-chip">'+x+'</span>').join('');
  if(!(d.incidents||[]).length){ list.innerHTML='<div class="oe-empty">🟢 近期无记录事故 — 三机链路稳定</div>'; return; }
  list.innerHTML=d.incidents.map(it=>incCard(it)).join('');
}
function incCard(it){
  const open=_incOpen[it.id]; const sevc=it.severity==='crit'?'crit':it.severity==='warn'?'warn':'info';
  const stc=it.state==='已恢复'?'ok':it.design_intended?'info':'crit';
  let h='<div class="inc-card '+sevc+(it.design_intended?' design':'')+'">'+
    '<div class="inc-h" onclick="incToggle('+it.id+')">'+
      '<span class="inc-sev '+sevc+'">'+esc(it.severity||'')+'</span>'+
      '<div class="inc-ttl"><b>'+esc(it.title)+'</b><span class="inc-meta">'+esc(it.sysname)+' · 持续 '+esc(it.duration_txt)+' · <i class="'+stc+'">'+esc(it.state)+'</i></span></div>'+
      '<span class="inc-exp">'+(open?'▾':'▸')+'</span></div>';
  if(open){
    h+='<div class="inc-body">'+
      '<div class="inc-sec"><h4>影响</h4><p>'+esc(it.impact)+'</p></div>'+
      '<div class="inc-sec"><h4>根因 (规则库自动)</h4><p>'+esc(it.cause)+'</p></div>'+
      '<div class="inc-sec"><h4>时间线</h4><div class="inc-tl">'+
        it.timeline.map(t=>'<div class="inc-te"><span class="inc-tt">'+fmtClock(t.ts)+'</span><i class="inc-td '+(t.sev||'info')+'"></i><span><b>'+esc(t.label)+'</b> '+esc(t.detail||'')+'</span></div>').join('')+
      '</div></div>'+
      (it.actions&&it.actions.length?'<div class="inc-sec"><h4>处置步骤</h4><ol class="inc-ol">'+it.actions.map(a=>'<li>'+esc(a)+'</li>').join('')+'</ol></div>':'')+
      '<div class="inc-sec"><h4>改进项</h4><ul class="inc-ul">'+it.followups.map(f=>'<li>'+esc(f)+'</li>').join('')+'</ul></div>'+
      (it.related?'<div class="inc-rel">关联: '+esc(it.related)+'</div>':'')+
    '</div>';
  }
  return h+'</div>';
}
function incToggle(id){ _incOpen[id]=!_incOpen[id]; loadIncidents(); }
function fmtClock(ts){ const d=new Date(ts*1000); const p=n=>('0'+n).slice(-2);
  return p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds()); }

/* ---- E7 全局时间机器 (historian 重建历史快照 + 回放) ---- */
const TM={tmin:0,tmax:0,at:0,data:null,playing:false,timer:null,fetchT:null};
const TM_SERV={real:['真机直连','ok'],mirror:['镜像兜底','info'],down:['离线','crit']};
async function loadTimemachine(){
  await tmFetch(null);
  if(TM.data) tmMarks();
}
function tmStop(){ TM.playing=false; if(TM.timer){ clearInterval(TM.timer); TM.timer=null; } const b=document.getElementById('tmPlay'); if(b) b.textContent='▶'; }
function _tmTsFromSlider(v){ return Math.round(TM.tmin+(TM.tmax-TM.tmin)*v/1000); }
function _tmSliderFromTs(ts){ return TM.tmax>TM.tmin? Math.round((ts-TM.tmin)/(TM.tmax-TM.tmin)*1000):1000; }
async function tmFetch(at){
  try{
    const q=(at!=null)?('?at='+at):'';
    const d=await fetch('/api/timemachine'+q,{cache:'no-store'}).then(r=>r.json());
    TM.data=d; TM.tmin=d.tmin; TM.tmax=d.tmax; TM.at=d.at;
    const rg=document.getElementById('tmRange'); if(rg&&at==null) rg.value=_tmSliderFromTs(d.at);
    tmRender();
  }catch(e){ const s=document.getElementById('tmSnap'); if(s) s.innerHTML='<div class="oe-empty">时间机器获取失败</div>'; }
}
function tmScrub(v){
  TM.at=_tmTsFromSlider(v);
  const nw=document.getElementById('tmNow'); if(nw) nw.textContent=fmtClock(TM.at);
  clearTimeout(TM.fetchT); TM.fetchT=setTimeout(()=>tmFetch(TM.at),120);  // 防抖
}
function tmGoLive(){ tmStop(); tmFetch(null); }
function tmTogglePlay(){
  if(TM.playing){ tmStop(); return; }
  TM.playing=true; document.getElementById('tmPlay').textContent='⏸';
  TM.timer=setInterval(()=>{
    const step=(TM.tmax-TM.tmin)/60;  // 60 帧走完全程
    TM.at=Math.min(TM.tmax, TM.at+step);
    const rg=document.getElementById('tmRange'); if(rg) rg.value=_tmSliderFromTs(TM.at);
    tmFetch(TM.at);
    if(TM.at>=TM.tmax) tmStop();
  }, 700);
}
function tmMarks(){
  const m=document.getElementById('tmMarks'); const d=TM.data; if(!m||!d) return;
  const span=d.tmax-d.tmin||1;
  m.innerHTML=(d.markers||[]).map(k=>{ const pct=(k.ts-d.tmin)/span*100;
    const c=k.sev==='crit'?'crit':k.sev==='warn'?'warn':k.kind==='alarm'?'warn':'info';
    return '<i class="tm-mk '+c+'" style="left:'+pct.toFixed(2)+'%" title="'+fmtClock(k.ts)+'"></i>'; }).join('');
}
function tmRender(){
  const d=TM.data; if(!d) return;
  document.getElementById('tmTs').textContent='可回溯 '+Math.round((d.tmax-d.tmin)/3600)+'h · '+(d.markers||[]).length+' 标记';
  const nw=document.getElementById('tmNow'); if(nw) nw.textContent=d.is_live?'现在 · '+fmtClock(d.at):fmtClock(d.at);
  const lv=document.getElementById('tmLive'); if(lv) lv.style.visibility=d.is_live?'hidden':'visible';
  const s=d.summary||{};
  let h='<div class="tm-head"><div class="tm-when">'+(d.is_live?'🟢 实时':'⏳ 回放')+' · <b>'+fmtClock(d.at)+'</b></div>'+
    '<div class="tm-serv">'+esc(s.serving_txt||'')+'</div></div>';
  h+='<div class="tm-cards">';
  ['lab','car','arm'].forEach(k=>{ const m=d.systems[k]||{}; const sv=m.serving; const sc=TM_SERV[sv]||['无数据','idle'];
    h+='<div class="tm-card '+sc[1]+'"><div class="tm-cn">'+esc(m.name||k)+'</div>'+
      '<div class="tm-cs '+sc[1]+'">'+sc[0]+'</div>'+
      '<div class="tm-cm">'+(m.ms!=null?(m.ms+' ms'):'—')+(m.age_s!=null&&m.age_s>60?(' · '+Math.round(m.age_s/60)+'分前采样'):'')+'</div></div>'; });
  h+='</div>';
  // KPI
  const k=d.kpi||{};
  if(k.predictions!=null||k.ci_coverage_pct!=null){
    h+='<div class="tm-kpi">'+
      (k.predictions!=null?'<span class="obs-chip">预测 '+k.predictions+' 条</span>':'')+
      (k.ci_coverage_pct!=null?'<span class="obs-chip">CI 覆盖 '+k.ci_coverage_pct+'%</span>':'')+
      (k.audit?'<span class="obs-chip">审计 '+esc(k.audit)+'</span>':'')+
      (k.llm?'<span class="obs-chip">LLM '+esc(k.llm)+'</span>':'')+'</div>';
  }
  // 当时活动告警
  h+='<div class="tm-2col"><div class="tm-box"><h4>该时刻活动告警 ('+(d.alarms||[]).length+')</h4>';
  if((d.alarms||[]).length){ h+=d.alarms.map(a=>'<div class="tm-al '+(a.severity||'info')+'"><i></i>'+esc(a.message||a.rule)+'</div>').join(''); }
  else h+='<div class="tm-empty">无活动告警</div>';
  h+='</div><div class="tm-box"><h4>周边事件</h4>';
  if((d.events||[]).length){ h+=d.events.map(e=>'<div class="tm-ev"><span>'+fmtClock(e.ts).slice(6)+'</span>'+esc(e.message||'')+'</div>').join(''); }
  else h+='<div class="tm-empty">无事件记录</div>';
  h+='</div></div><div class="tm-note">'+esc(d.note||'')+'</div>';
  document.getElementById('tmSnap').innerHTML=h;
}

/* ---- E8 统一运营总览 NOC Wall ---- */
let _nocTimer=null;
const NOC_SERV={real:['真机直连','ok','#10b981'],mirror:['镜像兜底','info','#2563eb'],down:['离线','crit','#f43f5e']};
function loadNoc(){ nocFetch(); nocStop(); _nocTimer=setInterval(()=>{ if(document.getElementById('nocview').classList.contains('show')) nocFetch(); },10000); }
function nocStop(){ if(_nocTimer){ clearInterval(_nocTimer); _nocTimer=null; } }
async function nocFetch(){
  let d; try{ d=await fetch('/api/noc',{cache:'no-store'}).then(r=>r.json()); }
  catch(e){ const b=document.getElementById('nocBody'); if(b) b.innerHTML='<div class="oe-empty">运营总览获取失败</div>'; return; }
  nocRender(d);
}
function miniSpark(arr,col){
  if(!arr||arr.length<2) return '';
  const mn=Math.min(...arr),mx=Math.max(...arr),rng=(mx-mn)||1,w=200,h=38;
  const pts=arr.map((v,i)=>(i/(arr.length-1)*w).toFixed(1)+','+(h-2-((v-mn)/rng)*(h-4)).toFixed(1));
  return '<svg viewBox="0 0 '+w+' '+h+'" class="noc-spark" preserveAspectRatio="none">'+
    '<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+col+'" stroke-width="1.6"/>'+
    '<circle cx="'+pts[pts.length-1].split(',')[0]+'" cy="'+pts[pts.length-1].split(',')[1]+'" r="2.2" fill="'+col+'"/></svg>';
}
function nocRender(d){
  const ts=document.getElementById('nocTs'); if(ts) ts.textContent='实时聚合 · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const f=d.fleet||{}, al=d.alarms||{}, k=d.kpi||{};
  let h='<div class="noc-grid">';
  // 舰队横幅
  const fv=f.down? 'crit': f.real===3?'ok':'info';
  h+='<div class="noc-banner '+fv+'" onclick="go(\'topo\')"><div class="noc-bl">舰队状态</div>'+
    '<div class="noc-bv">'+esc(f.verdict||'')+'</div>'+
    '<div class="noc-bc"><span class="ok">'+f.real+' 真机</span><span class="info">'+f.mirror+' 镜像</span><span class="crit">'+f.down+' 离线</span></div></div>';
  // KPI 条
  h+='<div class="noc-kpis">'+
    nocKpi('累计预测', k.predictions!=null?k.predictions+' 条':'—','lab')+
    nocKpi('CI 覆盖', k.ci!=null?k.ci+'%':'—','budget')+
    nocKpi('审计链', k.audit?(k.audit+(k.audit_intact?' ✓':' ✗')):'—','lab')+
    nocKpi('告警', al.total+(al.unacked?(' · '+al.unacked+' 未确认'):''),'ops',al.crit?'crit':al.warn?'warn':'ok')+
    '</div>';
  // 三机延迟面板
  h+='<div class="noc-machines">';
  (d.systems||[]).forEach(s=>{ const sc=NOC_SERV[s.serving]||['无','idle','#94a3b8'];
    h+='<div class="noc-mc '+sc[1]+'" onclick="go(\'obs\')"><div class="noc-mh"><b>'+esc(s.name)+'</b>'+
      '<span class="noc-mbadge '+sc[1]+'">'+sc[0]+'</span></div>'+
      '<div class="noc-mms">'+(s.ms!=null?s.ms:'—')+'<small>ms</small><span class="noc-mav">24h 可用 '+(s.avail24!=null?s.avail24+'%':'—')+'</span></div>'+
      miniSpark(s.spark,sc[2])+'</div>'; });
  h+='</div>';
  // 活动事故 + 事件流
  h+='<div class="noc-alarms" onclick="go(\'inc\')"><div class="noc-ph">🚨 活动事故 ('+(al.active||[]).length+')</div>';
  if((al.active||[]).length) h+=al.active.map(a=>'<div class="noc-al '+(a.severity||'info')+'"><i></i>'+esc(a.message||'')+'</div>').join('');
  else h+='<div class="tm-empty">🟢 无活动告警</div>';
  h+='</div>';
  h+='<div class="noc-events" onclick="go(\'ops\')"><div class="noc-ph">📜 实时事件流</div>';
  if((d.events||[]).length) h+=d.events.map(e=>'<div class="noc-ev"><span>'+fmtClock(e.ts).slice(6)+'</span><i class="noc-ed '+(e.sev||'info')+'"></i>'+esc(e.msg||'')+'</div>').join('');
  else h+='<div class="tm-empty">无事件</div>';
  h+='</div>';
  h+='</div>';
  document.getElementById('nocBody').innerHTML=h;
}
function nocKpi(label,val,nav,st){ return '<div class="noc-kpi'+(st?(' '+st):'')+'" onclick="go(\''+nav+'\')"><span>'+label+'</span><b>'+esc(val)+'</b></div>'; }

/* ============================================================ 工业级升级轮 I 前端 ============================================================ */
const _iTimers={};
function iStop(name){ if(_iTimers[name]){ clearInterval(_iTimers[name]); _iTimers[name]=null; } }
function iPoll(name,view,fn,ms){ iStop(name); _iTimers[name]=setInterval(()=>{ if(document.getElementById(view).classList.contains('show')) fn(); },ms); }
async function iPost(url,body,method){ const r=await fetch(url,{method:method||'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):'{}'}); let j={}; try{j=await r.json();}catch(e){} if(!r.ok){ toast((j&&j.error)||('请求失败 '+r.status)); throw new Error(r.status);} return j; }
function iModalOpen(html){ const m=document.getElementById('iModal'); document.getElementById('iModalBody').innerHTML=html; m.style.display='flex'; }
function iModalClose(){ const m=document.getElementById('iModal'); if(m) m.style.display='none'; }
function iKpi(label,val,sub,tone){ return '<div class="i-kpi'+(tone?(' '+tone):'')+'"><span>'+esc(label)+'</span><b>'+esc(val)+'</b>'+(sub?'<i>'+esc(sub)+'</i>':'')+'</div>'; }
function iEmpty(t){ return '<div class="oe-empty">'+esc(t||'暂无数据')+'</div>'; }
function iAdmin(){ return window._role==='admin'; }
function iMember(){ return window._role==='admin'||window._role==='member'; }
function iAge(s){ if(s==null) return '—'; if(s<60) return s+'s'; if(s<3600) return Math.floor(s/60)+'min'; if(s<86400) return (s/3600).toFixed(1)+'h'; return (s/86400).toFixed(1)+'d'; }

/* ---- I1 平台自监控 ---- */
function loadSelf(){ selfFetch(); iPoll('self','selfview',selfFetch,10000); }
async function selfFetch(){ let d; try{ d=await fetch('/api/self',{cache:'no-store'}).then(r=>r.json()); }catch(e){ document.getElementById('selfBody').innerHTML=iEmpty('自监控获取失败'); return; } selfRender(d); }
function selfRender(d){
  const ts=document.getElementById('selfTs'); if(ts) ts.textContent='自监控 · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const p=d.proc||{}, r=d.red||{};
  let h='<div class="i-kpis">'+
    iKpi('进程内存 RSS',(p.rss_kb/1024).toFixed(1)+' MB','/proc/self','info')+
    iKpi('线程数',p.threads,'threads')+
    iKpi('运行时长',iAge(p.uptime_s),'uptime')+
    iKpi('DB 大小',(p.db_kb/1024).toFixed(1)+' MB','historian')+'</div>';
  h+='<div class="i-sec"><div class="i-sh">RED 黄金信号 (Rate · Errors · Duration)</div><div class="i-kpis">'+
    iKpi('累计请求',r.req_total,'total')+
    iKpi('请求速率',(r.rate_per_s||0)+' /s','rate')+
    iKpi('错误率',(r.err_pct||0)+'%','5xx',r.err_pct>1?'crit':'ok')+
    iKpi('延迟 p50',(r.p50_ms!=null?r.p50_ms:'—')+' ms','median')+
    iKpi('延迟 p95',(r.p95_ms!=null?r.p95_ms:'—')+' ms','tail',r.p95_ms>800?'warn':'ok')+'</div></div>';
  // sparklines
  const se=d.series||{};
  if((se.rss_kb||[]).length>1){
    h+='<div class="i-sec"><div class="i-sh">趋势 (近 '+(se.ts||[]).length+' 采样点)</div><div class="i-spark-row">'+
      '<div class="i-spk"><span>内存 RSS (MB)</span>'+miniSpark((se.rss_kb||[]).map(v=>v/1024),'#7c3aed')+'</div>'+
      '<div class="i-spk"><span>延迟 p95 (ms)</span>'+miniSpark((se.p95||[]).map(v=>v||0),'#f59e0b')+'</div>'+
      '<div class="i-spk"><span>DB (KB)</span>'+miniSpark(se.db_kb||[],'#0d9488')+'</div></div></div>';
  }
  // routes
  h+='<div class="i-sec"><div class="i-sh">路由热度 (本进程累计)</div><div class="i-table"><table><thead><tr><th>路由</th><th>请求数</th><th>均延迟</th><th>5xx</th></tr></thead><tbody>';
  (d.routes||[]).forEach(rt=>{ h+='<tr><td><code>'+esc(rt.route)+'</code></td><td>'+rt.n+'</td><td>'+rt.avg_ms+' ms</td><td>'+(rt.err?('<b style="color:#e11d48">'+rt.err+'</b>'):'0')+'</td></tr>'; });
  h+='</tbody></table></div></div>';
  h+='<div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('selfBody').innerHTML=h;
}

/* ---- I2 OEE 生产线看板 + Andon ---- */
const OEE={win:'24h'};
function loadOee(){ const w=document.getElementById('oeeWin'); if(w) w.innerHTML=['6h','24h','7d'].map(x=>'<button class="i-tab'+(x===OEE.win?' on':'')+'" onclick="oeeSetWin(\''+x+'\')">'+x+'</button>').join(''); oeeFetch(); iPoll('oee','oeeview',oeeFetch,15000); }
function oeeSetWin(w){ OEE.win=w; loadOee(); }
async function oeeFetch(){ let d,a; try{ [d,a]=await Promise.all([fetch('/api/oee?window='+OEE.win,{cache:'no-store'}).then(r=>r.json()),fetch('/api/andon',{cache:'no-store'}).then(r=>r.json())]); }catch(e){ document.getElementById('oeeBody').innerHTML=iEmpty('OEE 获取失败'); return; } oeeRender(d,a); }
function oeeBar(label,v,tone){ const w=v==null?0:Math.max(0,Math.min(100,v)); return '<div class="oee-bar"><div class="oee-bl"><span>'+label+'</span><b>'+(v==null?'—':v+'%')+'</b></div><div class="oee-track"><div class="oee-fill '+(tone||'')+'" style="width:'+w+'%"></div></div></div>'; }
function oeeRender(d,a){
  const ts=document.getElementById('oeeTs'); if(ts) ts.textContent='OEE · '+OEE.win+' · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const o=d.overall||{};
  const oeeVal=o.oee==null?'—':o.oee+'%';
  const tone=o.oee==null?'':(o.oee>=85?'ok':o.oee>=60?'warn':'crit');
  let h='<div class="oee-top">';
  h+='<div class="oee-gauge '+tone+'"><div class="oee-g-l">综合设备效率 OEE</div><div class="oee-g-v">'+oeeVal+'</div>'+
     '<div class="oee-g-s">A × P × Q'+(o.oee!=null?(' = '+o.avail+'% × '+o.perf+'% × '+o.qual+'%'):'')+'</div></div>';
  h+='<div class="oee-bars">'+oeeBar('可用率 A (serving 在岗)',o.avail,'a')+oeeBar('性能 P (采样按额定节拍)',o.perf,'p')+oeeBar('质量 Q (审计×CI 覆盖)',o.qual,'q')+
     '<div class="oee-extra">窗口内真实产量 (预测增量): <b>'+(d.throughput!=null?d.throughput+' 条':'—')+'</b> · 采样 '+(o.samples||0)+'/'+(o.expected||0)+'</div></div>';
  h+='</div>';
  // Andon
  h+='<div class="i-sh" style="margin-top:18px">🚨 安灯板 Andon</div><div class="andon-row">';
  (a.tiles||[]).forEach(t=>{ h+='<div class="andon-tile '+t.tone+'"><div class="andon-dot"></div><div class="andon-nm">'+esc(t.name)+'</div>'+
    '<div class="andon-st">'+esc(t.label)+'</div><div class="andon-ms">'+(t.ms!=null?t.ms+'ms':'—')+'</div>'+
    (t.state==='call'?'<button class="andon-btn clr" onclick="andon(\''+t.key+'\',\'clear\')">解除呼叫</button>':'<button class="andon-btn" onclick="andon(\''+t.key+'\',\'raise\')">呼叫支援</button>')+'</div>'; });
  h+='</div>';
  // shifts
  h+='<div class="i-sec"><div class="i-sh">班次拆分</div><div class="i-table"><table><thead><tr><th>班次</th><th>可用率 A</th><th>性能 P</th><th>质量 Q</th><th>OEE</th></tr></thead><tbody>';
  (d.shifts||[]).forEach(s=>{ h+='<tr><td>'+esc(s.name)+'</td><td>'+(s.avail==null?'—':s.avail+'%')+'</td><td>'+(s.perf==null?'—':s.perf+'%')+'</td><td>'+(s.qual==null?'—':s.qual+'%')+'</td><td><b>'+(s.oee==null?'—':s.oee+'%')+'</b></td></tr>'; });
  h+='</tbody></table></div></div>';
  h+='<div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('oeeBody').innerHTML=h;
}
async function andon(sys,action){ try{ const a=await iPost('/api/andon',{sys:sys,action:action}); oeeFetch(); toast(action==='clear'?'已解除呼叫':'已呼叫支援'); }catch(e){} }

/* ---- I3 告警中心 ---- */
function loadAlert(){ alertFetch(); iPoll('alert','alertview',alertFetch,20000); }
async function alertFetch(){ let d; try{ d=await fetch('/api/alert_center',{cache:'no-store'}).then(r=>r.json()); }catch(e){ document.getElementById('alertBody').innerHTML=iEmpty('告警中心获取失败'); return; } alertRender(d); }
function alertRender(d){
  const ts=document.getElementById('alertTs'); if(ts) ts.textContent='告警中心 · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const ac=d.alarm_counts||{}; const A=iAdmin();
  let h='<div class="i-kpis">'+
    iKpi('活动告警',ac.total||0,(ac.unacked||0)+' 未确认',ac.crit?'crit':ac.warn?'warn':'ok')+
    iKpi('规则数',(d.rules||[]).length,'rules')+
    iKpi('活动静默',(d.active_silences||[]).length,'silence')+
    iKpi('当前值班',d.active_oncall?d.active_oncall.name:'未排班','oncall')+'</div>';
  // channels
  h+='<div class="i-sec"><div class="i-sh">通知渠道</div><div class="ch-row">';
  (d.channels||[]).forEach(c=>{ h+='<div class="ch-pill '+(c.configured?'on':'off')+'">'+esc(c.name)+' · '+(c.configured?'已配置':'未配置')+(A?'<button class="ch-test" onclick="notifyTest(\''+c.key+'\')">测试</button>':'')+'</div>'; });
  h+='</div></div>';
  // rules
  h+='<div class="i-sec"><div class="i-sh">告警规则'+(A?' <button class="i-mini" onclick="ruleForm()">+ 新建规则</button>':'')+'</div><div class="i-table"><table><thead><tr><th>名称</th><th>指标</th><th>条件</th><th>级别</th><th>渠道</th><th>状态</th>'+(A?'<th></th>':'')+'</tr></thead><tbody>';
  (d.rules||[]).forEach(r=>{ h+='<tr><td>'+esc(r.name)+'</td><td><code>'+esc(r.metric)+'</code></td><td>'+esc(r.op)+' '+r.threshold+(r.for_n>1?(' ×'+r.for_n):'')+(r.sys?(' ['+r.sys+']'):'')+'</td><td><span class="sev '+r.severity+'">'+r.severity+'</span></td><td>'+esc(r.channel||'—')+'</td><td>'+(r.enabled?'<span class="ok-dot">启用</span>':'<span class="off-dot">停用</span>')+'</td>'+(A?'<td><button class="i-mini" onclick="ruleToggle('+r.id+','+(r.enabled?0:1)+')">'+(r.enabled?'停用':'启用')+'</button> <button class="i-mini del" onclick="ruleDel('+r.id+')">删</button></td>':'')+'</tr>'; });
  if(!(d.rules||[]).length) h+='<tr><td colspan="7">'+iEmpty('暂无规则')+'</td></tr>';
  h+='</tbody></table></div></div>';
  // silences + oncall
  h+='<div class="i-2col"><div class="i-sec"><div class="i-sh">静默 / 维护窗口'+(A?' <button class="i-mini" onclick="silenceForm()">+ 新建</button>':'')+'</div>';
  if((d.silences||[]).length){ h+='<div class="i-list">'; d.silences.forEach(s=>{ const act=s.ts_start<=d.ts&&s.ts_end>=d.ts; h+='<div class="i-li"><b>'+esc(s.scope||'all')+'</b> '+esc(s.reason||'')+' <span class="i-dim">'+fmtClock(s.ts_start).slice(6)+'→'+fmtClock(s.ts_end).slice(6)+'</span>'+(act?'<span class="ok-dot">活动</span>'+(A?' <button class="i-mini" onclick="silenceDel('+s.id+')">结束</button>':''):'')+'</div>'; }); h+='</div>'; } else h+=iEmpty('无静默窗口');
  h+='</div><div class="i-sec"><div class="i-sh">值班表'+(A?' <button class="i-mini" onclick="oncallForm()">+ 排班</button>':'')+'</div>';
  if((d.oncall||[]).length){ h+='<div class="i-list">'; d.oncall.forEach(o=>{ const act=o.ts_start<=d.ts&&o.ts_end>=d.ts; h+='<div class="i-li"><b>'+esc(o.name)+'</b> '+esc(o.contact||'')+' <span class="i-dim">'+fmtClock(o.ts_start).slice(6)+'→'+fmtClock(o.ts_end).slice(6)+'</span>'+(act?'<span class="ok-dot">在岗</span>':'')+'</div>'; }); h+='</div>'; } else h+=iEmpty('未排班');
  h+='</div></div>';
  // notifications
  h+='<div class="i-sec"><div class="i-sh">通知日志</div>';
  if((d.notifications||[]).length){ h+='<div class="i-table"><table><thead><tr><th>时间</th><th>渠道</th><th>状态</th><th>规则/详情</th></tr></thead><tbody>';
    d.notifications.forEach(n=>{ h+='<tr><td>'+fmtClock(n.ts).slice(6)+'</td><td>'+esc(n.channel)+'</td><td><span class="sev '+(n.status==='sent'?'info':n.status==='error'?'crit':'warn')+'">'+esc(n.status)+'</span></td><td class="i-dim">'+esc(n.rule||'')+' '+esc(n.detail||'')+'</td></tr>'; });
    h+='</tbody></table></div>'; } else h+=iEmpty('暂无通知 (渠道未配置或未触发)');
  h+='</div><div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('alertBody').innerHTML=h;
}
function ruleForm(){ const ms={latency_ms:'延迟ms',serving_down:'离线1/0',req_err_pct:'错误率%',rss_mb:'内存MB',audit_broken:'审计断1/0',ci_coverage:'CI覆盖%',predictions:'预测数'};
  let opt=Object.keys(ms).map(k=>'<option value="'+k+'">'+ms[k]+'</option>').join('');
  iModalOpen('<h3>新建告警规则</h3><div class="i-form">'+
    '<label>名称 <input id="rfName" placeholder="如 延迟过高"></label>'+
    '<label>指标 <select id="rfMetric">'+opt+'</select></label>'+
    '<label>系统 <select id="rfSys"><option value="">全部</option><option>lab</option><option>car</option><option>arm</option></select></label>'+
    '<label>算子 <select id="rfOp"><option>&gt;</option><option>&lt;</option><option>&gt;=</option><option>&lt;=</option><option>==</option><option>!=</option></select></label>'+
    '<label>阈值 <input id="rfThr" type="number" value="2500"></label>'+
    '<label>连续N次 <input id="rfN" type="number" value="2"></label>'+
    '<label>级别 <select id="rfSev"><option>warn</option><option>crit</option><option>info</option></select></label>'+
    '<label>渠道 <input id="rfCh" placeholder="wecom,email (逗号分隔, 可空)"></label>'+
    '</div><button class="i-go" onclick="ruleCreate()">创建规则</button>');
}
async function ruleCreate(){ const g=id=>document.getElementById(id).value;
  try{ await iPost('/api/alert_rules',{name:g('rfName'),metric:g('rfMetric'),sys:g('rfSys'),op:g('rfOp').replace('&gt;','>').replace('&lt;','<'),threshold:parseFloat(g('rfThr')),for_n:parseInt(g('rfN'))||1,severity:g('rfSev'),channels:g('rfCh').split(',').map(x=>x.trim()).filter(Boolean)}); iModalClose(); toast('规则已创建'); alertFetch(); }catch(e){} }
async function ruleToggle(id,en){ try{ await iPost('/api/alert_rules/'+id,{enabled:en},'PATCH'); alertFetch(); }catch(e){} }
async function ruleDel(id){ try{ await iPost('/api/alert_rules/'+id,null,'DELETE'); toast('已删除'); alertFetch(); }catch(e){} }
function silenceForm(){ iModalOpen('<h3>新建静默 / 维护窗口</h3><div class="i-form"><label>范围 <select id="sfScope"><option value="all">全部</option><option>lab</option><option>car</option><option>arm</option></select></label><label>时长(分钟) <input id="sfMin" type="number" value="60"></label><label>原因 <input id="sfReason" placeholder="计划维护"></label></div><button class="i-go" onclick="silenceCreate()">创建</button>'); }
async function silenceCreate(){ const g=id=>document.getElementById(id).value; try{ await iPost('/api/silences',{scope:g('sfScope'),minutes:parseInt(g('sfMin'))||60,reason:g('sfReason')}); iModalClose(); toast('静默窗口已创建'); alertFetch(); }catch(e){} }
async function silenceDel(id){ try{ await iPost('/api/silences/'+id,null,'DELETE'); alertFetch(); }catch(e){} }
function oncallForm(){ iModalOpen('<h3>排班</h3><div class="i-form"><label>姓名 <input id="ocName"></label><label>联系方式 <input id="ocContact" placeholder="电话/邮箱"></label><label>时长(小时) <input id="ocHours" type="number" value="24"></label></div><button class="i-go" onclick="oncallCreate()">排班</button>'); }
async function oncallCreate(){ const g=id=>document.getElementById(id).value; try{ await iPost('/api/oncall',{name:g('ocName'),contact:g('ocContact'),hours:parseFloat(g('ocHours'))||24}); iModalClose(); toast('已排班'); alertFetch(); }catch(e){} }
async function notifyTest(ch){ try{ const r=await iPost('/api/notify_test',{channel:ch}); toast('测试结果: '+r.status); alertFetch(); }catch(e){} }

/* ---- I4 日志检索 ---- */
const LOGS={route:'',status:'',q:'',service:'',severity:'',trace_id:''};
function loadLogs(){ const f=document.getElementById('logsFilter'); if(f) f.innerHTML=
  '<input id="lgRoute" placeholder="路由过滤 如 /api/oee" oninput="logsDeb()"><input id="lgQ" placeholder="文本搜索" oninput="logsDeb()"><input id="lgTrace" placeholder="trace_id / req_id" oninput="logsDeb()">'+
  '<select id="lgService" onchange="logsFetch()"><option value="">全部服务</option><option value="fleet">fleet</option><option value="tasks">tasks</option><option value="observability">observability</option><option value="logs">logs/traces</option><option value="research">research</option><option value="twin">twin</option><option value="api">all api</option><option value="pages">pages</option></select>'+
  '<select id="lgSeverity" onchange="logsFetch()"><option value="">全部级别</option><option value="info">info</option><option value="warning">warning</option><option value="critical">critical</option></select>'+
  '<select id="lgStatus" onchange="logsFetch()"><option value="">全部状态</option><option value="2xx">2xx</option><option value="4xx">4xx</option><option value="5xx">5xx</option></select>'+
  '<button class="i-mini" onclick="logsFetch()">刷新</button>';
  logsFetch();
}
let _lgT; function logsDeb(){ clearTimeout(_lgT); _lgT=setTimeout(logsFetch,350); }
async function logsFetch(){ const g=id=>{const e=document.getElementById(id);return e?e.value:'';};
  const u='/api/logs?limit=150&route='+encodeURIComponent(g('lgRoute'))+'&status='+g('lgStatus')+'&q='+encodeURIComponent(g('lgQ'))+
    '&service='+encodeURIComponent(g('lgService'))+'&severity='+encodeURIComponent(g('lgSeverity'))+'&trace_id='+encodeURIComponent(g('lgTrace'));
  let d; try{ d=await fetch(u,{cache:'no-store'}).then(r=>r.json()); }catch(e){ document.getElementById('logsBody').innerHTML=iEmpty('日志获取失败'); return; }
  const ts=document.getElementById('logsTs'); if(ts) ts.textContent='环形日志 · 库内 '+d.total+' 行 · 显示 '+d.n;
  let h='<div class="i-table logs-t"><table><thead><tr><th>时间</th><th>服务</th><th>方法</th><th>路由</th><th>状态</th><th>延迟</th><th>用户</th><th>IP</th><th></th></tr></thead><tbody>';
  (d.logs||[]).forEach(l=>{ const sc=l.status>=500?'crit':l.status>=400?'warn':'ok'; h+='<tr onclick="logTrace(\''+l.req_id+'\')" class="lg-row"><td>'+fmtClock(l.ts).slice(6)+'</td><td>'+r8Src(l.service)+'</td><td><span class="mlab '+l.method+'">'+l.method+'</span></td><td><code>'+esc(l.route)+'</code></td><td><span class="sev '+sc+'">'+l.status+'</span></td><td>'+l.ms+'ms</td><td>'+esc(l.usr)+'</td><td class="i-dim">'+esc(l.ip)+'</td><td><button class="i-mini" onclick="event.stopPropagation();copyTraceLink(\''+esc(l.req_id)+'\')">link</button></td></tr>'; });
  if(!(d.logs||[]).length) h+='<tr><td colspan="9">'+iEmpty('无匹配日志; 过滤条件不会暴露 token 或密钥')+'</td></tr>';
  h+='</tbody></table></div>';
  document.getElementById('logsBody').innerHTML=h;
}
async function logTrace(rid){ let d; try{ d=await fetch('/api/trace/'+rid,{cache:'no-store'}).then(r=>r.json()); }catch(e){ return; }
  let h='<h3>请求追踪 <code>'+esc(rid)+'</code></h3><div class="trace-list">';
  (d.spans||[]).forEach(s=>{ const sc=s.status>=500?'crit':s.status>=400?'warn':'ok'; h+='<div class="trace-sp"><span class="mlab '+s.method+'">'+s.method+'</span>'+r8Src(s.service)+'<code>'+esc(s.route)+'</code><span class="sev '+sc+'">'+s.status+'</span><b>'+s.ms+'ms</b><span class="i-dim">'+fmtClock(s.ts).slice(6)+' · '+esc(s.usr)+' · '+esc(s.ip||'')+'</span></div>'; });
  if(!(d.spans||[]).length) h+=iEmpty('无追踪记录');
  h+='</div><button class="i-mini" onclick="copyTraceLink(\''+esc(rid)+'\')">复制追踪链接</button><div class="hl-foot">同一 req_id 关联该请求的全部 span; IP 已隐藏。</div>';
  iModalOpen(h);
}
function copyTraceLink(rid){ const url=location.origin+'/traces?trace_id='+encodeURIComponent(rid||''); if(navigator.clipboard) navigator.clipboard.writeText(url).then(()=>toast(uiText('追踪链接已复制','Trace link copied'),'info')).catch(()=>{}); else prompt(uiText('追踪链接','Trace link'),url); }
let _trT; function traceDeb(){ clearTimeout(_trT); _trT=setTimeout(loadTracesConsole,300); }
function traceFill(q){ const i=document.getElementById('traceQ'); if(i) i.value=q||''; loadTracesConsole(); }
async function loadTracesConsole(){
  const b=document.getElementById('tracesBody'); if(b) b.innerHTML='<div class="skel" style="height:220px"></div>';
  const q=(document.getElementById('traceQ')||{}).value||new URLSearchParams(location.search).get('trace_id')||'';
  let d; try{ d=await fetch('/api/traces?limit=80&q='+encodeURIComponent(q),{cache:'no-store'}).then(r=>r.json()); }catch(e){ if(b) b.innerHTML=iEmpty(uiText('Trace API 不可用','Trace API unavailable')); return; }
  const ts=document.getElementById('tracesTs'); if(ts) ts.textContent=(d.items||[]).length+' '+uiText('条追踪','traces')+' · release '+(d.release||'');
  const rows=d.items||[]; if(!rows.length){ if(b) b.innerHTML=iEmpty(uiTerm(d.empty_state||'No traces')); return; }
  let h='<div class="r8-traces">';
  rows.forEach(t=>{
    h+='<article class="r8-trace"><div class="r8-head"><div><b><code>'+esc(t.trace_id)+'</code></b><span>'+uiEsc(t.kind)+' · '+uiEsc(t.formula||t.work_order||'request')+'</span></div>'+r8Src(t.source)+'</div>'+
      '<div class="r8-row"><span>'+uiText('工单','Work order')+'</span><b>'+uiEsc(t.work_order||'not linked')+'</b></div>'+
      '<div class="r8-row"><span>'+uiText('判读 / 状态','Verdict/state')+'</span><b>'+uiEsc(t.verdict||'—')+' · '+uiEsc(t.state||'—')+'</b></div>'+
      traceWaterfall(t.waterfall||[])+'<button class="i-mini" onclick="copyTraceLink(\''+esc(t.trace_id)+'\')">'+uiText('复制分享链接','copy share link')+'</button></article>';
  });
  h+='</div><div class="hl-foot">'+uiEsc(d.note||'')+'</div>';
  if(b) b.innerHTML=h;
}
function traceWaterfall(stages){
  if(!stages.length) return '<div class="r8-emptyline">'+uiText('暂无公开瀑布','no public waterfall')+'</div>';
  return '<div class="r8-water">'+stages.map(s=>'<div class="r8-step '+r8Cls(s.status)+'"><span>'+uiEsc(s.label||s.key)+'</span><b>'+r8Val(s.duration_ms,'ms')+'</b><em>'+uiEsc(s.source||'')+'</em>'+(s.failure_reason?'<p>'+uiEsc(s.failure_reason)+'</p>':'')+'</div>').join('')+'</div>';
}

/* ---- I5 质量中心 QMS ---- */
function loadQms(){ qmsFetch(); }
async function qmsFetch(){ let d; try{ d=await fetch('/api/qms',{cache:'no-store'}).then(r=>r.json()); }catch(e){ document.getElementById('qmsBody').innerHTML=iEmpty('质量中心获取失败'); return; } qmsRender(d); }
function qmsRender(d){
  const ts=document.getElementById('qmsTs'); if(ts) ts.textContent='QMS · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const c=d.counts||{}; const M=iMember();
  let h='<div class="i-kpis">'+iKpi('开放 NCR',(c.ncr&&c.ncr.open)||0,'共 '+((c.ncr&&c.ncr.total)||0),(c.ncr&&c.ncr.open)?'warn':'ok')+
    iKpi('进行 CAPA',(c.capa&&c.capa.open)||0,'共 '+((c.capa&&c.capa.total)||0))+
    iKpi('批次档案',(d.batches||[]).length,'工单')+'</div>';
  // tools: COA + genealogy
  h+='<div class="i-sec"><div class="i-sh">合格证 COA / 批次族谱 (输入工单号 如 WO-... 或 ID)</div><div class="qms-tools"><input id="qmsBatch" placeholder="工单 code 或 id"><button class="i-go" onclick="coaShow()">生成 COA</button><button class="i-go alt" onclick="genShow()">查族谱</button>'+(M?' <button class="i-go alt" onclick="ncrForm()">+ 开 NCR</button>':'')+'</div>';
  if((d.batches||[]).length) h+='<div class="qms-batches">'+d.batches.slice(0,12).map(b=>'<button class="qms-chip" onclick="document.getElementById(\'qmsBatch\').value=\''+esc(b.code)+'\'">'+esc(b.code)+' <i>'+esc(b.verdict||'—')+'</i></button>').join('')+'</div>';
  h+='</div>';
  // NCR list
  h+='<div class="i-sec"><div class="i-sh">不合格报告 NCR</div>';
  if((d.ncr||[]).length){ h+='<div class="i-table"><table><thead><tr><th>编号</th><th>批次</th><th>缺陷</th><th>级别</th><th>状态</th><th></th></tr></thead><tbody>';
    d.ncr.forEach(n=>{ h+='<tr><td><code>'+esc(n.code)+'</code></td><td>'+esc(n.batch||'—')+'</td><td>'+esc(n.defect)+'</td><td><span class="sev '+(n.severity==='major'?'crit':'warn')+'">'+esc(n.severity)+'</span></td><td>'+esc(n.status)+'</td><td>'+(M&&n.status!=='closed'?'<button class="i-mini" onclick="capaForm('+n.id+')">立 CAPA</button>':'')+'</td></tr>'; });
    h+='</tbody></table></div>'; } else h+=iEmpty('暂无 NCR');
  h+='</div>';
  // CAPA board
  h+='<div class="i-sec"><div class="i-sh">纠正预防 CAPA</div>';
  if((d.capa||[]).length){ h+='<div class="capa-board">';
    ['open','in_progress','closed'].forEach(st=>{ const items=d.capa.filter(x=>x.status===st); h+='<div class="capa-col"><div class="capa-ch">'+({open:'待处理',in_progress:'进行中',closed:'已关闭'}[st])+' ('+items.length+')</div>';
      items.forEach(x=>{ h+='<div class="capa-card"><b>'+esc(x.action||'—')+'</b><span class="i-dim">根因: '+esc(x.root_cause||'—')+' · '+esc(x.owner||'')+'</span>'+(M&&st!=='closed'?'<div class="capa-acts">'+(st==='open'?'<button class="i-mini" onclick="capaSet('+x.id+',\'in_progress\')">开始</button>':'')+'<button class="i-mini" onclick="capaSet('+x.id+',\'closed\')">关闭闭环</button></div>':'')+'</div>'; });
      h+='</div>'; });
    h+='</div>'; } else h+=iEmpty('暂无 CAPA');
  h+='</div><div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('qmsBody').innerHTML=h;
}
function _qmsBatch(){ return (document.getElementById('qmsBatch')||{}).value||''; }
async function coaShow(){ const b=_qmsBatch(); if(!b){ toast('请输入工单号'); return; } let d; try{ d=await fetch('/api/coa/'+encodeURIComponent(b),{cache:'no-store'}).then(r=>r.json()); }catch(e){ return; } if(d.error){ toast(d.error); return; }
  const c=d.coa||{};
  let h='<div class="coa"><div class="coa-hd"><div><div class="coa-t">合格证 · Certificate of Analysis</div><div class="coa-sub">批次 '+esc(c.batch)+'</div></div><div class="coa-seal">QC</div></div>'+
    '<table class="coa-tb"><tr><td>配方</td><td>'+esc(c.formula||'—')+'</td></tr><tr><td>掺杂</td><td>'+esc(c.dopant||'—')+'</td></tr>'+
    '<tr><td>判读 verdict</td><td><b>'+esc(c.verdict||'—')+'</b></td></tr><tr><td>预测 λ_em</td><td>'+(c.lambda_em!=null?c.lambda_em+' nm':'—')+'</td></tr>'+
    '<tr><td>90% CI</td><td>'+(c.ci&&c.ci[0]!=null?(c.ci[0]+' ~ '+c.ci[1]+' nm'):'—')+'</td></tr><tr><td>实测 λ_obs</td><td>'+(c.lambda_obs!=null?c.lambda_obs+' nm':'未回填')+'</td></tr>'+
    '<tr><td>trace_id</td><td><code>'+esc(c.trace_id||'—')+'</code></td></tr><tr><td>数据源</td><td>'+esc(c.pred_source||'—')+'</td></tr>'+
    '<tr><td>证书哈希</td><td><code style="font-size:.62rem">'+esc((d.coa_hash||'').slice(0,32))+'…</code></td></tr></table>';
  h+='<div class="coa-sign"><div class="coa-sh">电子签名 (21 CFR Part-11)</div>';
  if((d.signatures||[]).length) h+=d.signatures.map(s=>'<div class="coa-sg">✓ '+esc(s.signer)+' ('+esc(s.role)+') · '+esc(s.meaning)+' · '+esc(s.reason||'')+' · '+fmtClock(s.ts).slice(6)+'</div>').join('');
  else h+='<div class="i-dim">尚无签名</div>';
  if(iMember()) h+='<button class="i-go" style="margin-top:8px" onclick="esignDo(\'coa\',\''+esc(c.batch)+'\')">电子签批 (approved)</button>';
  h+='</div><button class="i-go alt" style="margin-top:10px" onclick="window.print()">🖨 打印</button>';
  h+='<div class="hl-foot">'+esc(d.note||'')+'</div></div>';
  iModalOpen(h);
}
async function esignDo(ot,oid){ const reason=prompt('签批原因 (Part-11 要求记录变更原因):','批次放行 approved'); if(reason==null) return; try{ await iPost('/api/esign',{obj_type:ot,obj_id:oid,meaning:'approved',reason:reason}); toast('已电子签批'); coaShow(); }catch(e){} }
async function genShow(){ const b=_qmsBatch(); if(!b){ toast('请输入工单号'); return; } let d; try{ d=await fetch('/api/genealogy/'+encodeURIComponent(b),{cache:'no-store'}).then(r=>r.json()); }catch(e){ return; } if(d.error){ toast(d.error); return; }
  const tc={info:'#2563eb',blue:'#2563eb',violet:'#7c3aed',amber:'#f59e0b',emerald:'#10b981',rose:'#f43f5e',teal:'#0d9488'};
  let h='<h3>批次族谱 · '+esc(d.batch)+'</h3><div class="gen-flow">';
  (d.nodes||[]).forEach((n,i)=>{ h+='<div class="gen-node" style="border-color:'+(tc[n.tone]||'#94a3b8')+'"><span class="gen-dot" style="background:'+(tc[n.tone]||'#94a3b8')+'"></span><b>'+esc(n.label)+'</b><i>'+esc(n.sub||'')+'</i></div>'; if(i<d.nodes.length-1) h+='<span class="gen-arrow">→</span>'; });
  h+='</div>';
  if((d.timeline||[]).length){ h+='<div class="i-sh" style="margin-top:12px">留痕时间线</div><div class="i-list">'+d.timeline.map(t=>'<div class="i-li"><span class="i-dim">'+fmtClock(t.ts).slice(6)+'</span> '+esc(t.action)+' '+esc(t.detail||'')+'</div>').join('')+'</div>'; }
  iModalOpen(h);
}
function ncrForm(){ iModalOpen('<h3>开不合格报告 NCR</h3><div class="i-form"><label>批次 (工单号, 可空) <input id="ncBatch" value="'+esc(_qmsBatch())+'"></label><label>缺陷描述 <input id="ncDefect" placeholder="如 PL 强度低于规格"></label><label>级别 <select id="ncSev"><option value="minor">minor</option><option value="major">major</option></select></label></div><button class="i-go" onclick="ncrCreate()">开单</button>'); }
async function ncrCreate(){ const g=id=>document.getElementById(id).value; if(!g('ncDefect')){ toast('缺陷必填'); return; } try{ await iPost('/api/ncr',{batch:g('ncBatch'),defect:g('ncDefect'),severity:g('ncSev')}); iModalClose(); toast('NCR 已开单'); qmsFetch(); }catch(e){} }
function capaForm(nid){ iModalOpen('<h3>立 CAPA (NCR#'+nid+')</h3><div class="i-form"><label>根因 <input id="cpRoot" placeholder="根本原因分析"></label><label>纠正措施 <input id="cpAction" placeholder="纠正/预防措施"></label><label>负责人 <input id="cpOwner"></label><label>期限(天) <input id="cpDue" type="number" value="7"></label></div><button class="i-go" onclick="capaCreate('+nid+')">立项</button>'); }
async function capaCreate(nid){ const g=id=>document.getElementById(id).value; try{ await iPost('/api/ncr/'+nid+'/capa',{root_cause:g('cpRoot'),action:g('cpAction'),owner:g('cpOwner'),due_days:parseInt(g('cpDue'))||7}); iModalClose(); toast('CAPA 已立项'); qmsFetch(); }catch(e){} }
async function capaSet(cid,st){ try{ await iPost('/api/capa/'+cid,{status:st},'PATCH'); qmsFetch(); toast(st==='closed'?'CAPA 闭环':'已更新'); }catch(e){} }

/* ---- I6 维护管理 CMMS ---- */
function loadCmms(){ cmmsFetch(); }
async function cmmsFetch(){ let d; try{ d=await fetch('/api/cmms',{cache:'no-store'}).then(r=>r.json()); }catch(e){ document.getElementById('cmmsBody').innerHTML=iEmpty('维护管理获取失败'); return; } cmmsRender(d); }
function cmmsRender(d){
  const ts=document.getElementById('cmmsTs'); if(ts) ts.textContent='CMMS · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const A=iAdmin(),M=iMember();
  let h='<div class="i-kpis">'+iKpi('PM 逾期',d.pm_due||0,'preventive',d.pm_due?'warn':'ok')+iKpi('备件告急',d.spares_low||0,'low stock',d.spares_low?'crit':'ok')+iKpi('资产数',(d.assets||[]).length,'assets')+'</div>';
  // reliability cards
  h+='<div class="i-sec"><div class="i-sh">资产可靠性 (MTBF/MTTR 算自真告警历史)</div><div class="cmms-assets">';
  (d.assets||[]).forEach(a=>{ const r=a.reliability||{}; h+='<div class="cmms-card"><div class="cmms-h">'+(a.icon||'•')+' '+esc(a.name||a.key)+'</div>'+
    '<div class="cmms-rel"><span>MTBF</span><b>'+(r.mtbf_h!=null?r.mtbf_h+' h':'—')+'</b></div>'+
    '<div class="cmms-rel"><span>MTTR</span><b>'+(r.mttr_min!=null?r.mttr_min+' min':'—')+'</b></div>'+
    '<div class="cmms-rel"><span>故障/修复</span><b>'+(r.failures||0)+'/'+(r.repairs||0)+'</b></div>'+
    '<div class="i-dim" style="margin-top:4px">维保 '+(a.maint_n||0)+' 条</div></div>'; });
  h+='</div></div>';
  // PM
  h+='<div class="i-sec"><div class="i-sh">预防性维护 PM'+(A?' <button class="i-mini" onclick="pmForm()">+ 新排程</button>':'')+'</div><div class="i-table"><table><thead><tr><th>资产</th><th>任务</th><th>周期</th><th>下次到期</th><th>状态</th>'+(M?'<th></th>':'')+'</tr></thead><tbody>';
  (d.pm||[]).forEach(p=>{ const st=p.overdue?'<span class="sev crit">逾期</span>':p.due_soon?'<span class="sev warn">临期</span>':'<span class="ok-dot">正常</span>'; h+='<tr><td>'+esc(p.asset)+'</td><td>'+esc(p.task)+'</td><td>'+p.interval_days+'天</td><td>'+(p.next_due?fmtClock(p.next_due).slice(0,10):'—')+'</td><td>'+st+'</td>'+(M?'<td><button class="i-mini" onclick="pmDone('+p.id+')">完成</button></td>':'')+'</tr>'; });
  if(!(d.pm||[]).length) h+='<tr><td colspan="6">'+iEmpty('暂无 PM 排程')+'</td></tr>';
  h+='</tbody></table></div></div>';
  // spares
  h+='<div class="i-sec"><div class="i-sh">备件库存'+(A?' <button class="i-mini" onclick="spareForm()">+ 录入</button>':'')+'</div><div class="i-table"><table><thead><tr><th>备件</th><th>资产</th><th>库存</th><th>安全库存</th><th>状态</th></tr></thead><tbody>';
  (d.spares||[]).forEach(s=>{ h+='<tr'+(s.low?' class="row-low"':'')+'><td>'+esc(s.part)+'</td><td>'+esc(s.asset||'—')+'</td><td>'+s.qty+' '+esc(s.unit||'')+'</td><td>'+s.min_qty+'</td><td>'+(s.low?'<span class="sev crit">告急</span>':'<span class="ok-dot">充足</span>')+'</td></tr>'; });
  if(!(d.spares||[]).length) h+='<tr><td colspan="5">'+iEmpty('暂无备件')+'</td></tr>';
  h+='</tbody></table></div></div>';
  // maintenance history
  h+='<div class="i-sec"><div class="i-sh">维保历史</div>';
  if((d.maintenance||[]).length){ h+='<div class="i-list">'+d.maintenance.map(m=>'<div class="i-li"><span class="i-dim">'+fmtClock(m.ts).slice(6)+'</span> <b>'+esc(m.asset)+'</b> '+esc(m.note)+' <span class="i-dim">'+esc(m.author||'')+'</span></div>').join('')+'</div>'; } else h+=iEmpty('暂无维保记录');
  h+='</div><div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('cmmsBody').innerHTML=h;
}
function pmForm(){ iModalOpen('<h3>新建 PM 排程</h3><div class="i-form"><label>资产 <select id="pmAsset"><option>lab</option><option>car</option><option>arm</option></select></label><label>任务 <input id="pmTask" placeholder="例行点检"></label><label>周期(天) <input id="pmInt" type="number" value="90"></label></div><button class="i-go" onclick="pmCreate()">创建</button>'); }
async function pmCreate(){ const g=id=>document.getElementById(id).value; try{ await iPost('/api/pm',{asset:g('pmAsset'),task:g('pmTask'),interval_days:parseInt(g('pmInt'))||30}); iModalClose(); toast('PM 已创建'); cmmsFetch(); }catch(e){} }
async function pmDone(id){ try{ await iPost('/api/pm/'+id+'/done',{}); toast('PM 已完成并续期'); cmmsFetch(); }catch(e){} }
function spareForm(){ iModalOpen('<h3>录入备件</h3><div class="i-form"><label>备件名 <input id="spPart"></label><label>资产 <input id="spAsset" placeholder="arm/car/lab"></label><label>库存 <input id="spQty" type="number" value="0"></label><label>安全库存 <input id="spMin" type="number" value="1"></label><label>单位 <input id="spUnit" value="个"></label></div><button class="i-go" onclick="spareCreate()">录入</button>'); }
async function spareCreate(){ const g=id=>document.getElementById(id).value; if(!g('spPart')){ toast('备件名必填'); return; } try{ await iPost('/api/spares',{part:g('spPart'),asset:g('spAsset'),qty:parseInt(g('spQty'))||0,min_qty:parseInt(g('spMin'))||0,unit:g('spUnit')}); iModalClose(); toast('已录入'); cmmsFetch(); }catch(e){} }

/* ---- I7 安全态势 ---- */
function loadSec(){ secFetch(); }
async function secFetch(){
  let h9=null;
  const base={ts:Date.now()/1000, score:0, pass:0, total:0, fail:0, checks:[], roles:{},
    logins_recent:[], login_fails_recent:0,
    note:'Round 9 hardening is local and loaded first; live header/audit probes are filling in.'};
  try{
    h9=await fetch('/api/hardening',{cache:'no-store'}).then(r=>r.json());
    const localChecks=[
      {name:'RBAC role catalog', status:(h9.roles||[]).length>=6?'pass':'warn', detail:((h9.roles||[]).length||0)+' roles'},
      {name:'Audit coverage matrix', status:(h9.audit||[]).length>=8?'pass':'warn', detail:((h9.audit||[]).length||0)+' event classes'},
      {name:'OWASP Top 10 review', status:(h9.owasp||[]).length>=10?'pass':'warn', detail:((h9.owasp||[]).length||0)+' controls'},
      {name:'Performance checks', status:(h9.performance||[]).every(x=>x.state==='pass')?'pass':'warn', detail:(h9.performance||[]).map(x=>x.state).join(' / ')},
      {name:'Accessibility checks', status:(h9.accessibility||[]).every(x=>x.state==='pass')?'pass':'warn', detail:(h9.accessibility||[]).map(x=>x.state).join(' / ')},
      {name:'Public boundary layers', status:(h9.hardening_layers||[]).some(x=>x.state==='pass')?'pass':'warn', detail:(h9.hardening_layers||[]).map(x=>x.state).join(' / ')},
      {name:'Rollback path', status:h9.rollback?'pass':'warn', detail:h9.release||'local hardening'}
    ];
    const passLocal=localChecks.filter(x=>x.status==='pass').length;
    secRender(Object.assign({},base,{score:Math.round(passLocal/localChecks.length*100), pass:passLocal,
      total:localChecks.length, fail:localChecks.length-passLocal, checks:localChecks, hardening:h9}));
  }catch(e){}
  try{
    const d=await fetch('/api/security',{cache:'no-store'}).then(r=>r.json());
    d.hardening=h9;
    secRender(d);
  }catch(e){
    if(!h9) document.getElementById('secBody').innerHTML=iEmpty('安全态势获取失败');
  }
}
function r9State(s){ s=String(s||'unknown').toLowerCase(); return s==='pass'||s==='tracked'||s==='masked'||s==='gateway'?'ok':(s==='fail'?'crit':'warn'); }
function r9Bytes(n){ if(n==null) return '—'; if(n>1048576) return (n/1048576).toFixed(1)+' MB'; if(n>1024) return (n/1024).toFixed(0)+' KB'; return n+' B'; }
function r9MiniRows(rows, cols){
  return '<div class="r9-mini">'+(rows||[]).map(x=>'<div class="r9-mini-row">'+cols.map(c=>'<span>'+esc(x[c]||'—')+'</span>').join('')+'</div>').join('')+'</div>';
}
function secRender(d){
  const ts=document.getElementById('secTs'); if(ts) ts.textContent='安全态势 · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const tone=d.score>=90?'ok':d.score>=70?'warn':'crit';
  let h='<div class="sec-top"><div class="sec-score '+tone+'"><div class="sec-sl">安全态势评分</div><div class="sec-sv">'+d.score+'</div><div class="sec-ss">'+d.pass+'/'+d.total+' 项通过 · '+d.fail+' 待改进</div></div>';
  h+='<div class="sec-checks">';
  (d.checks||[]).forEach(c=>{ const ic=c.status==='pass'?'✓':c.status==='fail'?'✕':'•'; h+='<div class="sec-chk '+c.status+'"><span class="sec-ci">'+ic+'</span><div><b>'+esc(c.name)+'</b><i>'+esc(c.detail)+'</i></div></div>'; });
  h+='</div></div>';
  // roles
  if(d.roles&&Object.keys(d.roles).length){ h+='<div class="i-sec"><div class="i-sh">RBAC 角色分布</div><div class="ch-row">'+Object.entries(d.roles).map(([k,v])=>'<div class="ch-pill on">'+esc(k)+' × '+v+'</div>').join('')+'</div></div>'; }
  // login audit
  h+='<div class="i-sec"><div class="i-sh">登录审计 (近期, 失败 '+(d.login_fails_recent||0)+' 次)</div>';
  if((d.logins_recent||[]).length){ h+='<div class="i-list">'+d.logins_recent.map(l=>{ const ok=l.ok!==false&&l.success!==false; return '<div class="i-li"><span class="'+(ok?'ok-dot':'off-dot')+'">'+(ok?'✓':'✕')+'</span> '+esc(l.user||l.username||'?')+' <span class="i-dim">'+esc(l.ip||'')+' '+(l.ts?fmtClock(typeof l.ts==='number'?l.ts:Date.parse(l.ts)/1000).slice(6):'')+'</span></div>'; }).join('')+'</div>'; } else h+=iEmpty('无登录记录');
  h+='</div>';
  const h9=d.hardening||{};
  if(h9.release){
    h+='<div class="i-sec"><div class="i-sh">Round 9 Hardening · RBAC roles</div><div class="r9-role-grid">'+(h9.roles||[]).map(r=>
      '<article class="r9-role"><div><b>'+esc(r.label)+'</b><span>'+esc(r.ui)+'</span></div><p>'+esc(r.scope)+'</p><em>'+esc((r.public_actions||[]).join(' · '))+'</em></article>'
    ).join('')+'</div></div>';
    h+='<div class="i-sec"><div class="i-sh">Audit event coverage</div><div class="r9-audit">'+(h9.audit||[]).map(a=>
      '<div class="r9-audit-row '+r9State(a.state)+'"><b>'+esc(a.event)+'</b><span>'+esc(a.source)+'</span><strong>'+esc(a.records)+'</strong><em>'+esc(a.state)+'</em></div>'
    ).join('')+'</div></div>';
    h+='<div class="i-sec"><div class="i-sh">OWASP Top 10 review</div><div class="r9-owasp">'+(h9.owasp||[]).map(o=>
      '<div class="r9-ow '+r9State(o.state)+'"><span>'+esc(o.key)+'</span><b>'+esc(o.name)+'</b><em>'+esc(o.state)+'</em><p>'+esc(o.evidence)+'</p></div>'
    ).join('')+'</div></div>';
    h+='<div class="i-sec r9-two"><div><div class="i-sh">Performance / cache</div>'+r9MiniRows(h9.performance||[],['name','state','detail'])+'</div><div><div class="i-sh">Accessibility</div>'+r9MiniRows(h9.accessibility||[],['name','state'])+'</div></div>';
    h+='<div class="i-sec"><div class="i-sh">Site24 分层防护状态</div><div class="r9-owasp">'+(h9.hardening_layers||[]).map(o=>
      '<div class="r9-ow '+r9State(o.state)+'"><span>'+esc(o.layer)+'</span><b>'+esc(o.name)+'</b><em>'+esc(o.state)+'</em><p>'+esc(o.detail)+'</p></div>'
    ).join('')+'</div></div>';
    h+='<div class="i-sec"><div class="i-sh">Cloudflare WAF / 限速配置清单</div><div class="r9-audit">'+(h9.cloudflare_manual||[]).map(a=>
      '<div class="r9-audit-row '+r9State(a.status)+'"><b>'+esc(a.priority)+'</b><span>'+esc(a.item)+'</span><strong>'+esc(a.status)+'</strong><em>edge</em></div>'
    ).join('')+'</div></div>';
    h+='<div class="i-sec"><div class="i-sh">Static asset budget</div><div class="r9-assets">'+(h9.assets||[]).map(a=>'<span>'+esc(a.name)+' <b>'+r9Bytes(a.bytes)+'</b></span>').join('')+'</div><div class="hl-foot">Rollback path: <code>'+esc(h9.rollback||'')+'</code></div></div>';
  }
  h+='<div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('secBody').innerHTML=h;
}

/* ---- I8 发布与配置 ---- */
function loadRelease(){ relFetch(); }
async function relFetch(){ let d,c; try{ [d,c]=await Promise.all([fetch('/api/releases',{cache:'no-store'}).then(r=>r.json()),fetch('/api/config',{cache:'no-store'}).then(r=>r.json())]); }catch(e){ document.getElementById('relBody').innerHTML=iEmpty('发布信息获取失败'); return; } relRender(d,c); }
function relRender(d,c){
  const ts=document.getElementById('relTs'); if(ts) ts.textContent='发布 · 当前 '+d.current;
  const A=iAdmin();
  let h='<div class="i-kpis">'+iKpi('当前版本',d.current,'asset ver','info')+iKpi('发布记录',(d.releases||[]).length,'manifest')+iKpi('配置项',(c.config||[]).length,'config')+'</div>';
  // releases
  h+='<div class="i-sec"><div class="i-sh">发布历史 / 变更日志</div>';
  if((d.releases||[]).length){ h+='<div class="i-list">'+d.releases.map(r=>'<div class="i-li"><b>'+esc(r.ver)+'</b> <span class="i-dim">'+fmtClock(r.ts).slice(0,16)+' · '+esc(r.by||'')+'</span><br>'+esc(r.notes||'')+'</div>').join('')+'</div>'; } else h+=iEmpty('暂无发布记录 (deploy.sh 写入 manifest 后显示)');
  h+='<div class="hl-foot" style="margin-top:8px">回滚 (运维脚本, 非 Web 按钮以防误触): <code>'+esc(d.rollback_cmd||'')+'</code></div></div>';
  // config
  h+='<div class="i-sec"><div class="i-sh">配置中心'+(A?' <button class="i-mini" onclick="configForm()">+ 设置项</button>':'')+'</div>';
  if((c.config||[]).length){ h+='<div class="i-table"><table><thead><tr><th>键</th><th>值</th><th>类型</th><th>更新人</th></tr></thead><tbody>'+c.config.map(x=>'<tr><td><code>'+esc(x.key)+'</code></td><td>'+esc(x.value)+'</td><td>'+esc(x.type||'')+'</td><td class="i-dim">'+esc(x.updated_by||'')+'</td></tr>').join('')+'</tbody></table></div>'; } else h+=iEmpty('暂无配置项 (如通知 webhook URL 在此设置)');
  h+='</div><div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('relBody').innerHTML=h;
}
function configForm(){ iModalOpen('<h3>设置配置项</h3><div class="i-form"><label>键 <input id="cfKey" placeholder="如 webhook.wecom"></label><label>值 <input id="cfVal" placeholder="webhook URL 或值"></label><label>类型 <select id="cfType"><option>string</option><option>number</option><option>bool</option><option>url</option></select></label></div><button class="i-go" onclick="configSet()">保存</button><div class="hl-foot">提示: 通知 webhook 用键 webhook.wecom / webhook.dingtalk / webhook.feishu</div>'); }
async function configSet(){ const g=id=>document.getElementById(id).value; if(!g('cfKey')){ toast('键必填'); return; } try{ await iPost('/api/config',{key:g('cfKey'),value:g('cfVal'),type:g('cfType')}); iModalClose(); toast('已保存'); relFetch(); }catch(e){} }

/* ---- I9 数据治理 ---- */
function loadData(){ dataFetch(); }
async function dataFetch(){ let d; try{ d=await fetch('/api/data_inventory',{cache:'no-store'}).then(r=>r.json()); }catch(e){ document.getElementById('dataBody').innerHTML=iEmpty('数据治理获取失败'); return; } dataRender(d); }
function dataRender(d){
  const ts=document.getElementById('dataTs'); if(ts) ts.textContent='数据治理 · '+new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  const bk=d.backup||{}, rl=d.rollup||{}, rt=d.retention||{};
  let h='<div class="i-kpis">'+
    iKpi('库大小',(d.db_bytes/1048576).toFixed(2)+' MB','SQLite WAL')+
    iKpi('WAL',(d.wal_bytes/1024).toFixed(1)+' KB','写压力')+
    iKpi('完整性',d.integrity==='ok'?'OK ✓':d.integrity,'integrity_check',d.integrity==='ok'?'ok':'crit')+
    iKpi('备份心跳',bk.exists?iAge(bk.age_s)+' 前':'无','db_backup',bk.exists&&bk.age_s<172800?'ok':'warn')+'</div>';
  h+='<div class="i-2col"><div class="i-sec"><div class="i-sh">留存策略</div><div class="i-list">'+
    '<div class="i-li">原始 30s 样本 · 保留 <b>'+(rt.samples_raw_days||14)+' 天</b></div>'+
    '<div class="i-li">事件 / KPI / 告警 · 保留 <b>'+(rt.events_days||90)+' 天</b></div>'+
    '<div class="i-li">小时降采样 · <b>'+(rt.hourly_rollup||'长期')+'</b> (rollup 行 '+(rl.rows||0)+', 最新 '+(rl.last_hour?fmtClock(rl.last_hour).slice(0,13):'—')+')</div>'+
    '<div class="i-li">备份 · '+(bk.exists?('在线快照 '+(bk.bytes/1048576).toFixed(2)+' MB, '+iAge(bk.age_s)+'前'):'尚无快照 (db_backup.py 触发)')+'</div>'+
    '</div></div>';
  h+='<div class="i-sec"><div class="i-sh">数据导出 (治理留痕)</div><div class="i-list">'+
    '<div class="i-li"><a href="/api/export/events.csv?hours=168" class="i-link">事件 CSV (近 7 天)</a></div>'+
    '<div class="i-li"><a href="/api/export/history.csv?sys=lab&hours=168" class="i-link">状态历史 CSV (lab, 7 天)</a></div>'+
    '<div class="i-li"><a href="/api/export/workorders.csv" class="i-link">工单 CSV</a></div></div></div></div>';
  // table inventory
  h+='<div class="i-sec"><div class="i-sh">全表清单 ('+(d.tables||[]).length+' 表)</div><div class="data-tables">';
  (d.tables||[]).forEach(t=>{ h+='<div class="data-tb"><code>'+esc(t.table)+'</code><b>'+(t.rows==null?'—':t.rows.toLocaleString())+'</b></div>'; });
  h+='</div></div><div class="hl-foot" style="margin-top:14px">'+esc(d.note||'')+'</div>';
  document.getElementById('dataBody').innerHTML=h;
}

/* ---- I10 i18n 中英双语 ---- */
const I18N_DEFAULT_VER='site27-highlight-claims-20260709';
function i18nBootLang(){
  try{
    const sp=new URLSearchParams(location.search||'');
    const explicit=sp.get('lang');
    if(explicit==='zh'||explicit==='en'){
      localStorage.setItem('xrd_lang', explicit);
      localStorage.setItem('xrd_lang_default_ver', I18N_DEFAULT_VER);
      return explicit;
    }
    if(localStorage.getItem('xrd_lang_default_ver')!==I18N_DEFAULT_VER){
      localStorage.setItem('xrd_lang','zh');
      localStorage.setItem('xrd_lang_default_ver', I18N_DEFAULT_VER);
      return 'zh';
    }
  }catch(e){}
  try{ return localStorage.getItem('xrd_lang')||'zh'; }catch(e){ return 'zh'; }
}
function i18nToggle(){ const cur=(localStorage.getItem('xrd_lang')||'zh'); const nx=cur==='zh'?'en':'zh'; localStorage.setItem('xrd_lang',nx); localStorage.setItem('xrd_lang_default_ver',I18N_DEFAULT_VER); applyI18n(nx); toast(nx==='en'?'English':'中文'); }
const _i18nOrig=new WeakMap();
function applyI18n(lang){ lang=lang||localStorage.getItem('xrd_lang')||'zh'; const dict=(window.I18N&&window.I18N[lang])||{};
  document.documentElement.setAttribute('lang', lang==='en'?'en':'zh-CN');
  document.querySelectorAll('[data-i18n]').forEach(el=>{ const k=el.getAttribute('data-i18n');
    if(!_i18nOrig.has(el)) _i18nOrig.set(el, el.innerHTML);        // 首次快照中文原文
    if(lang==='en' && dict[k]!=null) el.innerHTML=dict[k];
    else el.innerHTML=_i18nOrig.get(el); });                       // 回中文用快照
  document.querySelectorAll('[data-i18n-ph]').forEach(el=>{ const k=el.getAttribute('data-i18n-ph');
    if(!el._phZh) el._phZh=el.getAttribute('placeholder')||'';
    el.setAttribute('placeholder', (lang==='en'&&dict[k]!=null)?dict[k]:el._phZh); });
  const b=document.getElementById('btnLang'); if(b) b.textContent=lang==='en'?'EN/中':'中/EN';
  _i18nFullApply(lang); _i18nObserve();   // 文本节点级全站翻译 + 动态内容监听
}

/* ---- 全站文本节点翻译引擎 (中→英 zhmap), 覆盖过渡动画/各页/动态渲染 ---- */
const _txtSnap=new WeakMap();
let _i18nObserver=null, _i18nBusy=false;
function _i18nSkip(p){ if(!p) return true; const t=p.nodeName;
  if(t==='SCRIPT'||t==='STYLE'||t==='CODE'||t==='TEXTAREA'||t==='INPUT'||t==='svg'||t==='SVG') return true;
  if(p.closest && p.closest('[data-i18n],code,.mono,svg')) return true;  // data-i18n 与代码/SVG 不碰
  return false; }
function _translateUnder(root, lang){
  const map=(window.I18N&&window.I18N.zhmap)||{};
  if(root.nodeType===3){ _translateOne(root, lang, map); return; }
  if(root.nodeType!==1) return;
  const w=document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {acceptNode(n){
    return (n.nodeValue && n.nodeValue.trim() && !_i18nSkip(n.parentNode)) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT; }});
  const ns=[]; let n; while(n=w.nextNode()) ns.push(n);
  ns.forEach(x=>_translateOne(x, lang, map));
}
function _translateOne(node, lang, map){
  const raw=node.nodeValue, t=raw.trim(); if(!t) return;
  if(lang==='en'){ const en=map[t]; if(en!=null && en!==t){ if(!_txtSnap.has(node)) _txtSnap.set(node, raw); node.nodeValue=raw.replace(t, en); } }
  else if(_txtSnap.has(node)){ node.nodeValue=_txtSnap.get(node); }
}
function _i18nFullApply(lang){ _i18nBusy=true; try{ _translateUnder(document.body, lang); }catch(e){} _i18nBusy=false; }
function _i18nObserve(){
  if(_i18nObserver) return;
  _i18nObserver=new MutationObserver(muts=>{
    if(_i18nBusy) return; if((localStorage.getItem('xrd_lang')||'zh')!=='en') return;
    _i18nBusy=true;
    try{ muts.forEach(m=>{
      if(m.type==='characterData'){ _translateOne(m.target, 'en', (window.I18N&&window.I18N.zhmap)||{}); }
      else m.addedNodes && m.addedNodes.forEach(nd=>{ if(nd.nodeType===1) _translateUnder(nd,'en'); else if(nd.nodeType===3) _translateOne(nd,'en',(window.I18N&&window.I18N.zhmap)||{}); });
    }); }catch(e){}
    _i18nBusy=false;
  });
  _i18nObserver.observe(document.body, {childList:true, subtree:true, characterData:true});
}

/* ---- H13 术语百科 ---- */
const GLOSSARY=[
  {cat:'材料/光谱', t:'近红外荧光粉 (NIR phosphor)', d:'受激发后在近红外 (700-1700nm) 发光的材料。本项目主攻 Cr³⁺/Ni²⁺ 掺杂石榴石, 用于生物成像、夜视、光谱检测。'},
  {cat:'材料/光谱', t:'掺杂 (doping)', d:'在主晶格 (host) 中替入少量激活离子 (如 Cr³⁺)。掺杂的元素、位点、浓度决定发光波长与效率。'},
  {cat:'材料/光谱', t:'λ_em (发射波长)', d:'荧光粉受激发后发出光的峰值波长。本项目用可微 Tanabe-Sugano 模型预测, MAE 6.2nm。'},
  {cat:'材料/光谱', t:'Tanabe-Sugano (晶体场)', d:'描述过渡金属离子 d 电子在晶体场中能级分裂的理论。本项目对 d³ 离子做 6×6 哈密顿对角化, autograd 可微反向设计配方。'},
  {cat:'材料/光谱', t:'Huang-Rhys 因子', d:'刻画电子跃迁时晶格弛豫强度的参数, 决定发射谱的展宽与斯托克斯位移。'},
  {cat:'材料/光谱', t:'XRD / PL', d:'XRD=X 射线衍射 (测晶体结构/相纯度); PL=光致发光谱 (测发光性质)。两者是荧光粉表征的核心手段。'},
  {cat:'AI/方法', t:'verdict (判决)', d:'AI 对一个候选配方给出的 5 级结论: GO 可做 / REVISE 需调整 / DROP 弃 / UNKNOWN 跨 host 未知。规则可推翻 BPU 虚拟推理。'},
  {cat:'AI/方法', t:'Conformal CI', d:'共形预测区间 —— 给预测附一个有覆盖率保证的置信区间 (本项目 90% CI 实测覆盖)。比单点预测诚实。'},
  {cat:'AI/方法', t:'MatterGen', d:'微软的生成式材料发现扩散模型, 按元素约束生成新晶体结构。本项目用它产出新颖候选池。'},
  {cat:'AI/方法', t:'MACE / MatterSim / CHGNet', d:'机器学习原子间势 (MLIP), 近 DFT 精度地算晶体能量/弛豫。本项目用作候选稳定性的双 MLIP 交叉验证。'},
  {cat:'AI/方法', t:'GP/EI 主动学习', d:'高斯过程 + 期望提升 —— 用已有实测数据指导下一轮该试哪个配方, 形成闭环越跑越准。'},
  {cat:'边缘 AI', t:'BPU (Bayes-e)', d:'RDK X5 上的神经网络加速单元 (10 TOPS)。本项目把 Qwen2 Transformer、YOLO、OCR 等搬上 BPU 做边缘推理。'},
  {cat:'边缘 AI', t:'swap-load slot', d:'BPU 显存 (CMA 391MB) 有限, 大模型切成多段、按需换入换出。本项目 5 个 LLM 槽轮换推理。'},
  {cat:'边缘 AI', t:'INT8 量化', d:'把模型权重从 float32 压到 8 位整数, 体积/算力降 4 倍。BPU 只跑量化模型; 但开放生成会精度崩坏 (本项目诚实用 probing 而非生成)。'},
  {cat:'机器人', t:'SLAM', d:'即时定位与建图 —— 机器人边走边建环境地图并定位自己。车载脑用激光雷达 + slam_toolbox 实现。'},
  {cat:'机器人', t:'Nav2 / MPPI', d:'Nav2=ROS2 导航栈; MPPI=模型预测路径积分控制 (采样大量轨迹选最优)。本项目把 MPPI 代价评估搬上 BPU 加速 4.1×。'},
  {cat:'机器人', t:'AprilTag', d:'类似二维码的视觉基准标记, 用于精确估计相机相对工件/工位的 6 自由度位姿。双臂工位用它做标定。'},
  {cat:'机器人', t:'DH 参数 / 正运动学', d:'描述机械臂连杆几何的标准参数 (Denavit-Hartenberg)。由关节角算末端位置叫正运动学 (FK), 本项目孪生场景用真 DH 链逐帧 FK。'},
  {cat:'平台/工程', t:'镜像兜底 / active health', d:'设备关机时, 公网自动切到 VPS 上的常驻镜像 (跑真算法), UI 与功能照常。Caddy 每 10s 探活、2 连败切换。'},
  {cat:'平台/工程', t:'SSO / RBAC', d:'单点登录 + 基于角色的访问控制。一次登录三系统通; admin>member>judge 分级权限, 评委只读。'},
  {cat:'平台/工程', t:'historian / SSE', d:'historian=工业历史数据库 (状态/告警/KPI 落盘); SSE=服务端推送, 前端实时收事件。对标 Ignition。'},
  {cat:'平台/工程', t:'ISA-18.2 / ISA-88 / Part 11', d:'工业标准: 报警管理 / 批记录 / 电子记录审计。本平台的告警中心、批次工单、SHA-256 审计链分别对标它们。'},
];
function loadGlossary(){ glRender((document.getElementById('glIn')||{}).value||''); }
function glRender(q){
  q=(q||'').trim().toLowerCase();
  const body=document.getElementById('glBody'); if(!body) return;
  const items=GLOSSARY.filter(g=>!q||(g.t+' '+g.d+' '+g.cat).toLowerCase().includes(q));
  if(!items.length){ body.innerHTML='<div class="oe-empty">无匹配术语</div>'; return; }
  const cats={}, order=[];
  items.forEach(g=>{ if(!cats[g.cat]){cats[g.cat]=[];order.push(g.cat);} cats[g.cat].push(g); });
  body.innerHTML=order.map(c=>'<div class="gl-cat"><div class="gl-ch">'+esc(c)+'</div><div class="gl-grid">'+
    cats[c].map(g=>'<div class="gl-card"><b>'+esc(g.t)+'</b><span>'+esc(g.d)+'</span></div>').join('')+'</div></div>').join('');
}

/* ---- H27 更新日志 / What's New ---- */
const CHANGELOG=[
  {d:'2026-06-13', tag:'UI', t:'全面功能 + UI 升级轮 (进行中)', items:['演示就绪预检 + 在险册','问平台运维副驾 (实时接地)','材料图鉴 Atlas (250 真候选)','配方构建器 (点周期表 → 真预测 + 血缘溯源)','通知中心 + favicon 角标 + 告警 AI 诊断','工程档案馆 / 模型注册表 / 标准合规墙 / 成本能效 / 术语百科','设计 token 体系 + 骨架屏 + 微交互 + 视图过渡 + 3D/孪生质感']},
  {d:'2026-06-13', tag:'核心', t:'工业级全栈 + 指挥中心门户', items:['SQLite historian + SSE 实时推送','ISA-18.2 告警生命周期 + ISA-88 批次工单','StatusPage SLO 双口径 + SHA-256 审计链','RBAC + 登录审计 + 数字孪生','apex 统一门户 + 一条龙演示 + 大屏轮播']},
  {d:'2026-06-12', tag:'部署', t:'三机真机部署 + 公网整合', items:['车载脑 cockpit_bridge 上电','主机械臂 arm01 真机驾驶舱 (真关节角 FK)','指挥中心统一门户 (一处登录统管三脑)']},
  {d:'2026-06-11', tag:'公网', t:'安全公网上线 (双保险)', items:['Cloudflare + 香港 VPS Caddy + frp 隧道','SSO 单点登录 (member/judge 分级)','VPS 常驻镜像 (设备关机 UI 照常在)','AI 预测大迭代 (/copilot /campaign /pareto /audit)']},
  {d:'2026-04 ~ 05', tag:'里程碑', t:'具身脑 + 双臂工位', items:['具身脑 8 ROS2 包全栈 + Round 4 BPU Sprint','SmolVLM VLM / PP-OCRv4 / MPPI / XFeat 上 BPU','双 myCobot 280 双臂工位 v4 十幕剧本']},
  {d:'2026-04', tag:'里程碑', t:'9 本地 LLM + BPU Transformer', items:['X5 BPU 24 层 Qwen2 真机实测 (553ms)','5 BPU swap-load slot + 4 CPU llama-server','合成预测系统 dashboard:8888 53 路由']},
];
function loadChangelog(){
  const body=document.getElementById('clBody'); if(!body) return;
  body.innerHTML='<div class="cl-feed">'+CHANGELOG.map(c=>
    '<div class="cl-item"><div class="cl-dot"></div><div class="cl-c">'+
    '<div class="cl-h"><span class="cl-date">'+esc(c.d)+'</span><span class="cl-tag '+(c.tag==='核心'||c.tag==='公网'?'hot':'')+'">'+esc(c.tag)+'</span><b>'+esc(c.t)+'</b></div>'+
    '<ul>'+c.items.map(i=>'<li>'+esc(i)+'</li>').join('')+'</ul></div></div>').join('')+'</div>';
}

/* ---- H24 周期表配方构建器 ---- */
// 荧光粉常用元素子集 (sym, name, period-row, group-col, kind), 真周期表布局
const BD_ELEMS=[
  ['Li','锂',2,1,'a'],['Mg','镁',3,2,'a'],['Ca','钙',4,2,'a'],['Sr','锶',5,2,'a'],['Ba','钡',6,2,'a'],
  ['Sc','钪',4,3,'t'],['Y','钇',5,3,'t'],['Ti','钛',4,4,'t'],['Zr','锆',5,4,'t'],
  ['Cr','铬',4,6,'d'],['Mn','锰',4,7,'d'],['Ni','镍',4,10,'d'],['Zn','锌',4,12,'t'],
  ['B','硼',2,13,'p'],['Al','铝',3,13,'p'],['Ga','镓',4,13,'p'],['In','铟',5,13,'p'],
  ['Si','硅',3,14,'p'],['Ge','锗',4,14,'p'],['Sn','锡',5,14,'p'],
  ['N','氮',2,15,'n'],['P','磷',3,15,'n'],['O','氧',2,16,'n'],['F','氟',2,17,'n'],
  ['La','镧',6,3,'l'],['Gd','钆',6,3.4,'l'],['Lu','镥',6,3.8,'l'],['Yb','镱',6,4.2,'l'],
];
const BD_KIND={a:['#22c55e','碱/碱土'],t:['#0891b2','过渡(主)'],d:['#e11d48','激活离子'],p:['#7c3aed','主族阳离子'],n:['#d97706','阴离子'],l:['#2563eb','稀土']};
let BD_F=[];  // [[sym,count],...]
function loadBuild(){
  const pt=document.getElementById('bdPtable'); if(!pt) return;
  if(!pt.dataset.built){
    pt.innerHTML=BD_ELEMS.map(e=>{
      const k=BD_KIND[e[4]]||BD_KIND.p;
      const col=Math.round(e[3]); const row=e[2];
      return '<button class="bd-el" data-s="'+e[0]+'" style="grid-row:'+(row-1)+';grid-column:'+col+';--ec:'+k[0]+'" '+
        'onclick="bdAdd(\''+e[0]+'\')" title="'+e[1]+' ('+k[1]+')"><b>'+e[0]+'</b><span>'+e[1]+'</span></button>';
    }).join('');
    pt.dataset.built='1';
  }
  bdRenderF();
}
function bdAdd(s){
  const x=BD_F.find(f=>f[0]===s); if(x) x[1]++; else BD_F.push([s,1]); bdRenderF();
}
function bdClear(){ BD_F=[]; bdRenderF(); document.getElementById('bdResult').innerHTML=''; }
function bdFormulaStr(){ return BD_F.map(f=>f[0]+(f[1]>1?f[1]:'')).join(''); }
function _sub(s){ const m={'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉'};
  return s.replace(/([A-Za-z\)])(\d+)/g,(_,a,b)=>a+b.replace(/\d/g,d=>m[d])); }
function bdRenderF(){
  const el=document.getElementById('bdFormula');
  if(el) el.innerHTML=BD_F.length?_sub(bdFormulaStr()):'<span style="color:var(--ink3)">点元素开始…</span>';
  document.querySelectorAll('.bd-el').forEach(b=>{ const f=BD_F.find(x=>x[0]===b.dataset.s);
    b.classList.toggle('sel', !!f); const c=b.querySelector('.bd-cnt');
    if(f){ if(!c){ const s=document.createElement('span'); s.className='bd-cnt'; b.appendChild(s); b.querySelector('.bd-cnt').textContent=f[1]; }
      else c.textContent=f[1]; } else if(c) c.remove(); });
}
function bdPreset(formula, site){
  // 解析简单化学式 → BD_F
  BD_F=[]; const re=/([A-Z][a-z]?)(\d*)/g; let m;
  while((m=re.exec(formula))){ if(!m[1]) continue; BD_F.push([m[1], m[2]?+m[2]:1]); }
  document.getElementById('bdSite').value=site||''; bdRenderF();
}
async function bdPredict(){
  const formula=bdFormulaStr();
  if(formula.length<2){ toast('请先搭出化学式 (至少 2 个元素)'); return; }
  const symbol=document.getElementById('bdSymbol').value;
  const site=(document.getElementById('bdSite').value||'').trim();
  const pct=parseFloat(document.getElementById('bdPct').value||'1')||1;
  const btn=document.getElementById('bdGo'), res=document.getElementById('bdResult');
  btn.disabled=true; btn.textContent='预测中 (调真 predict_engine)…';
  res.innerHTML='<div class="skel-rows"><div class="skel"></div><div class="skel"></div></div>';
  try{
    const d=await fetch('/api/quick_predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({formula,symbol,site,pct})}).then(r=>r.json());
    if(d.error){ res.innerHTML='<div class="oe-empty">'+esc(d.error)+'</div>'; }
    else{
      const s=d.summary||{};
      const vcls=s.verdict==='GO'?'go':s.verdict==='REVISE'?'rev':s.verdict==='DROP'?'drop':'';
      const ci=(s.ci_lo!=null&&s.ci_hi!=null)?(' · CI80 ['+Math.round(s.ci_lo)+'–'+Math.round(s.ci_hi)+']'):'';
      res.innerHTML='<div class="bd-rcard '+vcls+'">'+
        '<div class="bd-rv"><span class="bd-verd '+vcls+'">'+esc(s.verdict||'—')+'</span>'+
          '<span class="bd-src">'+(d.source==='real'?'● 真机':'◐ 镜像真算')+'</span></div>'+
        '<div class="bd-rrow"><span>发射波长 λ_em</span><b>'+(s.lambda_em!=null?(Math.round(s.lambda_em)+' nm'+ci):'—')+'</b></div>'+
        (s.confidence!=null?'<div class="bd-rrow"><span>置信度</span><b>'+s.confidence+'</b></div>':'')+
        (d.sinter_temp_C!=null?'<div class="bd-rrow"><span>烧结温度</span><b>'+d.sinter_temp_C+' °C</b></div>':'')+
        (s.analog?'<div class="bd-rrow"><span>最近类比</span><b>'+esc(s.analog)+'</b></div>':'')+
        (s.reason?'<div class="bd-reason">'+esc(s.reason)+'</div>':'')+
        ((d.flags||[]).length?'<div class="bd-flags">'+d.flags.slice(0,4).map(f=>'<span>⚑ '+esc(typeof f==='string'?f:(f.msg||f.type||''))+'</span>').join('')+'</div>':'')+
        bdLineage(d,s)+
        (s.trace_id?'<div class="bd-trace">trace: '+esc(s.trace_id)+'</div>':'')+'</div>';
    }
  }catch(e){ res.innerHTML='<div class="oe-empty">预测请求失败</div>'; }
  btn.disabled=false; btn.textContent='⚡ 真预测';
}

// H15 血缘: 把 predict_engine 真实管线画成溯源链, 标哪几步对本次预测真出了值
function bdLineage(d,s){
  const stages=[
    {k:'解析',on:true,t:'化学式 + 掺杂解析'},
    {k:'Shannon',on:true,t:'离子半径表'},
    {k:s.analog?'类比':'Vegard',on:true,t:s.analog?('PL 类比 '+s.analog):'峰位修正'},
    {k:'TS 谱',on:s.lambda_em!=null,t:'Tanabe-Sugano 虚拟谱'},
    {k:'失配',on:(d.flags||[]).length>0,t:(d.flags||[]).length+' flag'},
    {k:'R1 判决',on:!!s.verdict,t:s.verdict||'—'},
  ];
  return '<div class="bd-lin"><div class="bd-lin-h">🧬 预测血缘 (predict_engine 管线溯源)</div><div class="bd-lin-flow">'+
    stages.map((st,i)=>'<span class="bd-ln '+(st.on?'on':'off')+'" title="'+esc(st.t)+'">'+esc(st.k)+'</span>'+
      (i<stages.length-1?'<i class="bd-lar">→</i>':'')).join('')+'</div></div>';
}

/* ---- H8/H21 实测导入向导 (DropZone, 纯本地解析) ---- */
let IMP_BOUND=false;
function loadImport(){
  const z=document.getElementById('impZone'); if(!z||IMP_BOUND) return; IMP_BOUND=true;
  ['dragenter','dragover'].forEach(ev=>z.addEventListener(ev,e=>{e.preventDefault();z.classList.add('over');}));
  ['dragleave','drop'].forEach(ev=>z.addEventListener(ev,e=>{e.preventDefault();z.classList.remove('over');}));
  z.addEventListener('drop',e=>{ if(e.dataTransfer&&e.dataTransfer.files) impFiles(e.dataTransfer.files); });
}
const IMP_COLS=[
  {key:'λ_em 发射波长',re:/(lambda|λ|em|emission|波长|peak)/i,c:'#7c3aed'},
  {key:'掺杂',re:/(dopant|掺杂|cr|ni|activator)/i,c:'#e11d48'},
  {key:'主晶格 host',re:/(host|formula|composition|主晶|化学式|sample)/i,c:'#2563eb'},
  {key:'浓度',re:/(pct|conc|浓度|percent|mol)/i,c:'#d97706'},
  {key:'激发',re:/(ex|excit|激发)/i,c:'#0891b2'},
];
function impParse(text){
  const lines=text.replace(/\r/g,'').split('\n').filter(l=>l.trim().length);
  if(!lines.length) return null;
  const delim=(lines[0].split('\t').length>lines[0].split(',').length)?'\t':',';
  const rows=lines.map(l=>l.split(delim).map(c=>c.trim()));
  return {header:rows[0], body:rows.slice(1), delim, n:rows.length-1};
}
function impFiles(files){
  if(!files||!files.length) return;
  const f=files[0];
  if(f.size>4*1024*1024){ toast('文件过大 (>4MB), 仅预览前段'); }
  const rd=new FileReader();
  rd.onload=()=>{ try{ impRender(f.name, impParse(String(rd.result).slice(0,500000))); }
    catch(e){ document.getElementById('impResult').innerHTML='<div class="oe-empty">解析失败, 确认是 CSV/TSV</div>'; } };
  rd.readAsText(f.slice(0,500000));
}
function impRender(name, p){
  const res=document.getElementById('impResult'); if(!res) return;
  if(!p||!p.header){ res.innerHTML='<div class="oe-empty">文件为空</div>'; return; }
  // 列识别
  const detected=p.header.map(h=>{ for(const c of IMP_COLS){ if(c.re.test(h)) return c; } return null; });
  const nDet=detected.filter(Boolean).length;
  const preview=p.body.slice(0,8);
  let html='<div class="imp-card"><div class="imp-fh"><b>📄 '+esc(name)+'</b>'+
    '<span class="imp-meta">'+p.n+' 行 · '+p.header.length+' 列 · 分隔符 "'+(p.delim==='\t'?'Tab':',')+'" · 识别 '+nDet+' 个关键列</span></div>';
  // 识别到的列徽章
  if(nDet){ html+='<div class="imp-tags">';
    p.header.forEach((h,i)=>{ const c=detected[i]; if(c) html+='<span class="imp-tag" style="--tc:'+c.c+'">'+esc(c.key)+' ← '+esc(h)+'</span>'; });
    html+='</div>'; }
  // 表预览
  html+='<div class="imp-tblw"><table class="imp-tbl"><tr>'+
    p.header.map((h,i)=>'<th'+(detected[i]?' class="det" style="--tc:'+detected[i].c+'"':'')+'>'+esc(h)+'</th>').join('')+'</tr>'+
    preview.map(r=>'<tr>'+p.header.map((_,i)=>'<td>'+esc(r[i]||'')+'</td>').join('')+'</tr>').join('')+
    '</table></div>'+
    (p.n>8?'<div class="imp-more">… 另有 '+(p.n-8)+' 行 (预览前 8 行)</div>':'')+
    '<div class="imp-next">✓ 本地预览完成 · 数据未上传。确认无误后进 <b onclick="go(\'lab\')" style="cursor:pointer;color:var(--blue)">AI 脑</b> 走正式回填 (actuals.csv + 准确率 KPI 重算)。</div>'+
    '</div>';
  res.innerHTML=html;
}

/* ---- H30 电子实验笔记 (ELN, 真持久化) ---- */
let _elnQ='', _elnT=null;
function elnSearchDebounce(v){ _elnQ=v; clearTimeout(_elnT); _elnT=setTimeout(loadEln, 280); }
function elnTime(ts){ const t=new Date(ts*1000);
  return t.toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'})+' '+t.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'}); }
async function loadEln(){
  // 评委只读: 隐藏创建表单
  const isJudge=(window._role==='judge');
  const cr=document.getElementById('elnCreate'); if(cr) cr.style.display=isJudge?'none':'';
  const list=document.getElementById('elnList'); if(!list) return;
  try{
    const d=await fetch('/api/eln'+(_elnQ?('?q='+encodeURIComponent(_elnQ)):''),{cache:'no-store'}).then(r=>r.json());
    const es=d.entries||[];
    const ts=document.getElementById('elnTs');
    if(ts) ts.textContent=(d.total||0)+' 篇笔记'+(_elnQ?(' · 命中 '+es.length):'');
    if(!es.length){ list.innerHTML='<div class="oe-empty" style="padding:40px 0">'+(_elnQ?'无匹配笔记':'还没有笔记 — 用上方表单记录第一条')+'</div>'; return; }
    list.innerHTML=es.map(e=>'<div class="eln-card">'+
      '<div class="eln-h"><b>'+esc(e.title)+'</b>'+
        (e.formula?'<span class="eln-fm" onclick="go(\'build\')">'+esc(e.formula)+'</span>':'')+
        (isJudge?'':'<button class="eln-del" onclick="elnDel('+e.id+')" title="删除">🗑</button>')+'</div>'+
      (e.body?'<div class="eln-body">'+esc(e.body)+'</div>':'')+
      '<div class="eln-meta">'+(e.tags||[]).map(t=>'<span class="eln-tag">#'+esc(t)+'</span>').join('')+
        '<span class="eln-by">'+esc(e.author||'—')+' · '+elnTime(e.ts)+'</span></div>'+
      '</div>').join('');
  }catch(e){ list.innerHTML='<div class="oe-empty">加载失败</div>'; }
}
async function elnCreate(){
  const title=(document.getElementById('elnTitle').value||'').trim();
  if(!title){ toast('请填标题'); return; }
  const body=(document.getElementById('elnBody').value||'').trim();
  const formula=(document.getElementById('elnFormula').value||'').trim();
  const tags=(document.getElementById('elnTags').value||'').split(/[,，]/).map(t=>t.trim()).filter(Boolean);
  const btn=document.getElementById('elnBtn'); btn.disabled=true;
  try{
    const r=await fetch('/api/eln',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title,body,formula,tags})});
    if(r.status===403){ toast('评委账号为只读演示, 无法记录'); }
    else if(r.ok){ toast('✓ 已保存','ok');
      document.getElementById('elnTitle').value=''; document.getElementById('elnBody').value='';
      document.getElementById('elnFormula').value=''; document.getElementById('elnTags').value='';
      _elnQ=''; const s=document.getElementById('elnSearch'); if(s) s.value=''; loadEln(); }
    else toast('保存失败 ('+r.status+')');
  }catch(e){ toast('保存失败'); }
  btn.disabled=false;
}
async function elnDel(id){
  try{
    const r=await fetch('/api/eln/'+id,{method:'DELETE'});
    if(r.status===403){ toast('评委账号只读'); return; }
    if(r.ok){ toast('已删除'); loadEln(); } else toast('删除失败');
  }catch(e){ toast('删除失败'); }
}

/* ---- H3 问平台 Copilot (运维副驾) ---- */
let CP_OPEN=false, CP_INIT=false, CP_BUSY=false;
function cpToggle(){
  CP_OPEN=!CP_OPEN;
  document.getElementById('cpPanel').classList.toggle('show', CP_OPEN);
  document.getElementById('cpFab').classList.toggle('hide', CP_OPEN);
  if(CP_OPEN){
    if(!CP_INIT){ cpInit(); CP_INIT=true; }
    setTimeout(()=>{ const i=document.getElementById('cpIn'); if(i) i.focus(); }, 80);
  }
}
function cpMd(s){ // 极简 markdown: **bold** + 换行
  return esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');
}
function cpPush(role, html){
  const b=document.getElementById('cpBody');
  const d=document.createElement('div'); d.className='cp-msg '+role;
  d.innerHTML=html; b.appendChild(d); b.scrollTop=b.scrollHeight; return d;
}
async function cpInit(){
  try{
    const d=await fetch('/api/copilot',{cache:'no-store'}).then(r=>r.json());
    cpPush('bot','<div class="cp-bub">👋 我是<b>指挥中心运维副驾</b>。'+esc(d.intro||'')+
      '<br>一问可<b>多步巡检</b>: <b>状态 / 告警 / 就绪 / KPI / 事件 / 候选 / 模型</b>, 也能讲<b>架构 / 亮点 / 自愈</b>。试试「巡检一遍」。</div>');
    cpSuggest(d.suggest||[]);
    window._cpLLM = !!d.llm;
    const db=document.getElementById('cpDeepBar'); if(db) db.style.display = d.llm ? 'flex' : 'none';
  }catch(e){ cpPush('bot','<div class="cp-bub">副驾连接失败, 请稍后重试。</div>'); }
}
function cpSuggest(list){
  const s=document.getElementById('cpSugg'); if(!s) return; s.innerHTML='';
  (list||[]).forEach(q=>{ const c=document.createElement('span'); c.className='cp-chip';
    c.textContent=q; c.onclick=()=>{ document.getElementById('cpIn').value=q; cpSend(); }; s.appendChild(c); });
}
function cpDeepToggle(){
  const on=!!(document.getElementById('cpDeep')||{}).checked;
  const h=document.getElementById('cpDeepHint');
  if(h) h.textContent = on ? 'DeepSeek 据真数据自然语言作答 (~1-2s)' : '规则巡检即时, 据实可核';
  const bar=document.getElementById('cpDeepBar'); if(bar) bar.classList.toggle('on',on);
}
async function cpSend(){
  if(CP_BUSY) return;
  const inp=document.getElementById('cpIn'); const q=(inp.value||'').trim();
  if(!q) return;
  inp.value=''; cpPush('user','<div class="cp-bub">'+esc(q)+'</div>');
  document.getElementById('cpSugg').innerHTML='';
  CP_BUSY=true; const tip=cpPush('bot','<div class="cp-bub cp-typing"><i></i><i></i><i></i></div>');
  try{
    const deep=!!(document.getElementById('cpDeep')||{}).checked;
    const d=await fetch('/api/copilot',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q,deep})}).then(r=>r.json());
    let html='<div class="cp-bub">';
    const steps=(d.steps||[]).filter(s=>s);
    if(steps.length>1){ html+='<div class="cp-steps"><span class="cp-stp-h">🔎 巡检 '+steps.length+' 项</span>'+
      steps.map((s,i)=>'<span class="cp-stp">'+esc(s)+'</span>'+(i<steps.length-1?'<span class="cp-arr">→</span>':'')).join('')+'</div>'; }
    html+=cpMd(d.answer||'(无回答)');
    if((d.facts||[]).length){ html+='<div class="cp-facts">';
      d.facts.forEach(f=>{ const st=f.status?(' '+f.status):'';
        html+='<div class="cp-fact'+st+'"><span>'+esc(f.label)+'</span><b>'+esc(f.value)+'</b></div>'; });
      html+='</div>'; }
    if(d.grounded){ html+='<div class="cp-grd">● 实时数据接地'+(steps.length>1?' · 多步 Agent':'')+
      (d.llm?' · 🧠 '+esc(d.llm)+' 合成':'')+'</div>'; }
    else if(d.llm){ html+='<div class="cp-grd">🧠 '+esc(d.llm)+' 合成</div>'; }
    html+='</div>';
    tip.innerHTML=html;
    const acts=(d.actions||[]).filter(a=>a&&a[1]);
    if(acts.length){ const row=document.createElement('div'); row.className='cp-acts';
      acts.forEach(a=>{ const btn=document.createElement('button'); btn.className='cp-act';
        btn.textContent='→ '+a[0]; btn.onclick=()=>{ cpToggle(); go(a[1]); }; row.appendChild(btn); });
      tip.appendChild(row); }
    const fps=(d.followups||[]).filter(x=>x);
    if(fps.length){ const fr=document.createElement('div'); fr.className='cp-fups';
      fr.innerHTML='<span class="cp-fup-h">追问</span>';
      fps.forEach(f=>{ const c=document.createElement('span'); c.className='cp-fup';
        c.textContent=f; c.onclick=()=>{ document.getElementById('cpIn').value=f; cpSend(); }; fr.appendChild(c); });
      tip.appendChild(fr); }
    document.getElementById('cpBody').scrollTop=1e9;
  }catch(e){ tip.innerHTML='<div class="cp-bub">请求失败, 请重试。</div>'; }
  CP_BUSY=false;
}

/* ---- 运维总览 ---- */
const OPS_META={ lab:{ic:'🧠',nm:'AI 脑'}, car:{ic:'🚗',nm:'车载脑'}, arm:{ic:'🦾',nm:'机械臂'} };
function servText(s){ return s==='real'?'真机直连':s==='mirror'?'镜像演示':'离线'; }
async function pollOps(){
  try{
    const d=await fetch('/api/ops',{cache:'no-store'}).then(r=>r.json());
    const grid=document.getElementById('opsGrid'); const sys=d.systems||{};
    let on=0;
    grid.innerHTML='';
    ['lab','car','arm'].forEach(k=>{
      const s=sys[k]||{}; const m=OPS_META[k]; const serv=s.serving||'down';
      if(s.real_online) on++;
      const realA = serv==='real', mirA = serv==='mirror';
      const realChips = Object.entries(s.metrics||{}).map(([kk,vv])=>`<span class="chip">${kk}: ${vv}</span>`).join('');
      const card=document.createElement('div'); card.className='opc';
      card.innerHTML=`
        <div class="oh"><span class="ic">${m.ic}</span><span class="nm">${m.nm} · ${(s.name||'').split(' · ')[1]||''}</span>
          <span class="serv ${serv}">${serv==='real'?'● ':serv==='mirror'?'◐ ':'○ '}${servText(serv)}</span></div>
        <div class="chain">
          <div class="node active"><div class="nt">访客</div><div class="nv">🌐 Cloudflare</div></div>
          <div class="conn">→</div>
          <div class="node active"><div class="nt">回源</div><div class="nv">🖥 VPS Caddy</div></div>
          <div class="conn">→</div>
          <div class="node split">
            <div class="leg ${realA?'active':''}"><span class="ld ${s.real_online?'on':'off'}"></span>真机隧道<span class="lms">${s.real_online?(s.real_ms+'ms'):'—'}</span></div>
            <div class="leg ${mirA?'active':''}"><span class="ld ${s.mirror_online?'on':'off'}"></span>VPS 镜像<span class="lms">${s.mirror_online?(s.mirror_ms+'ms'):'—'}</span></div>
          </div>
        </div>
        ${realChips?`<div class="ometrics">${realChips}</div>`:''}`;
      grid.appendChild(card);
    });
    document.getElementById('opsTs').textContent = `真机在线 ${on}/3 · 镜像兜底全在 · ${new Date().toLocaleTimeString('zh-CN')}`;
  }catch(e){}
}
setInterval(()=>{ if(cur==='ops') pollOps(); }, 5000);

/* ---- P5: 批次工单 (真工单系统 — 创建即绑 lab 真预测, 取代旧演示种子) ---- */
function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
let _woCur=null;
async function loadWos(){
  try{
    const d=await fetch('/api/workorders?limit=50',{cache:'no-store'}).then(r=>r.json());
    const grid=document.getElementById('mqGrid'); if(!grid) return; grid.innerHTML='';
    const wos=d.workorders||[];
    if(!wos.length) grid.innerHTML='<div class="oe-empty">还没有工单 — 用上方表单创建第一条 (自动跑 AI 预测)</div>';
    wos.forEach(w=>grid.appendChild(woCard(w)));
    const ts=document.getElementById('mqTs');
    if(ts) ts.textContent='工单 '+wos.length+' 条 · 进行中 '+(d.n_open||0)+' · 已收单 '+(d.n_done||0)+' · '+new Date().toLocaleTimeString('zh-CN');
  }catch(e){}
}
function renderMissions(){ loadWos(); }   // 兼容旧入口名
function woCard(w){
  const st=w.state==='open'?'RUN':w.state==='done'?'GO':'QUEUE';
  const badge=w.state==='open'?'● 进行中':w.state==='done'?'✓ 已收单':'✕ 已取消';
  const stages=w.stages||[]; let sh='';
  stages.forEach((s,i)=>{
    const parts=s.split('·');
    const cls=i<w.stage?'done':(i===w.stage&&w.state==='open'?'run':'');
    sh+='<div class="mqs '+cls+'"><div class="node2"><div class="s1"><span class="dotmark"></span>'+parts[0]+'</div><div class="s2">'+(parts[1]||'')+'</div></div>'+(i<stages.length-1?'<span class="ar">→</span>':'')+'</div>';
  });
  const overall=Math.min(100, (w.stage||0)/stages.length*100);
  const p=w.pred||{};
  const vchip=w.verdict?('<span class="mq-dop">verdict: <b>'+esc(w.verdict)+'</b></span>'):'';
  const lam=p.lambda_em?('<span class="mq-dop">λ_em '+Math.round(p.lambda_em)+'nm</span>'):'';
  const card=document.createElement('div'); card.className='mq';
  card.innerHTML='<div class="mq-top"><span class="mq-id">'+esc(w.code)+'</span><span class="mq-f">'+esc(w.formula)+'</span>'+
    '<span class="mq-dop">'+esc(w.dop_symbol||'')+' @'+esc(w.dop_site||'?')+' '+esc(w.dop_pct)+'%</span>'+vchip+lam+
    '<span class="mq-vd vd-'+st+'">'+badge+'</span></div>'+
    '<div class="mq-stages">'+sh+'</div>'+
    '<div class="mq-bar"><div class="f" style="width:'+overall+'%"></div></div>'+
    '<div class="mq-meta"><span>进度 '+overall.toFixed(0)+'% · 建单 '+esc(w.created_by||'')+'</span><span class="wo-acts">'+
      '<a onclick="woDetail('+w.id+')">📄 批次档案</a>'+
      (w.state==='open'&&w.stage<4?'<a onclick="woAdvance('+w.id+')">▶ 推进</a>':'')+
      (w.state==='open'&&w.stage>=4?'<a onclick="woBackfillUi('+w.id+')">⏎ 实测回填</a>':'')+
      (w.state==='open'?'<a onclick="woCancel('+w.id+')">✕ 取消</a>':'')+
    '</span></div>';
  return card;
}
async function createWo(){
  const fSel=document.getElementById('woFormula');
  const formula=fSel.value==='_custom'?(document.getElementById('woCustom').value||'').trim():fSel.value;
  const symbol=document.getElementById('woSymbol').value;
  const site=(document.getElementById('woSite').value||'').trim();
  const pct=parseFloat(document.getElementById('woPct').value)||1.0;
  if(!formula){ toast('请填化学式'); return; }
  const btn=document.getElementById('woBtn'); btn.disabled=true; btn.textContent='AI 预测中…';
  try{
    const r=await fetch('/api/workorders',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({formula,symbol,site,pct})});
    if(r.status===403) toast('评委账号为只读演示, 无法建单');
    else if(r.ok){ const d=await r.json();
      toast('✓ '+d.workorder.code+' 已建 · verdict: '+(d.workorder.verdict||'—')); loadWos(); }
    else{ const d=await r.json().catch(()=>({})); toast('建单失败: '+(d.error||r.status)); }
  }catch(e){ toast('建单失败'); }
  btn.disabled=false; btn.textContent='⚡ 创建 + AI 预测';
}
async function _woPost(url, body){
  try{
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
      body:body?JSON.stringify(body):null});
    if(r.status===403){ toast('评委账号为只读演示'); return false; }
    if(r.ok){ loadWos(); return true; }
    const d=await r.json().catch(()=>({})); toast(d.error||('失败 '+r.status)); return false;
  }catch(e){ toast('请求失败'); return false; }
}
async function woAdvance(id){ if(await _woPost('/api/workorders/'+id+'/advance')) toast('▶ 已推进'); }
async function woCancel(id){ if(await _woPost('/api/workorders/'+id+'/cancel')) toast('✕ 已取消'); }
function woBackfillUi(id){
  const v=prompt('实测发射峰 λ_obs (nm), 可留空:'); if(v===null) return;
  const note=prompt('收单备注 (可留空):'); if(note===null) return;
  _woPost('/api/workorders/'+id+'/backfill',{lambda_obs:(v||'').trim(),note:(note||'').trim()})
    .then(ok=>{ if(ok) toast('✅ 已收单'); });
}
async function woDetail(id){
  try{
    const w=await fetch('/api/workorders/'+id,{cache:'no-store'}).then(r=>r.json());
    if(w.error){ toast(w.error); return; }
    _woCur=w;
    document.getElementById('wodCode').textContent=w.code+' · '+w.formula;
    const stw=document.getElementById('wodState');
    stw.textContent=w.state==='open'?'进行中':w.state==='done'?'已收单':'已取消';
    stw.className='wost '+w.state;
    const p=w.pred||{};
    let rows='';
    const add=(k,v)=>{ if(v!=null&&v!=='') rows+='<div class="wodr"><span>'+k+'</span><b>'+esc(v)+'</b></div>'; };
    add('化学式', w.formula);
    add('掺杂', (w.dop_symbol||'')+' @'+(w.dop_site||'?')+' '+w.dop_pct+'%');
    add('AI verdict', p.verdict); add('置信度', p.confidence); add('判据', p.reason);
    add('预测 λ_em', p.lambda_em?Math.round(p.lambda_em)+' nm':null);
    add('CI80', (p.ci_lo!=null&&p.ci_hi!=null)?Math.round(p.ci_lo)+' ~ '+Math.round(p.ci_hi)+' nm':null);
    add('最近类比', p.analog); add('T50 热稳', p.t50_k?p.t50_k+' K':null);
    add('烧结温度', p.sinter_temp_C?p.sinter_temp_C+' °C':null);
    add('trace_id', p.trace_id);
    add('预测来源', w.pred_source==='real'?'真机直连':w.pred_source==='mirror'?'VPS 镜像 (真 predict_engine)':null);
    add('实测 λ_obs', w.lambda_obs?w.lambda_obs+' nm':null);
    add('收单备注', w.close_note);
    let tl='';
    (w.log||[]).forEach(l=>{ const t=new Date(l.ts*1000);
      tl+='<div class="wotl"><span class="t">'+t.toLocaleString('zh-CN',{hour12:false})+'</span><b>'+esc(l.author)+'</b><span class="act">'+esc(l.action)+'</span><span>'+esc(l.detail||'')+'</span></div>'; });
    document.getElementById('wodBody').innerHTML=
      '<div class="wod-grid">'+rows+'</div><div class="aw-hist-h">⏱ 全程留痕 (wo_log)</div><div class="wod-log">'+tl+'</div>';
    document.getElementById('woModal').classList.add('show');
    document.body.classList.add('modal-open');
  }catch(e){}
}
function woClose(){ document.getElementById('woModal').classList.remove('show'); document.body.classList.remove('modal-open'); }
function woExport(){ if(_woCur) location.href='/api/workorders/'+_woCur.id+'/export'; }
const _woCustomSel=document.getElementById('woFormula');
if(_woCustomSel) _woCustomSel.addEventListener('change',function(){
  document.getElementById('woCustom').style.display=this.value==='_custom'?'':'none'; });

/* ---- P7: 报表中心 ---- */
async function loadReports(){
  try{
    const d=await fetch('/api/reports',{cache:'no-store'}).then(r=>r.json());
    const list=document.getElementById('repList'); if(!list) return; list.innerHTML='';
    if(!(d.reports||[]).length){
      list.innerHTML='<div class="oe-empty">暂无日报 — historian 跨天后自动生成 (admin 可立即生成今日部分日报)</div>'; return; }
    d.reports.forEach(day=>{
      const row=document.createElement('div'); row.className='ev info'; row.style.cursor='pointer';
      row.innerHTML='<span class="sev"></span><span class="t">'+esc(day)+'</span><span class="msg">运行日报 — 点击查看</span>';
      row.onclick=()=>repView(day);
      list.appendChild(row);
    });
  }catch(e){}
}
async function repView(day){
  try{
    const r=await fetch('/api/reports/'+day,{cache:'no-store'}).then(x=>x.json());
    if(r.error){ toast(r.error); return; }
    document.getElementById('repTitle').textContent='📊 运行日报 · '+day+(r.partial?' (当日进行中, 部分数据)':'');
    let rows='';
    const add=(k,v)=>{ if(v!=null&&v!=='') rows+='<div class="wodr"><span>'+k+'</span><b>'+esc(v)+'</b></div>'; };
    ['lab','car','arm'].forEach(k=>{
      const s=(r.systems||{})[k]||{}; const nm=OPS_META[k].nm;
      add(nm+' UI 可用', s.availability_pct!=null?s.availability_pct+'%':'无样本');
      add(nm+' 真机在线', s.real_pct!=null?s.real_pct+'%':null);
      add(nm+' 平均延迟', s.avg_ms!=null?s.avg_ms+' ms':null);
    });
    const al=r.alarms_raised||{};
    add('当日升警', 'crit '+(al.crit||0)+' · warn '+(al.warn||0)+' · info '+(al.info||0));
    add('事件条数', r.events_n);
    add('工单', '+'+((r.workorders||{}).created||0)+' 建 / '+((r.workorders||{}).closed||0)+' 收');
    if(r.kpi_last) add('KPI 末值', '预测 '+r.kpi_last.predictions+' 条 · CI 覆盖 '+r.kpi_last.ci_coverage_pct+'% · 审计 '+r.kpi_last.audit+' ('+r.kpi_last.source+')');
    add('生成时间', new Date(r.generated_ts*1000).toLocaleString('zh-CN',{hour12:false}));
    document.getElementById('repBody').innerHTML='<div class="wod-grid">'+rows+'</div>';
    document.getElementById('repModal').classList.add('show');
    document.body.classList.add('modal-open');
  }catch(e){}
}
function repClose(){ document.getElementById('repModal').classList.remove('show'); document.body.classList.remove('modal-open'); }
async function repGenNow(){
  try{
    const r=await fetch('/api/reports/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({date:new Date().toISOString().slice(0,10)})});
    if(r.status===403){ toast('需要 admin 角色'); return; }
    if(r.ok){ toast('✓ 日报已生成'); loadReports(); } else toast('生成失败');
  }catch(e){ toast('生成失败'); }
}

/* ---- P6: 管理面板 (admin only) ---- */
async function openAdmin(){
  document.getElementById('adminModal').classList.add('show');
  const body=document.getElementById('admBody');
  body.innerHTML='<div class="oe-empty">加载中…</div>';
  try{
    const [ov,us,lg]=await Promise.all([
      fetch('/api/admin/overview',{cache:'no-store'}).then(r=>r.json()),
      fetch('/api/admin/users',{cache:'no-store'}).then(r=>r.json()),
      fetch('/api/admin/logins?limit=60',{cache:'no-store'}).then(r=>r.json())]);
    if(ov.error){ body.innerHTML='<div class="oe-empty">'+esc(ov.error)+'</div>'; return; }
    const up=fmtDur(ov.uptime_s||0);
    let chips='';
    const t=ov.tables||{};
    chips+='<span class="chip">服务运行 '+up+'</span><span class="chip">historian '+(ov.db_kb||'—')+' KB</span>';
    chips+='<span class="chip">样本 '+(t.samples||0)+'</span><span class="chip">事件 '+(t.events||0)+'</span>';
    chips+='<span class="chip">告警 '+(t.alarms||0)+'</span><span class="chip">工单 '+(t.workorders||0)+'</span>';
    chips+='<span class="chip">维保 '+(t.maintenance||0)+'</span><span class="chip">采样周期 '+(ov.sample_every_s||30)+'s</span>';
    chips+='<span class="chip">样本保留 '+(ov.retain_samples_d||14)+' 天</span>';
    let urows='';
    (us.users||[]).forEach(u=>{ urows+='<div class="wodr"><span>'+esc(u.user)+'</span><b>'+esc(u.role)+' · '+esc(u.name||'')+'</b></div>'; });
    let lrows='';
    (lg.logins||[]).forEach(l=>{
      const tm=new Date(l.ts*1000).toLocaleString('zh-CN',{hour12:false});
      lrows+='<div class="wotl"><span class="t">'+tm+'</span><b>'+esc(l.user)+'</b>'+
        '<span class="act" style="color:'+(l.ok?'#059669':'#dc2626')+'">'+(l.ok?'✓ 成功':'✕ 失败'+(l.role==='locked'?' (锁定)':''))+'</span>'+
        '<span>'+esc(l.ip||'')+(l.ok?' · '+esc(l.role||''):'')+'</span></div>'; });
    if(!lrows) lrows='<div class="oe-empty">暂无登录记录 (审计自 P6 上线起落盘)</div>';
    body.innerHTML='<div class="kpi-spec" style="margin-bottom:14px">'+chips+'</div>'+
      '<div class="aw-hist-h">👥 账号 (users.json, 不含口令散列)</div><div class="wod-grid">'+urows+'</div>'+
      '<div class="aw-hist-h">🔐 登录审计 (logins.jsonl, 最近 60 条)</div><div class="wod-log">'+lrows+'</div>';
  }catch(e){ body.innerHTML='<div class="oe-empty">加载失败</div>'; }
}
function adminClose(){ document.getElementById('adminModal').classList.remove('show'); }

/* ---- 剧场模式 (autoplay theater) ---- */
const TOUR=[
  { sys:'lab', route:'/',         n:'01', t:'AI 脑 · 配方预测', live:'pred', bullets:['输入近红外荧光粉化学式 + 掺杂','9 本地 LLM + 云 R1 并行推理','30 秒出 verdict + 烧结条件 + Conformal CI'] },
  { sys:'car', route:'/livemap',  n:'02', t:'车载脑 · 导航取料', live:'car', bullets:['AI 脑下发取料派单','SLAM 建图 + Nav2 自主导航到试剂柜','视觉伺服 + 电磁铁取瓶带回'] },
  { sys:'arm', route:'/pipeline', n:'03', t:'机械臂 · 研磨灌装', live:'arm', bullets:['双臂工位 v4 十幕剧本接力','接瓶 → 倒粉 → 研磨 → 灌装 → 装车','DH 正运动学 + capsule 防撞互锁'] },
  { sys:'lab', route:'/campaign', n:'04', t:'AI 脑 · 实测回填', live:'kpi', bullets:['烧结/XRD/PL 实测结果回填','GP/EI 闭环主动学习','驱动下一轮配方, 自主实验飞轮'] },
];
/* H1 真联动: 每幕从真实 API 拉一条实时数据填进字幕 (而非纯文案) — 镜像兜底时也是真算/真历史. */
let _thLiveSeq=0;
async function theaterLive(kind){
  const seq=++_thLiveSeq;
  const box=document.getElementById('thLive'), tx=document.getElementById('thLiveTx');
  if(!box||!tx) return;
  box.style.display='flex'; tx.textContent='读取实时数据…'; box.className='tc-live loading';
  let out=null;
  try{
    if(kind==='pred'){
      const d=await fetch('/api/search/index',{cache:'no-store'}).then(r=>r.json());
      const p=(d.predictions||[])[0];
      if(p) out='最新真实预测 · '+p.formula+(p.dopant?(' '+p.dopant):'')+' → '+p.verdict+
        ' ('+(d.source&&d.source.lab==='real'?'真机':'镜像真算')+')';
    } else if(kind==='kpi'){
      const d=await fetch('/api/kpi',{cache:'no-store'}).then(r=>r.json()); const k=d.kpi||{};
      const bits=[];
      if(k.ci_coverage_pct!=null) bits.push('CI 覆盖 '+k.ci_coverage_pct+'%');
      if(k.audit_intact) bits.push('审计链 '+k.audit_valid+'/'+k.audit_total+' 零篡改');
      if(k.predictions!=null) bits.push('累计 '+k.predictions+' 条');
      if(bits.length) out='实测回填闭环 · '+bits.join(' · ');
    } else if(kind==='car'||kind==='arm'){
      const d=await fetch('/api/twin',{cache:'no-store'}).then(r=>r.json());
      const src=(d.source||{})[kind];
      if(kind==='car'&&d.car){ const b=d.car.battery_pct; const v=(d.car.velocity||{});
        out='车载脑遥测 · '+(b!=null?('电量 '+b+'% · '):'')+'位姿在线'+(src==='mirror'?' (镜像演示)':' (真机)'); }
      else if(kind==='arm'){ const a=(d.arms||{}).arm01||(d.arms||{}).arm02;
        if(a&&a.angles) out='双臂关节角 [' + a.angles.slice(0,3).map(x=>x.toFixed(0)).join(', ')+'…] J · '+
          (src==='mirror'?'镜像演示':'真机 FK'); }
      if(!out){ // 设备离线: 退回 serving 态
        const o=await fetch('/api/ops',{cache:'no-store'}).then(r=>r.json());
        const s=(o.systems||{})[kind]||{}; out=SYSLABEL[kind]+' · '+servText(s.serving||'down')+
          (s.serving==='mirror'?' (设备未上电, 镜像兜底)':''); }
    }
  }catch(e){}
  if(seq!==_thLiveSeq) return;   // 已切到下一幕, 丢弃过期结果
  if(out){ tx.textContent=out; box.className='tc-live'; }
  else { box.style.display='none'; }
}
/* 剧场深度联动: 切到某幕后, 给对应系统 iframe 发 scene 消息 → 该系统自动导航到相关页 (mock 镜像也生效).
   带重试 (iframe 可能还在加载), 接收方对同路由幂等。 */
function sendScene(sys, route){
  if(!route) return; const f=frameEls[sys]; if(!f) return;
  if(f.dataset.userEmbed!=='1' || f.dataset.loaded!=='1') return;
  let origin; try{ origin=new URL(SYS[sys].url).origin; }catch(e){ return; }
  const msg={ source:'xrd-cmdcenter', action:'scene', route };
  [0,700,1500].forEach(d=>setTimeout(()=>{ try{ f.contentWindow.postMessage(msg, origin); }catch(e){} }, d));
}
const SYSLABEL={lab:'AI 脑',car:'车载脑',arm:'机械臂'};
let TH={on:false, idx:0, playing:true, start:Infinity, paused:0, pauseAt:0, dur:9500, raf:0, guard:0, enterSeq:0};
function tourStart(){ onbEnd(); TH.on=true; TH.idx=0; TH.playing=true;
  document.getElementById('theater').classList.add('show'); theaterEnter(); }
function buildDots(){
  const c=document.getElementById('thDots'); c.innerHTML='';
  TOUR.forEach((s,i)=>{ const d=document.createElement('div'); d.className='tdot'; d.dataset.i=i;
    d.innerHTML=`<div class="lbl">${s.n} ${s.t.split(' · ')[1]||s.t}</div><div class="track"><div class="f"></div></div>`;
    d.onclick=()=>{ TH.idx=i; theaterEnter(); }; c.appendChild(d); });
}
function theaterStageReady(seq, s, fallback){
  if(seq!==TH.enterSeq || !TH.on || TH.start!==Infinity) return;
  clearTimeout(TH.guard);
  if(fallback){
    booting=false; bootTimers.forEach(clearTimeout); bootTimers=[];
    const ov=document.getElementById('boot'); if(ov) ov.classList.remove('show','out');
    revealView(s.sys);
    toast('演示已切换到本地镜像视图继续', 'info');
  }
  TH.start=performance.now();
  sendScene(s.sys, s.route);
}
function theaterEnter(){
  const s=TOUR[TH.idx];
  const seq=++TH.enterSeq;
  clearTimeout(TH.guard);
  document.getElementById('thN').textContent=s.n;
  document.getElementById('thT').textContent=s.t;
  document.getElementById('thSys').textContent='● '+SYSLABEL[s.sys]+' 视图';
  const ul=document.getElementById('thB'); ul.innerHTML='';
  s.bullets.forEach(b=>{ const li=document.createElement('li'); li.textContent=b; ul.appendChild(li); });
  document.getElementById('thNow').textContent=`第 ${TH.idx+1} / ${TOUR.length} 幕`;
  // dots state
  document.querySelectorAll('#thDots .tdot').forEach((d,i)=>{
    d.className='tdot '+(i<TH.idx?'done':i===TH.idx?'active':'');
    const f=d.querySelector('.f'); if(i<TH.idx) f.style.width='100%'; else if(i>TH.idx) f.style.width='0%';
  });
  document.getElementById('thPrev').disabled = TH.idx===0;
  // H1 真联动: 拉本幕实时数据进字幕
  if(s.live) theaterLive(s.live); else { const lb=document.getElementById('thLive'); if(lb) lb.style.display='none'; }
  // 切系统 (boot 动画), reveal 后才开始计时
  TH.start=Infinity; TH.paused=0; TH.pauseAt=0;
  go(s.sys, {force:true, after:()=>theaterStageReady(seq, s, false)});
  TH.guard=setTimeout(()=>theaterStageReady(seq, s, true), 3800);
  cancelAnimationFrame(TH.raf); TH.raf=requestAnimationFrame(theaterTick);
  updatePlay();
}
function theaterTick(now){
  if(!TH.on) return;
  if(TH.playing && TH.start!==Infinity){
    const el=now-TH.start-TH.paused;
    const pct=Math.max(0,Math.min(1, el/TH.dur));
    const f=document.querySelector('#thDots .tdot[data-i="'+TH.idx+'"] .f');
    if(f) f.style.width=(pct*100)+'%';
    if(pct>=1){ theaterNext(); return; }
  }
  TH.raf=requestAnimationFrame(theaterTick);
}
function theaterNext(){
  if(TH.idx>=TOUR.length-1){ TH.playing=false; updatePlay();
    const f=document.querySelector('#thDots .tdot[data-i="'+TH.idx+'"] .f'); if(f) f.style.width='100%';
    toast('演示完成 ✓ 可点退出或重看'); return; }
  TH.idx++; theaterEnter();
}
function theaterPrev(){ if(TH.idx>0){ TH.idx--; theaterEnter(); } }
function theaterPlayPause(){
  if(TH.playing){ TH.playing=false; TH.pauseAt=performance.now(); }
  else { if(TH.pauseAt) TH.paused += performance.now()-TH.pauseAt; TH.playing=true;
    if(TH.start===Infinity) TH.start=performance.now(); }
  updatePlay();
}
function updatePlay(){ const b=document.getElementById('thPlay');
  b.textContent = TH.playing ? '⏸ 暂停' : '▶ 播放'; }
function tourExit(){ TH.on=false; cancelAnimationFrame(TH.raf);
  clearTimeout(TH.guard); TH.enterSeq++;
  document.getElementById('theater').classList.remove('show'); go('home',{force:true}); }

/* ---- 首次导览 spotlight ---- */
const ONB=[
  { sel:'#nav', title:'🧭 四个视图', body:'在 总览 / 亮点 / AI脑 / 车载脑 / 机械臂 之间切换 (也可用键盘 0/1/2/3)。', pad:8 },
  { sel:'#navHighlight', title:'✨ 技术亮点', body:'5 大 BPU 实测亮点速览, 答辩看这里 —— 每条都附真机实测指标。', pad:8 },
  { sel:'#btnTour', title:'▶ 一条龙演示', body:'一键自动播放端到端剧场: 配方 → 取料 → 研磨灌装 → 回填闭环。', pad:8 },
  { sel:'#cards', title:'🖥 三大系统', body:'真机在线或镜像演示都能进去看功能 —— 设备关机也照常可查看。', pad:12 },
];
let onbI=0;
function maybeOnboard(){
  if((document.body.dataset.view||cur)!=='home') return;
  const sp=new URLSearchParams(location.search||'');
  if(sp.get('onboard')!=='1') return;
  try{ if(localStorage.getItem('cmdcenter.onboarded.v1')==='1') return; }catch(e){}
  onbI=0; document.getElementById('onb').classList.add('show'); onbShow(0);
}
function onbShow(i){
  const s=ONB[i]; const el=document.querySelector(s.sel);
  if(!el){ return (i+1<ONB.length) ? onbShow(i+1) : onbEnd(); }
  const r=el.getBoundingClientRect(); const pad=s.pad||8;
  const ring=document.getElementById('onbRing');
  ring.style.left=(r.left-pad)+'px'; ring.style.top=(r.top-pad)+'px';
  ring.style.width=(r.width+pad*2)+'px'; ring.style.height=(r.height+pad*2)+'px';
  document.getElementById('onbTitle').textContent=s.title;
  document.getElementById('onbBody').textContent=s.body;
  document.getElementById('onbStep').textContent=(i+1)+' / '+ONB.length;
  document.getElementById('onbNext').textContent = (i===ONB.length-1)?'开始探索':'下一步';
  const tip=document.getElementById('onbTip');
  const below = r.bottom+170 < window.innerHeight;
  let top = below ? r.bottom+14 : Math.max(12, r.top-14-tip.offsetHeight);
  let left = Math.min(Math.max(12, r.left), window.innerWidth-tip.offsetWidth-12);
  tip.style.top=top+'px'; tip.style.left=left+'px';
}
function onbNext(){ onbI++; if(onbI>=ONB.length) return onbEnd(); onbShow(onbI); }
function onbEnd(){ document.getElementById('onb').classList.remove('show');
  try{ localStorage.setItem('cmdcenter.onboarded.v1','1'); }catch(e){} }
window.addEventListener('resize', ()=>{ if(document.getElementById('onb').classList.contains('show')) onbShow(onbI); });

/* ---- LG7: subtle multi-layer liquid-glass parallax (background only) ---- */
const LG3={ tx:0, ty:0, sx:0, sy:0, scroll:0, glx:0, gly:0, lx:0, ly:0, raf:0 };
function lg3MotionOff(){
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
function lg3Apply(){
  LG3.raf=0;
  const r=document.documentElement;
  r.style.setProperty('--lg-tx', LG3.tx.toFixed(2)+'px');
  r.style.setProperty('--lg-ty', LG3.ty.toFixed(2)+'px');
  r.style.setProperty('--lg-tx-soft', LG3.sx.toFixed(2)+'px');
  r.style.setProperty('--lg-ty-soft', LG3.sy.toFixed(2)+'px');
  r.style.setProperty('--lg-scroll', LG3.scroll.toFixed(2)+'px');
  r.style.setProperty('--lg-glint-x', LG3.glx.toFixed(2)+'px');
  r.style.setProperty('--lg-glint-y', LG3.gly.toFixed(2)+'px');
  r.style.setProperty('--lg-glint-x-card', (LG3.glx*.45).toFixed(2)+'px');
  r.style.setProperty('--lg-glint-y-card', (LG3.gly*.25).toFixed(2)+'px');
  r.style.setProperty('--lg-glint-x-rev', (LG3.glx*-.55).toFixed(2)+'px');
  r.style.setProperty('--lg-glint-y-rev', (LG3.gly*-.55).toFixed(2)+'px');
  r.style.setProperty('--lg-lens-x', LG3.lx.toFixed(2)+'px');
  r.style.setProperty('--lg-lens-y', LG3.ly.toFixed(2)+'px');
  r.style.setProperty('--lg-lens-x-soft', (LG3.lx*.36).toFixed(2)+'px');
  r.style.setProperty('--lg-lens-y-soft', (LG3.ly*.30).toFixed(2)+'px');
}
function lg3Schedule(){ if(!LG3.raf) LG3.raf=requestAnimationFrame(lg3Apply); }
function lg3Pointer(e){
  if(lg3MotionOff() || window.innerWidth<760) return;
  const dx=(e.clientX/window.innerWidth)-.5, dy=(e.clientY/window.innerHeight)-.5;
  LG3.tx=dx*4; LG3.ty=dy*4; LG3.sx=dx*2.4; LG3.sy=dy*2.4;
  LG3.glx=dx*8; LG3.gly=dy*6; LG3.lx=dx*10; LG3.ly=dy*8; lg3Schedule();
}
function lg3Scroll(){
  if(lg3MotionOff()) return;
  const p=document.querySelector('.page.show');
  const y=p ? Math.max(-3, Math.min(3, p.scrollTop*.006)) : 0;
  LG3.scroll=y; lg3Schedule();
}
window.addEventListener('pointermove', lg3Pointer, {passive:true});
document.addEventListener('scroll', lg3Scroll, true);
window.addEventListener('resize', lg3Scroll, {passive:true});

/* ---- Site9 R10: living XRD background (canvas particles + route-aware spectral crystal wings) ---- */
const LBG_MODE_MAP={
  home:'home', highlight:'home', story:'home', glossary:'home', changelog:'home',
  mq:'mq', preflight:'mq', tasks:'mq',
  lab:'lab', models:'lab', build:'lab', ar:'lab',
  car:'car', fleet:'car',
  arm:'arm',
  status:'status', ops:'status', obs:'status', logs:'status', traces:'status', topo:'status',
  budget:'status', inc:'status', tm:'status', noc:'status', self:'status', oee:'status',
  alert:'status', qms:'status', cmms:'status', sec:'status', release:'status',
  studio:'studio', fsd:'fsd', replay:'replay',
  command:'command', defense:'defense', benchmark:'benchmark',
  assets:'assets', atlas:'assets', detail:'assets', archive:'assets', importw:'assets',
  eln:'assets', sync:'assets', repro:'assets', standards:'assets', cost:'assets', data:'assets',
  twin:'twin'
};
const LBG_MODES={
  home:{a:'#7c3aed',b:'#2563eb',c:'#06b6d4',w:'#f59e0b',k:['crystal','crystal','route','archive'],i:['crystal','spectrum','flask','lattice','chip','route','tray','radar','atlas']},
  mq:{a:'#0891b2',b:'#2563eb',c:'#22d3ee',w:'#f59e0b',k:['batch','batch','crystal','route'],i:['flask','tray','spectrum','route','command','atlas','lattice']},
  lab:{a:'#7c3aed',b:'#2563eb',c:'#a78bfa',w:'#f59e0b',k:['crystal','crystal','ops','archive'],i:['chip','lattice','crystal','spectrum','command','benchmark','atlas']},
  car:{a:'#2563eb',b:'#06b6d4',c:'#34d399',w:'#f59e0b',k:['route','route','ops','crystal'],i:['route','radar','lattice','shield','replay','command','crystal']},
  arm:{a:'#f59e0b',b:'#7c3aed',c:'#fbbf24',w:'#ef4444',k:['arm','arm','route','crystal'],i:['arm','tray','route','flask','shield','replay','crystal']},
  status:{a:'#10b981',b:'#2563eb',c:'#22d3ee',w:'#f59e0b',k:['ops','ops','route','batch'],i:['shield','command','radar','route','benchmark','replay','lattice']},
  benchmark:{a:'#0ea5e9',b:'#7c3aed',c:'#10b981',w:'#f59e0b',k:['ops','archive','crystal','route'],i:['benchmark','radar','shield','atlas','lattice','spectrum','command']},
  assets:{a:'#7c3aed',b:'#0891b2',c:'#22d3ee',w:'#f59e0b',k:['archive','archive','crystal','batch'],i:['atlas','tray','crystal','flask','spectrum','benchmark','lattice']},
  twin:{a:'#06b6d4',b:'#2563eb',c:'#7c3aed',w:'#f59e0b',k:['route','crystal','arm','archive'],i:['route','radar','arm','replay','lattice','command','crystal']},
  studio:{a:'#0f766e',b:'#2563eb',c:'#22d3ee',w:'#f59e0b',k:['batch','ops','crystal','route'],i:['flask','command','chip','lattice','spectrum','benchmark','atlas']},
  fsd:{a:'#2563eb',b:'#0891b2',c:'#10b981',w:'#f59e0b',k:['route','ops','batch','crystal'],i:['radar','route','shield','lattice','replay','command','benchmark']},
  replay:{a:'#f59e0b',b:'#2563eb',c:'#10b981',w:'#ef4444',k:['route','batch','ops','arm'],i:['replay','route','arm','tray','shield','command','radar']},
  command:{a:'#0ea5e9',b:'#2563eb',c:'#10b981',w:'#f59e0b',k:['ops','batch','route','archive'],i:['command','tray','route','shield','replay','benchmark','atlas']},
  defense:{a:'#7c3aed',b:'#2563eb',c:'#10b981',w:'#f59e0b',k:['ops','archive','crystal','route'],i:['shield','benchmark','command','atlas','lattice','spectrum','radar']}
};
function lbgModeFor(k){ return LBG_MODE_MAP[k]||LBG_MODE_MAP[cur]||'home'; }
function lbgHex(hex){
  const h=String(hex||'#7c3aed').replace('#','');
  const n=parseInt(h.length===3?h.split('').map(x=>x+x).join(''):h,16);
  return {r:(n>>16)&255,g:(n>>8)&255,b:n&255};
}
function lbgRgba(hex,a){ const c=lbgHex(hex); return 'rgba('+c.r+','+c.g+','+c.b+','+a+')'; }
function lbgMix(a,b,t){
  const A=lbgHex(a), B=lbgHex(b), m=1-t;
  const r=Math.round(A.r*m+B.r*t), g=Math.round(A.g*m+B.g*t), bl=Math.round(A.b*m+B.b*t);
  return 'rgb('+r+','+g+','+bl+')';
}
function lbgRand(seed){
  let s=seed>>>0;
  return function(){ s=(s*1664525+1013904223)>>>0; return s/4294967296; };
}
function lbgSeed(s){ let h=2166136261; for(let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619); } return h>>>0; }
function initLivingBackground(){
  const host=document.getElementById('livingBg'), field=document.getElementById('livingWingfield');
  const symField=document.getElementById('livingSymbolfield');
  const cv=document.getElementById('livingCanvas');
  if(!host||!field||!cv) return;
  const ctx=cv.getContext && cv.getContext('2d');
  const motion=window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : {matches:false, addEventListener:null};
  const S={mode:'home', wings:[], symbols:[], parts:[], raf:0, w:0, h:0, dpr:1, hidden:false, reduce:!!motion.matches, last:0};
  function makeWings(){
    if(S.wings.length) return;
    for(let i=0;i<32;i++){
      const el=document.createElement('div');
      el.className='lb-wing';
      el.innerHTML='<span class="lb-core"></span>';
      field.appendChild(el);
      S.wings.push(el);
    }
  }
  function lbgIconSvg(kind){
    const head='<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">';
    const tail='</svg>';
    const plate='<rect class="lb-plate" x="14" y="14" width="72" height="72" rx="19"/>';
    if(kind==='spectrum') return head+plate+'<path class="lb-line" d="M16 70c16-4 18-30 34-28s16 31 34 20"/><path class="lb-line thin" d="M18 82h64M18 82V28"/><circle class="lb-dot" cx="51" cy="42" r="4"/>'+tail;
    if(kind==='flask') return head+plate+'<path class="lb-line" d="M40 18h20M47 18v22L29 74c-5 9 2 16 12 16h18c10 0 17-7 12-16L53 40V18"/><path class="lb-line thin" d="M35 70c10 7 23-7 34 2"/>'+tail;
    if(kind==='lattice') return head+plate+'<path class="lb-line thin" d="M28 30h28l16 24-16 24H28L12 54zM28 30l28 48M56 30L28 78M12 54h60"/><circle class="lb-dot" cx="28" cy="30" r="4"/><circle class="lb-dot" cx="56" cy="30" r="4"/><circle class="lb-dot" cx="72" cy="54" r="4"/><circle class="lb-dot" cx="56" cy="78" r="4"/><circle class="lb-dot" cx="28" cy="78" r="4"/><circle class="lb-dot" cx="12" cy="54" r="4"/>'+tail;
    if(kind==='chip') return head+plate+'<rect class="lb-line" x="31" y="31" width="38" height="38" rx="8"/><path class="lb-line thin" d="M24 36h7M24 50h7M24 64h7M69 36h7M69 50h7M69 64h7M36 24v7M50 24v7M64 24v7M36 69v7M50 69v7M64 69v7"/><circle class="lb-dot" cx="50" cy="50" r="8"/>'+tail;
    if(kind==='route') return head+plate+'<path class="lb-line" d="M18 72c16-30 32 10 48-20 8-15 12-24 24-28"/><circle class="lb-dot" cx="18" cy="72" r="6"/><circle class="lb-dot" cx="53" cy="55" r="6"/><circle class="lb-dot" cx="90" cy="24" r="6"/>'+tail;
    if(kind==='arm') return head+plate+'<path class="lb-line" d="M23 76h20M34 76V55l22-17 22 9"/><circle class="lb-line thin" cx="34" cy="55" r="10"/><circle class="lb-line thin" cx="56" cy="38" r="9"/><circle class="lb-line thin" cx="78" cy="47" r="8"/><path class="lb-line thin" d="M78 47l12-16M78 47l15 6M78 47l5 16"/>'+tail;
    if(kind==='tray') return head+plate+'<rect class="lb-line" x="22" y="28" width="56" height="44" rx="9"/><path class="lb-line thin" d="M40 28v44M60 28v44M22 50h56"/><circle class="lb-dot" cx="31" cy="39" r="4"/><circle class="lb-dot" cx="50" cy="61" r="4"/><circle class="lb-dot" cx="69" cy="39" r="4"/>'+tail;
    if(kind==='shield') return head+plate+'<path class="lb-line" d="M50 18l29 12v19c0 20-12 34-29 43-17-9-29-23-29-43V30z"/><path class="lb-line thin" d="M35 52l11 11 22-25"/>'+tail;
    if(kind==='radar') return head+plate+'<circle class="lb-line thin" cx="50" cy="52" r="31"/><circle class="lb-line thin" cx="50" cy="52" r="18"/><path class="lb-line" d="M50 52l31-18"/><path class="lb-line thin" d="M18 52h64M50 20v64"/><circle class="lb-dot" cx="65" cy="35" r="5"/>'+tail;
    if(kind==='replay') return head+plate+'<path class="lb-line" d="M28 38a29 29 0 1 1-2 27"/><path class="lb-fill" d="M24 26l3 25 19-15z"/><path class="lb-line thin" d="M44 39v23l20-12z"/>'+tail;
    if(kind==='command') return head+plate+'<rect class="lb-line" x="22" y="23" width="56" height="54" rx="9"/><path class="lb-line thin" d="M32 38h34M32 52h24M32 66h32"/><circle class="lb-dot" cx="70" cy="38" r="4"/>'+tail;
    if(kind==='atlas') return head+plate+'<rect class="lb-line thin" x="26" y="22" width="42" height="50" rx="7"/><rect class="lb-line thin" x="33" y="28" width="42" height="50" rx="7"/><path class="lb-line thin" d="M43 42h19M43 55h21M43 68h13"/><circle class="lb-dot" cx="38" cy="35" r="4"/>'+tail;
    if(kind==='benchmark') return head+plate+'<path class="lb-line thin" d="M25 68a30 30 0 0 1 50 0"/><path class="lb-line" d="M50 68l18-24"/><path class="lb-line thin" d="M30 68h40M31 53l8 5M69 53l-8 5M50 38v10"/><circle class="lb-dot" cx="50" cy="68" r="5"/>'+tail;
    return head+plate+'<polygon class="lb-fill" points="50,14 78,30 78,62 50,86 22,62 22,30"/><path class="lb-line thin" d="M50 14v72M22 30l56 32M78 30L22 62M35 38h30v24H35z"/>'+tail;
  }
  function makeSymbols(){
    if(!symField || S.symbols.length) return;
    for(let i=0;i<22;i++){
      const el=document.createElement('div');
      el.className='lb-symbol';
      symField.appendChild(el);
      S.symbols.push(el);
    }
  }
  function modeCfg(mode){ return LBG_MODES[mode]||LBG_MODES.home; }
  function setCssMode(mode){
    const c=modeCfg(mode);
    host.dataset.mode=mode;
    host.style.setProperty('--lb-a',c.a);
    host.style.setProperty('--lb-b',c.b);
    host.style.setProperty('--lb-c',c.c);
    host.style.setProperty('--lb-warm',c.w);
  }
  function wingPoint(mode,i,r){
    const near=i%7===0, far=i%5===0&&!near;
    let x,y,size,rot;
    if(mode==='home'){
      x=5+r()*92; y=8+r()*84; size=far?58+r()*42:near?146+r()*68:86+r()*64; rot=-34+r()*68;
      if(i<5){ x=[8,18,72,86,48][i]+r()*5; y=[22,68,16,62,86][i]+r()*5; size=[156,116,136,168,86][i]; }
    }else if(mode==='mq'){
      x=12+r()*82; y=14+r()*72; size=far?62+r()*35:90+r()*62; rot=-18+r()*36;
      if(i<7){ x=14+i*12+r()*4; y=22+(i%2)*36+r()*7; size=82+r()*34; }
    }else if(mode==='lab'){
      x=42+r()*54; y=8+r()*78; size=far?54+r()*36:96+r()*72; rot=-42+r()*84;
      if(i<6){ x=52+r()*40; y=12+i*12+r()*4; size=118-r()*20; }
    }else if(mode==='car'){
      x=6+r()*88; y=30+r()*58; size=far?56+r()*28:86+r()*52; rot=-8+r()*28;
      if(i<8){ x=9+i*11+r()*3; y=70-28*Math.sin(i*.74)+r()*4; size=74+r()*28; }
    }else if(mode==='arm'){
      x=18+r()*74; y=12+r()*76; size=far?58+r()*34:92+r()*64; rot=-55+r()*95;
      if(i<6){ x=52+r()*40; y=24+i*10+r()*6; size=104+r()*34; }
    }else if(mode==='assets'){
      x=8+r()*86; y=12+r()*78; size=far?52+r()*34:82+r()*56; rot=-22+r()*44;
      if(i<8){ x=12+(i%4)*22+r()*3; y=22+Math.floor(i/4)*38+r()*5; size=78+r()*24; }
    }else if(mode==='twin'){
      x=7+r()*88; y=14+r()*76; size=far?56+r()*32:86+r()*62; rot=-26+r()*52;
      if(i<7){ x=14+i*12+r()*4; y=78-i*7+r()*5; size=82+r()*24; }
    }else{
      x=10+r()*84; y=10+r()*78; size=far?52+r()*34:82+r()*56; rot=-28+r()*56;
      if(i<7){ x=18+i*11+r()*4; y=28+(i%3)*18+r()*5; size=76+r()*26; }
    }
    return {x,y,size,rot,near,far};
  }
  function placeWings(mode){
    makeWings();
    const cfg=modeCfg(mode), r=lbgRand(lbgSeed('lbg-'+mode)), mobile=window.innerWidth<760;
    const max=mobile?14:30;
    S.wings.forEach((w,i)=>{
      const p=wingPoint(mode,i,r), kind=cfg.k[i%cfg.k.length];
      w.dataset.kind=kind;
      w.dataset.depth=p.near?'near':p.far?'far':'mid';
      w.style.display=i<max?'':'none';
      w.style.setProperty('--lb-x',p.x.toFixed(2)+'%');
      w.style.setProperty('--lb-y',p.y.toFixed(2)+'%');
      w.style.setProperty('--lb-size',p.size.toFixed(1)+'px');
      w.style.setProperty('--lb-rot',p.rot.toFixed(2)+'deg');
      w.style.setProperty('--lb-scale',(p.near?1.08:p.far?.82:1).toFixed(2));
      w.style.setProperty('--lb-o',(p.near?.66:p.far?.34:.52).toFixed(2));
      w.style.setProperty('--lb-delay',(-r()*9).toFixed(2)+'s');
      w.style.setProperty('--lb-dur',(11+r()*9).toFixed(2)+'s');
      w.style.setProperty('--lb-flap',(3.6+r()*2.6).toFixed(2)+'s');
    });
  }
  function symbolPoint(mode,i,r){
    const near=i%9===0, far=i%5===0&&!near;
    let x,y,size,rot;
    const home=[[10,18,118,-18],[18,72,94,22],[82,18,112,16],[88,64,132,-12],[48,86,82,10],[64,34,74,-24]];
    const lab=[[75,18,104,10],[88,38,82,-18],[70,72,96,20],[42,16,76,-8],[16,66,82,24]];
    const car=[[12,70,98,12],[33,58,82,-8],[58,72,92,18],[82,36,106,-14],[72,16,72,20]];
    const status=[[18,22,92,-10],[42,18,76,16],[72,24,98,-16],[86,64,86,10],[28,78,78,-22]];
    const assets=[[12,26,86,8],[28,68,96,-12],[72,18,84,18],[86,62,112,-10],[54,82,76,22]];
    const map={home,lab,studio:lab,car,fsd:car,twin:car,arm:[[72,20,90,-16],[86,46,112,18],[68,76,94,-12],[22,72,82,24],[44,24,74,-20]],status,command:status,defense:status,benchmark:status,assets};
    const a=map[mode]||home;
    if(i<a.length){
      x=a[i][0]+(r()-.5)*5; y=a[i][1]+(r()-.5)*5; size=a[i][2]; rot=a[i][3];
    }else{
      const edge=i%4;
      if(edge===0){ x=6+r()*28; y=12+r()*72; }
      else if(edge===1){ x=66+r()*28; y=10+r()*74; }
      else if(edge===2){ x=18+r()*64; y=8+r()*24; }
      else { x=14+r()*70; y=70+r()*18; }
      size=far?54+r()*24:near?112+r()*34:68+r()*42;
      rot=-30+r()*60;
    }
    return {x,y,size,rot,near,far};
  }
  function placeSymbols(mode){
    makeSymbols();
    if(!S.symbols.length) return;
    const cfg=modeCfg(mode), r=lbgRand(lbgSeed('sym-'+mode)), mobile=window.innerWidth<760;
    const icons=cfg.i||LBG_MODES.home.i, colors=[cfg.a,cfg.b,cfg.c,cfg.w,'#e11d48','#10b981'];
    const max=mobile?9:20;
    S.symbols.forEach((el,i)=>{
      const kind=icons[i%icons.length], p=symbolPoint(mode,i,r);
      if(el.dataset.kind!==kind) el.innerHTML=lbgIconSvg(kind);
      el.dataset.kind=kind;
      el.dataset.depth=p.near?'near':p.far?'far':'mid';
      el.style.display=i<max?'':'none';
      el.style.setProperty('--ls-x',p.x.toFixed(2)+'%');
      el.style.setProperty('--ls-y',p.y.toFixed(2)+'%');
      el.style.setProperty('--ls-size',p.size.toFixed(1)+'px');
      el.style.setProperty('--ls-rot',p.rot.toFixed(2)+'deg');
      el.style.setProperty('--ls-scale',(p.near?1.08:p.far?.82:1).toFixed(2));
      el.style.setProperty('--ls-o',(p.near?.58:p.far?.24:.42).toFixed(2));
      el.style.setProperty('--ls-color',colors[i%colors.length]);
      el.style.setProperty('--ls-delay',(-r()*12).toFixed(2)+'s');
      el.style.setProperty('--ls-dur',(13+r()*10).toFixed(2)+'s');
    });
  }
  function resize(){
    S.dpr=Math.min(window.devicePixelRatio||1,2);
    S.w=window.innerWidth||1200; S.h=window.innerHeight||760;
    cv.width=Math.max(1,Math.round(S.w*S.dpr)); cv.height=Math.max(1,Math.round(S.h*S.dpr));
    cv.style.width=S.w+'px'; cv.style.height=S.h+'px';
    if(ctx) ctx.setTransform(S.dpr,0,0,S.dpr,0,0);
    buildParticles();
    placeWings(S.mode);
    placeSymbols(S.mode);
    draw(performance.now());
  }
  function buildParticles(){
    const r=lbgRand(lbgSeed('p-'+S.mode+'-'+Math.round(S.w/100)+'-'+Math.round(S.h/100)));
    const n=S.reduce?24:(S.w<760?48:118);
    S.parts=[];
    const cfg=modeCfg(S.mode);
    for(let i=0;i<n;i++){
      S.parts.push({
        x:r()*S.w, y:r()*S.h, z:.45+r()*1.15, vx:(r()-.5)*(.12+r()*.22), vy:(r()-.5)*(.10+r()*.16),
        r:1.2+r()*3.3, rot:r()*Math.PI*2, spin:(r()-.5)*.018,
        c:[cfg.a,cfg.b,cfg.c,cfg.w][i%4]
      });
    }
  }
  function poly(ctx,x,y,r,n,rot){
    ctx.beginPath();
    for(let i=0;i<n;i++){
      const a=rot+i*Math.PI*2/n;
      const px=x+Math.cos(a)*r, py=y+Math.sin(a)*r;
      if(i) ctx.lineTo(px,py); else ctx.moveTo(px,py);
    }
    ctx.closePath();
  }
  function ribbon(ctx,t,c1,c2,shift,amp){
    const y=S.h*(.42+shift), h=S.h, w=S.w;
    const g=ctx.createLinearGradient(0,0,w,0); g.addColorStop(0,lbgRgba(c1,.02)); g.addColorStop(.5,lbgRgba(c2,.34)); g.addColorStop(1,lbgRgba(c1,.02));
    ctx.save();
    ctx.lineWidth=Math.max(6,w*.006); ctx.lineCap='round'; ctx.strokeStyle=g; ctx.globalAlpha=.68;
    ctx.beginPath();
    ctx.moveTo(-80,y);
    ctx.bezierCurveTo(w*.18,y-h*amp*Math.sin(t*.00034+shift), w*.30,y+h*amp, w*.47,y-h*amp*.35);
    ctx.bezierCurveTo(w*.62,y-h*amp, w*.78,y+h*amp*Math.cos(t*.00028+shift), w+80,y+h*amp*.2);
    ctx.stroke();
    ctx.lineWidth=2; ctx.globalAlpha=.38; ctx.setLineDash([42,28]); ctx.lineDashOffset=-t*.018;
    ctx.stroke();
    ctx.restore();
  }
  function rings(ctx,t,cfg){
    ctx.save();
    ctx.lineWidth=1.2;
    const spots=S.mode==='home'?[[S.w*.88,S.h*.10,120],[S.w*.08,S.h*.92,110]]:[[S.w*.78,S.h*.32,90],[S.w*.18,S.h*.76,76]];
    spots.forEach((s,idx)=>{
      for(let i=0;i<5;i++){
        ctx.beginPath();
        ctx.strokeStyle=lbgRgba([cfg.a,cfg.b,cfg.c,cfg.w][(i+idx)%4], .12-i*.015);
        ctx.arc(s[0],s[1],s[2]+i*54+Math.sin(t*.00045+i)*2,0,Math.PI*2);
        ctx.stroke();
      }
    });
    ctx.restore();
  }
  function modeOverlay(ctx,t,cfg){
    ctx.save();
    ctx.globalAlpha=.34;
    ctx.lineCap='round';
    if(S.mode==='mq'){
      ctx.strokeStyle=lbgRgba(cfg.c,.32); ctx.lineWidth=5;
      for(let i=0;i<6;i++){ const y=S.h*(.18+i*.10); ctx.beginPath(); ctx.moveTo(S.w*.06,y); ctx.lineTo(S.w*(.36+.05*Math.sin(i)),y+Math.sin(t*.001+i)*8); ctx.stroke(); }
      ctx.strokeStyle=lbgRgba(cfg.b,.24); ctx.lineWidth=3; ctx.strokeRect(S.w*.69,S.h*.18,S.w*.22,S.h*.34);
    }else if(S.mode==='lab'||S.mode==='studio'){
      ctx.strokeStyle=lbgRgba(cfg.a,.25); ctx.lineWidth=2;
      const cx=S.w*.76, cy=S.h*.34, step=Math.min(S.w,S.h)*.055;
      for(let i=-3;i<=3;i++) for(let j=-2;j<=2;j++){
        const x=cx+i*step+j*step*.5, y=cy+j*step*.86;
        ctx.beginPath(); ctx.arc(x,y,3.2,0,Math.PI*2); ctx.stroke();
        if(i<3){ ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x+step,y); ctx.stroke(); }
      }
    }else if(S.mode==='car'||S.mode==='twin'||S.mode==='fsd'){
      ctx.strokeStyle=lbgRgba(cfg.c,.38); ctx.lineWidth=4; ctx.setLineDash([22,18]); ctx.lineDashOffset=-t*.025;
      ctx.beginPath(); ctx.moveTo(S.w*.07,S.h*.70); ctx.bezierCurveTo(S.w*.26,S.h*.55,S.w*.36,S.h*.88,S.w*.53,S.h*.66); ctx.bezierCurveTo(S.w*.68,S.h*.48,S.w*.79,S.h*.58,S.w*.92,S.h*.30); ctx.stroke();
      ctx.setLineDash([]);
    }else if(S.mode==='arm'||S.mode==='replay'){
      ctx.strokeStyle=lbgRgba(cfg.w,.28); ctx.lineWidth=2.5;
      [[.76,.44,80],[.70,.54,120],[.82,.62,150]].forEach(a=>{ ctx.beginPath(); ctx.arc(S.w*a[0],S.h*a[1],a[2],-.7,Math.PI*.92); ctx.stroke(); });
    }else if(S.mode==='assets'||S.mode==='benchmark'){
      ctx.strokeStyle=lbgRgba(cfg.b,.16); ctx.lineWidth=1;
      const step=42; for(let x=S.w*.05;x<S.w*.48;x+=step){ ctx.beginPath(); ctx.moveTo(x,S.h*.18); ctx.lineTo(x,S.h*.76); ctx.stroke(); }
      for(let y=S.h*.18;y<S.h*.76;y+=step){ ctx.beginPath(); ctx.moveTo(S.w*.05,y); ctx.lineTo(S.w*.48,y); ctx.stroke(); }
    }else{
      ctx.strokeStyle=lbgRgba(cfg.c,.26); ctx.lineWidth=2;
      const pts=[[.16,.42],[.38,.28],[.58,.44],[.78,.25],[.86,.58]];
      pts.forEach((p,i)=>{ const x=S.w*p[0], y=S.h*p[1]; ctx.beginPath(); ctx.arc(x,y,5+i%2*2,0,Math.PI*2); ctx.stroke(); if(i){ const q=pts[i-1]; ctx.beginPath(); ctx.moveTo(S.w*q[0],S.h*q[1]); ctx.lineTo(x,y); ctx.stroke(); } });
    }
    ctx.restore();
  }
  function particles(ctx,t,cfg){
    ctx.save();
    S.parts.forEach((p,i)=>{
      if(!S.reduce){
        p.x+=p.vx*p.z; p.y+=p.vy*p.z; p.rot+=p.spin;
        if(p.x<-30) p.x=S.w+30; if(p.x>S.w+30) p.x=-30;
        if(p.y<-30) p.y=S.h+30; if(p.y>S.h+30) p.y=-30;
      }
      ctx.globalAlpha=.16+.18*p.z;
      ctx.fillStyle=lbgMix(p.c,'#ffffff',.10);
      poly(ctx,p.x,p.y,p.r*p.z,6,p.rot+t*.0002);
      ctx.fill();
      if(i%9===0){
        ctx.globalAlpha=.15;
        ctx.strokeStyle=lbgRgba(p.c,.25);
        ctx.lineWidth=1;
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r*5,0,Math.PI*2); ctx.stroke();
      }
    });
    ctx.globalAlpha=.11;
    ctx.lineWidth=1;
    for(let i=0;i<S.parts.length;i+=2){
      const a=S.parts[i];
      for(let j=i+1;j<Math.min(S.parts.length,i+18);j+=3){
        const b=S.parts[j], dx=a.x-b.x, dy=a.y-b.y, d=dx*dx+dy*dy;
        if(d<12000){ ctx.strokeStyle=lbgRgba(cfg.c, Math.max(0,.18-d/90000)); ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); }
      }
    }
    ctx.restore();
  }
  function draw(t){
    if(!ctx) return;
    const cfg=modeCfg(S.mode);
    ctx.clearRect(0,0,S.w,S.h);
    ribbon(ctx,t,cfg.a,cfg.c,-.18,.09);
    ribbon(ctx,t,cfg.w,cfg.b,.22,.075);
    rings(ctx,t,cfg);
    modeOverlay(ctx,t,cfg);
    particles(ctx,t,cfg);
  }
  function loop(t){
    S.raf=0;
    if(S.hidden) return;
    draw(t);
    if(!S.reduce) S.raf=requestAnimationFrame(loop);
  }
  function kick(){ if(!S.raf && !S.hidden) S.raf=requestAnimationFrame(loop); }
  window.livingSetView=function(k){
    const mode=lbgModeFor(k);
    if(mode===S.mode){ setCssMode(mode); return; }
    S.mode=mode;
    setCssMode(mode);
    placeWings(mode);
    placeSymbols(mode);
    buildParticles();
    draw(performance.now());
    kick();
  };
  document.addEventListener('visibilitychange',()=>{ S.hidden=document.hidden; if(!S.hidden) kick(); });
  if(motion.addEventListener) motion.addEventListener('change',e=>{ S.reduce=!!e.matches; buildParticles(); draw(performance.now()); kick(); });
  window.addEventListener('resize',resize,{passive:true});
  makeWings();
  makeSymbols();
  setCssMode(S.mode);
  resize();
  kick();
}
initLivingBackground();

/* ---- 角色 / 落地 ---- */
async function boot0(){
  const pathKey=location.pathname.replace(/^\/+|\/+$/g,'');
  const explicit=!!(pathKey||location.search);
  let land=detailRouteFromLocation('home');
  if(!PAGES[land]) ensureFrame(land);
  runBoot(land, ()=>{ revealView(land); maybeOnboard(); try{ applyI18n(i18nBootLang()); }catch(e){} });
  fetch('/api/me',{cache:'no-store'}).then(r=>r.json()).then(me=>{
    window._role=me.role;
    if(me.is_judge){
      document.getElementById('roleBadge').classList.add('show');
      if(!explicit && cur==='home') go('highlight',{force:true});
    }
    if(me.is_admin){ const b=document.getElementById('mmAdmin'); if(b) b.style.display='';
      const g=document.getElementById('repGenBtn'); if(g) g.style.display=''; }
  }).catch(()=>{});
}
buildDots();
boot0();

/* ---- U: 返回顶部 (每个 .page 内部独立滚动, 用捕获监听) ---- */
function totopGo(){ const p=document.querySelector('.page.show'); if(p) p.scrollTo({top:0,behavior:'smooth'}); }
(function(){
  const stage=document.querySelector('.stage');
  function upd(){ const p=document.querySelector('.page.show'); const b=document.getElementById('totop');
    if(p&&b) b.classList.toggle('show', p.scrollTop>360); }
  if(stage) stage.addEventListener('scroll', upd, true);  // capture: 子页滚动不冒泡
  document.addEventListener('click', ()=>setTimeout(upd,60));  // 切页后重判
  window.addEventListener('resize', upd);
})();

/* ---- 真实 3D 三机场景 (Three.js 自托管, 加载/构建失败回落 SVG) ---- */
function init3D(){
  try{
    const THREE=window.THREE; if(!THREE) return;
    const host=document.querySelector('.hero-scene'), cv=document.getElementById('c3d');
    if(!host||!cv) return;
    let W=host.clientWidth||560, H=Math.round(W*0.62);
    const renderer=new THREE.WebGLRenderer({canvas:cv, antialias:true, alpha:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2)); renderer.setSize(W,H);
    renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    const scene=new THREE.Scene();
    const cam=new THREE.PerspectiveCamera(40, W/H, 0.1, 100); cam.position.set(0,5.8,13.5); cam.lookAt(0,0.5,0);
    scene.add(new THREE.AmbientLight(0xffffff,0.66));
    const key=new THREE.DirectionalLight(0xffffff,0.95); key.position.set(7,13,8); key.castShadow=true;
    key.shadow.mapSize.set(1024,1024); const scam=key.shadow.camera; scam.left=-14;scam.right=14;scam.top=14;scam.bottom=-14; scene.add(key);
    const pV=new THREE.PointLight(0x7c3aed,0.5,40); pV.position.set(-8,6,5); scene.add(pV);
    const pC=new THREE.PointLight(0x06b6d4,0.5,40); pC.position.set(8,6,5); scene.add(pC);
    const ground=new THREE.Mesh(new THREE.PlaneGeometry(50,50), new THREE.MeshStandardMaterial({color:0xeef2fb,roughness:0.96,metalness:0}));
    ground.rotation.x=-Math.PI/2; ground.position.y=-1.45; ground.receiveShadow=true; scene.add(ground);
    const grid=new THREE.GridHelper(40,40,0xc7d2fe,0xe6ebf7); grid.position.y=-1.44; scene.add(grid);
    scene.fog=new THREE.Fog(0xeef2fb, 17, 34);   // U3: 远景柔化, 增景深
    // U3: 三机正下方各一团品牌色辉光圆盘 (canvas 径向渐变贴图, 柔和不刺眼)
    (function(){
      function glowTex(hex){ const c=document.createElement('canvas'); c.width=c.height=128;
        const g=c.getContext('2d'), r=g.createRadialGradient(64,64,0,64,64,64);
        g.fillStyle=r; r.addColorStop(0,hex+'cc'); r.addColorStop(.45,hex+'55'); r.addColorStop(1,hex+'00');
        g.fillStyle=r; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c); }
      [['#7c3aed',-5.2],['#06b6d4',0],['#d97706',5.0]].forEach(([hex,x])=>{
        const m=new THREE.Mesh(new THREE.PlaneGeometry(5.2,5.2),
          new THREE.MeshBasicMaterial({map:glowTex(hex),transparent:true,opacity:.5,depthWrite:false}));
        m.rotation.x=-Math.PI/2; m.position.set(x,-1.43,0.4); scene.add(m);
      });
    })();
    const root=new THREE.Group(); scene.add(root);
    const M=(c,m,r)=>new THREE.MeshStandardMaterial({color:c,metalness:m==null?0.35:m,roughness:r==null?0.45:r});
    // AI 脑 服务器
    const brain=new THREE.Group(); brain.position.set(-5.2,0,0);
    const rack=new THREE.Mesh(new THREE.BoxGeometry(2.0,3.0,1.6),M(0x7c3aed,0.4,0.4)); rack.position.y=0.05; rack.castShadow=true; brain.add(rack);
    const core=new THREE.Mesh(new THREE.IcosahedronGeometry(0.52,0), new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xa78bfa,emissiveIntensity:1.1,metalness:0.2,roughness:0.3}));
    core.position.set(0,0.3,0.95); brain.add(core);
    const cL=new THREE.PointLight(0xa78bfa,0.8,9); cL.position.set(0,0.3,1.3); brain.add(cL); root.add(brain);
    // 车
    const car=new THREE.Group(); car.position.set(0,-0.55,1.6);
    const body=new THREE.Mesh(new THREE.BoxGeometry(3.2,0.85,1.9),M(0x2563eb)); body.position.y=0.5; body.castShadow=true; car.add(body);
    const cabin=new THREE.Mesh(new THREE.BoxGeometry(1.8,0.8,1.6),M(0x06b6d4,0.3,0.4)); cabin.position.set(-0.15,1.2,0); cabin.castShadow=true; car.add(cabin);
    const wg=new THREE.CylinderGeometry(0.45,0.45,0.34,20); wg.rotateX(Math.PI/2); const wm=M(0x1f2937,0.5,0.6); const wheels=[];
    [[-1.05,0,1.0],[1.05,0,1.0],[-1.05,0,-1.0],[1.05,0,-1.0]].forEach(p=>{ const w=new THREE.Mesh(wg,wm); w.position.set(p[0],p[1],p[2]); w.castShadow=true; car.add(w); wheels.push(w); });
    const lidar=new THREE.Mesh(new THREE.CylinderGeometry(0.3,0.3,0.28,18),M(0x334155,0.4,0.4)); lidar.position.set(-0.15,1.85,0); car.add(lidar);
    const sweepGeo=new THREE.CircleGeometry(1.25,28,0,Math.PI/3); sweepGeo.rotateX(-Math.PI/2);
    const sweep=new THREE.Mesh(sweepGeo,new THREE.MeshBasicMaterial({color:0x06b6d4,transparent:true,opacity:0.22,side:THREE.DoubleSide})); sweep.position.set(-0.15,1.75,0); car.add(sweep); root.add(car);
    // 臂
    const arm=new THREE.Group(); arm.position.set(5.0,-1.45,0.6);
    const abase=new THREE.Mesh(new THREE.CylinderGeometry(0.7,0.85,0.5,24),M(0xd97706)); abase.position.y=0.25; abase.castShadow=true; arm.add(abase);
    const shoulder=new THREE.Group(); shoulder.position.y=0.5; arm.add(shoulder);
    const up=new THREE.Mesh(new THREE.BoxGeometry(0.46,1.7,0.46),M(0xf59e0b)); up.position.y=0.85; up.castShadow=true; shoulder.add(up);
    const elbow=new THREE.Group(); elbow.position.y=1.7; shoulder.add(elbow);
    const fore=new THREE.Mesh(new THREE.BoxGeometry(0.38,1.4,0.38),M(0xd97706)); fore.position.y=0.7; fore.castShadow=true; elbow.add(fore);
    const wrist=new THREE.Group(); wrist.position.y=1.4; elbow.add(wrist);
    const g1=new THREE.Mesh(new THREE.BoxGeometry(0.12,0.42,0.12),M(0xb45309)); g1.position.set(0.13,0.22,0); wrist.add(g1);
    const g2=new THREE.Mesh(new THREE.BoxGeometry(0.12,0.42,0.12),M(0xb45309)); g2.position.set(-0.13,0.22,0); wrist.add(g2); root.add(arm);
    // 交互: 拖动旋转 + 闲时自转
    let tRY=-0.25,cRY=-0.25,tRX=0,cRX=0,drag=false,lx=0,ly=0,idle=0;
    // P4 资产联动: 聚焦某台设备 (转到正面 + 脉冲高亮)
    const FOCUS={lab:{g:brain,ry:Math.PI/2},car:{g:car,ry:0},arm:{g:arm,ry:-1.45}};
    let pulseG=null,pulseT=0;
    window.focus3D=function(k){
      const f=FOCUS[k]; if(!f) return;
      tRY=f.ry; tRX=0.08; idle=-8; pulseG=f.g; pulseT=1.6;
      toast('⬢ 3D 聚焦: '+({lab:'AI 脑',car:'车载脑',arm:'机械臂'})[k]);
    };
    cv.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;idle=0;cv.style.cursor='grabbing';});
    window.addEventListener('pointerup',()=>{drag=false;cv.style.cursor='grab';});
    window.addEventListener('pointermove',e=>{ if(!drag)return; tRY+=(e.clientX-lx)*0.01; tRX=Math.max(-0.45,Math.min(0.6,tRX+(e.clientY-ly)*0.005)); lx=e.clientX;ly=e.clientY;idle=0; });
    function resize(){ const w=host.clientWidth||W,h=Math.round(w*0.62); renderer.setSize(w,h); cam.aspect=w/h; cam.updateProjectionMatrix(); }
    window.addEventListener('resize',resize);
    document.getElementById('scene2d').style.display='none'; cv.style.display='block';
    const hint=document.createElement('div'); hint.textContent='🖱 拖动旋转 · 真实 3D';
    hint.style.cssText='position:absolute;left:12px;bottom:12px;font-size:.66rem;font-weight:700;color:#475569;background:rgba(255,255,255,.82);border:1px solid rgba(15,23,42,.1);border-radius:999px;padding:4px 11px;backdrop-filter:blur(6px);pointer-events:none;';
    host.style.position='relative'; host.appendChild(hint);
    // 真实 GLTF 高模切换 (懒加载 GLTFLoader + RobotExpressive.glb, CC0 动画机器人)
    let gltfObj=null, mixer=null, showG=false, gLoading=false;
    function ensureGLTFLoader(cb){ if(THREE.GLTFLoader){cb();return;} var s=document.createElement('script'); s.src='/GLTFLoader.js'; s.onload=cb; s.onerror=cb; document.head.appendChild(s); }
    const gbtn=document.createElement('button'); gbtn.textContent='🤖 真实 GLTF 模型';
    gbtn.style.cssText='position:absolute;right:12px;bottom:12px;font-size:.66rem;font-weight:700;color:#fff;background:linear-gradient(120deg,#7c3aed,#2563eb);border:none;border-radius:999px;padding:6px 13px;cursor:pointer;box-shadow:0 4px 14px rgba(99,102,241,.3);';
    host.appendChild(gbtn);
    gbtn.onclick=function(){
      if(showG){ showG=false; root.visible=true; if(gltfObj)gltfObj.visible=false; gbtn.textContent='🤖 真实 GLTF 模型'; return; }
      if(gltfObj){ showG=true; root.visible=false; gltfObj.visible=true; gbtn.textContent='↩ 返回三机场景'; return; }
      if(gLoading) return; gLoading=true; gbtn.textContent='加载中…';
      ensureGLTFLoader(function(){
        if(!THREE.GLTFLoader){ gbtn.textContent='GLTF 不可用'; return; }
        new THREE.GLTFLoader().load('/RobotExpressive.glb', function(g){
          gltfObj=g.scene;
          const box=new THREE.Box3().setFromObject(gltfObj), sz=new THREE.Vector3(); box.getSize(sz);
          const s=5.2/Math.max(sz.x,sz.y,sz.z); gltfObj.scale.setScalar(s);
          const ctr=new THREE.Vector3(); box.getCenter(ctr);
          gltfObj.position.set(-ctr.x*s, -box.min.y*s-1.45, -ctr.z*s);
          gltfObj.traverse(function(o){ if(o.isMesh) o.castShadow=true; });
          if(g.animations&&g.animations.length){ mixer=new THREE.AnimationMixer(gltfObj);
            const clip=THREE.AnimationClip.findByName(g.animations,'Idle')||THREE.AnimationClip.findByName(g.animations,'Wave')||g.animations[0];
            if(clip) mixer.clipAction(clip).play(); }
          scene.add(gltfObj); showG=true; root.visible=false; gbtn.textContent='↩ 返回三机场景';
        }, undefined, function(){ gbtn.textContent='GLTF 加载失败'; gLoading=false; });
      });
    };
    let t=0;
    (function loop(){
      t+=0.016; idle+=0.016;
      wheels.forEach(w=>w.rotation.z-=0.07); sweep.rotation.y+=0.045;
      core.rotation.y+=0.012; core.rotation.x+=0.007;
      cL.intensity=0.6+Math.sin(t*2)*0.35; core.material.emissiveIntensity=1.0+Math.sin(t*2)*0.5;
      shoulder.rotation.z=Math.sin(t*0.6)*0.1; elbow.rotation.z=-0.25+Math.sin(t*0.85+0.5)*0.32;
      const gp=0.07+Math.abs(Math.sin(t*0.9))*0.09; g1.position.x=gp; g2.position.x=-gp;
      if(!drag&&idle>2.5) tRY+=0.0025;
      if(pulseG){ pulseT-=0.016;
        if(pulseT>0) pulseG.scale.setScalar(1+0.1*Math.abs(Math.sin(pulseT*8)));
        else{ pulseG.scale.setScalar(1); pulseG=null; } }
      cRY+=(tRY-cRY)*0.08; cRX+=(tRX-cRX)*0.08; root.rotation.y=cRY; root.rotation.x=cRX;
      if(mixer&&showG) mixer.update(0.016);
      if(gltfObj) gltfObj.rotation.y=cRY;
      if(cur==='home') renderer.render(scene,cam);   // 只在总览可见时渲染省 GPU
      requestAnimationFrame(loop);
    })();
  }catch(e){ /* 3D 失败 → 保留 SVG */ }
}
window.init3DRealRig=function(){
  try{
    const THREE=window.THREE; if(!THREE) return init3D();
    const host=document.querySelector('.hero-scene'), cv=document.getElementById('c3d'), fallback=document.getElementById('scene2d');
    if(!host||!cv) return init3D();
    let W=host.clientWidth||560, H=Math.round(W*0.62);
    const renderer=new THREE.WebGLRenderer({canvas:cv, antialias:true, alpha:true});
    const reducedMotion=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.65)); renderer.setSize(W,H);
    renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    if(THREE.ACESFilmicToneMapping){ renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.06; }
    if(THREE.sRGBEncoding) renderer.outputEncoding=THREE.sRGBEncoding;

    const scene=new THREE.Scene();
    scene.fog=new THREE.Fog(0xf6f9ff,15,31);
    const cam=new THREE.PerspectiveCamera(39,W/H,.1,100); cam.position.set(0,5.55,12.35); cam.lookAt(.1,.86,0);
    scene.add(new THREE.HemisphereLight(0xffffff,0xdbeafe,.72));
    const key=new THREE.DirectionalLight(0xffffff,1.15); key.position.set(5.8,10,7.5); key.castShadow=true;
    key.shadow.mapSize.set(1536,1536); const sc=key.shadow.camera; sc.left=-12; sc.right=12; sc.top=10; sc.bottom=-9; scene.add(key);
    [[0x7c3aed,-6,4,4,.36],[0x06b6d4,-1,3,5,.42],[0xf59e0b,5.5,4,3,.34]].forEach(p=>{ const l=new THREE.PointLight(p[0],p[4],22); l.position.set(p[1],p[2],p[3]); scene.add(l); });

    const root=new THREE.Group(); root.rotation.y=-.18; root.scale.setScalar(1.15); root.position.set(-.06,-.04,.22); scene.add(root);
    const M=(c,m,r,o)=>new THREE.MeshStandardMaterial({color:c,metalness:m==null?0.25:m,roughness:r==null?0.48:r,transparent:o!=null,opacity:o==null?1:o});
    const glass=new THREE.MeshPhysicalMaterial({color:0xffffff,metalness:0,roughness:.18,transmission:.35,transparent:true,opacity:.42,clearcoat:1,clearcoatRoughness:.12});
    function box(g,x,y,z,sx,sy,sz,mat){ const o=new THREE.Mesh(new THREE.BoxGeometry(sx,sy,sz),mat); o.position.set(x,y,z); o.castShadow=o.receiveShadow=true; g.add(o); return o; }
    function cyl(g,x,y,z,r1,r2,h,mat,seg){ const o=new THREE.Mesh(new THREE.CylinderGeometry(r1,r2,h,seg||24),mat); o.position.set(x,y,z); o.castShadow=o.receiveShadow=true; g.add(o); return o; }
    function rod(g,x,y,z,h,mat){ return cyl(g,x,y,z,.035,.035,h,mat,12); }
    function cable(g,pts,color){ const c=new THREE.CatmullRomCurve3(pts.map(p=>new THREE.Vector3(p[0],p[1],p[2]))); const o=new THREE.Mesh(new THREE.TubeGeometry(c,30,.018,7,false),M(color||0x111827,.05,.55)); g.add(o); return o; }
    function glowTex(hex){ const c=document.createElement('canvas'); c.width=c.height=128; const x=c.getContext('2d'), r=x.createRadialGradient(64,64,0,64,64,64); r.addColorStop(0,hex+'bb'); r.addColorStop(.46,hex+'44'); r.addColorStop(1,hex+'00'); x.fillStyle=r; x.fillRect(0,0,128,128); return new THREE.CanvasTexture(c); }
    function led(g,x,y,z,color,r){
      const mat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:color,emissiveIntensity:1.35,roughness:.18,metalness:.05});
      const o=new THREE.Mesh(new THREE.SphereGeometry(r||.045,14,10),mat); o.position.set(x,y,z); g.add(o); return o;
    }
    function floorRing(g,x,z,color,r){
      const mat=new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:.22,depthWrite:false});
      const o=new THREE.Mesh(new THREE.TorusGeometry(r||.92,.018,8,72),mat); o.rotation.x=Math.PI/2; o.position.set(x,-1.39,z); g.add(o); return o;
    }
    function glassPlate(g,x,y,z,sx,sy,sz,color,op){
      return box(g,x,y,z,sx,sy,sz,M(color||0xffffff,.02,.22,op==null ? .34 : op));
    }

    const floor=new THREE.Mesh(new THREE.PlaneGeometry(42,28),new THREE.MeshStandardMaterial({color:0xf7fbff,roughness:.86,metalness:0}));
    floor.rotation.x=-Math.PI/2; floor.position.y=-1.46; floor.receiveShadow=true; root.add(floor);
    const grid=new THREE.GridHelper(34,34,0xcbd5ff,0xe7eefc); grid.position.y=-1.445; root.add(grid);
    const backPane=new THREE.Mesh(new THREE.PlaneGeometry(16,7.8),glass);
    backPane.position.set(.2,2.45,-5.55); root.add(backPane);
    const backLineMat=new THREE.LineBasicMaterial({color:0x93c5fd,transparent:true,opacity:.28});
    for(let i=-3;i<=3;i++){
      const v=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(i*2.2,-1.1,-5.5),new THREE.Vector3(i*2.2,5.6,-5.5)]),backLineMat);
      const h=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-7.4,i*.9+1.2,-5.48),new THREE.Vector3(7.6,i*.9+1.2,-5.48)]),backLineMat);
      root.add(v); root.add(h);
    }
    [['#7c3aed',-3.75,0],['#06b6d4',-1.05,1.05],['#f59e0b',2.9,.15]].forEach(([hex,x,z])=>{
      const m=new THREE.Mesh(new THREE.PlaneGeometry(4.8,4.8),new THREE.MeshBasicMaterial({map:glowTex(hex),transparent:true,opacity:.34,depthWrite:false}));
      m.rotation.x=-Math.PI/2; m.position.set(x,-1.435,z); root.add(m);
    });

    const metal=M(0xb8c1d1,.75,.28), green=M(0x2fb881,.18,.45), yellow=M(0xfacc15,.2,.44), black=M(0x111827,.35,.48);
    const slotLights=[], stageNodes=[], routePulses=[], statusRings=[], bevTiles=[];
    const rack=new THREE.Group(); rack.position.set(-3.25,-1.35,.12); root.add(rack);
    const vehicleBrain=new THREE.Group(); rack.add(vehicleBrain);
    [[-1.22,-.74],[-1.22,.74],[1.22,-.74],[1.22,.74]].forEach(p=>rod(rack,p[0],1.75,p[1],5.35,metal));
    [-.02,1.54,3.1].forEach(y=>{
      box(rack,0,y,0,2.82,.13,1.7,green);
      box(rack,0,y+.12,0,2.48,.06,1.28,M(0x0f172a,.24,.54,.2));
    });
    [[0,.82],[0,-.82],[-1.35,0],[1.35,0]].forEach((p,i)=>box(rack,p[0],1.55,p[1],i<2?2.8:.08,.5,i<2?.08:1.62,M(0x0f172a,.22,.42,.86)));
    box(rack,0,.72,.89,2.55,.68,.08,M(0x0f172a,.22,.42)); box(rack,-.58,.76,.95,1.0,.42,.05,M(0x2563eb,.05,.35));
    box(rack,-.45,.94,.18,1.04,.1,.72,yellow); box(rack,-.45,1.16,.18,.9,.45,.08,M(0x1e293b,.28,.42));
    box(rack,.65,.98,.12,.82,.12,.62,black); box(rack,.66,1.22,.12,.66,.26,.42,M(0x111827,.22,.38));
    box(rack,-.72,2.52,.22,1.0,.09,.72,yellow); box(rack,-.72,2.75,.2,.86,.48,.08,M(0x1e293b,.3,.4));
    box(rack,.48,.9,-.62,.52,.34,.08,M(0x111827,.2,.38)); cyl(rack,-1.05,1.24,-.6,.12,.12,.2,black,16);
    const camBox=box(rack,-1.5,1.46,.24,.36,.28,.28,M(0x111827,.25,.38)); camBox.rotation.y=.15; cyl(rack,-1.5,1.18,.24,.14,.14,.3,black,16);
    cyl(vehicleBrain,.42,3.26,.06,.48,.54,.24,black); cyl(vehicleBrain,.42,3.58,.06,.42,.42,.12,M(0x0f172a,.2,.4));
    box(vehicleBrain,.9,3.45,.08,.72,.36,.56,M(0xf8fafc,.04,.25)); cyl(vehicleBrain,.12,3.72,.08,.22,.22,.18,black,18);
    const wheelGeo=new THREE.CylinderGeometry(.25,.25,.2,20); wheelGeo.rotateZ(Math.PI/2);
    const wheels=[]; [[-1.15,-.12,.78],[1.15,-.12,.78],[-1.15,-.12,-.78],[1.15,-.12,-.78]].forEach(p=>{ const w=new THREE.Mesh(wheelGeo,black); w.position.set(p[0],p[1],p[2]); w.castShadow=true; rack.add(w); wheels.push(w); });
    [[-1.15,.1,.78],[1.15,.1,.78],[-1.15,.1,-.78],[1.15,.1,-.78]].forEach(p=>box(rack,p[0],p[1],p[2],.38,.14,.28,M(0xd1d5db,.5,.32)));
    cable(rack,[[-1.06,3.0,.62],[-.25,2.2,.9],[.38,1.32,.76],[.85,.18,.5]],0xffffff);
    cable(rack,[[-.85,1.02,.62],[-.1,.38,.92],[.65,-.08,.44]],0x111827);
    cable(rack,[[-1.42,1.4,.24],[-.8,1.25,.74],[-.34,.92,.62]],0xffffff);
    const rackCore=new THREE.Mesh(new THREE.IcosahedronGeometry(.25,0),new THREE.MeshStandardMaterial({color:0xffffff,emissive:0x8b5cf6,emissiveIntensity:1.05,roughness:.28,metalness:.1})); rackCore.position.set(.92,2.35,.58); rack.add(rackCore);
    for(let i=0;i<5;i++){
      const y=.42+i*.33;
      box(rack,-.96,y,.96,.38,.06,.04,M(0x2563eb,.08,.32));
      box(rack,-.52,y,.96,.38,.06,.04,M(0x7c3aed,.08,.32));
      slotLights.push(led(rack,-1.18,y,.99,0x22d3ee,.032));
      slotLights.push(led(rack,-.31,y,.99,0xa78bfa,.032));
    }
    for(let i=0;i<7;i++) box(rack,.42+.065*i,2.78,.68,.035,.34,.08,M(0xdbeafe,.55,.25));
    for(let i=0;i<4;i++) cable(rack,[[-.88,2.45,.84],[-.35,2.2,.98],[.12,1.62,.92],[.7,1.12,.7-.12*i]],i%2?0x7c3aed:0x06b6d4);
    glassPlate(rack,0,3.23,.02,2.74,.055,1.54,0xffffff,.24);
    statusRings.push(floorRing(root,-3.25,.12,0x7c3aed,1.28));
    const sweepGeo=new THREE.CircleGeometry(1.15,36,0,Math.PI/3); sweepGeo.rotateX(-Math.PI/2);
    const sweep=new THREE.Mesh(sweepGeo,new THREE.MeshBasicMaterial({color:0x06b6d4,transparent:true,opacity:.2,side:THREE.DoubleSide})); sweep.position.set(.42,3.56,.06); vehicleBrain.add(sweep);
    const bevDeck=new THREE.Group(); bevDeck.position.set(.72,3.34,.72); bevDeck.rotation.x=-.84; bevDeck.rotation.z=-.24; vehicleBrain.add(bevDeck);
    for(let r=0;r<4;r++) for(let c=0;c<5;c++){
      const occ=(r+c)%4===0, warn=(r===0&&c===4)||(r===3&&c===1);
      const cell=box(bevDeck,(c-2)*.2,0,(r-1.5)*.18,.15,.018,.12,M(warn?0xf59e0b:occ?0x34d399:0xdbeafe,.03,.24,.64));
      cell.receiveShadow=false; bevTiles.push(cell);
    }

    const table=new THREE.Group(); table.position.set(2.15,-1.44,.05); root.add(table);
    box(table,.55,.06,0,5.1,.14,2.25,M(0xb45f35,.15,.44)); box(table,.55,.15,0,5.15,.08,2.28,M(0xc06b3c,.08,.5));
    [[-1.85,-.92],[-1.85,.92],[2.95,-.92],[2.95,.92]].forEach(p=>rod(table,p[0],-.63,p[1],1.65,metal));
    box(table,-.25,-.95,1.0,.95,.44,.05,M(0x0f172a,.2,.5)); box(table,1.92,-.95,1.0,.82,.34,.05,M(0x0f172a,.2,.5));
    cable(table,[[-1.2,.28,.7],[-.4,.12,1.2],[.3,-.45,1.0],[.9,-.92,.96]],0x111827);
    box(table,.6,.35,-.3,.8,.28,.52,M(0xd1d5db,.05,.58));
    const tray=new THREE.Group(); tray.position.set(.62,.55,.55); table.add(tray);
    box(tray,0,0,0,1.42,.08,.66,M(0xf8fafc,.04,.22));
    for(let r=0;r<2;r++) for(let c=0;c<5;c++){
      cyl(tray,(c-2)*.25,.09,(r-.5)*.26,.055,.064,.055,M((r+c)%3===0?0xa78bfa:(r+c)%3===1?0x22d3ee:0xfbbf24,.02,.28,.72),18);
    }
    const bowl=cyl(table,-.45,.58,-.56,.36,.46,.22,M(0xf1f5f9,.03,.26),32);
    cyl(table,-.45,.72,-.56,.24,.30,.06,M(0xdbeafe,.02,.22,.62),32);
    [[1.54,.62,.62,0x22d3ee],[1.82,.58,.62,0xa78bfa],[2.08,.54,.62,0xf59e0b]].forEach(p=>{
      cyl(table,p[0],p[1],p[2],.08,.10,.38,M(0xf8fafc,.02,.18,.56),18);
      led(table,p[0],p[1]+.24,p[2],p[3],.035);
    });
    box(table,-1.55,.55,.55,.56,.08,.46,M(0xe2e8f0,.04,.35)); box(table,-1.55,.69,.55,.42,.20,.26,M(0xffffff,.02,.22,.42));
    statusRings.push(floorRing(root,2.7,.05,0xf59e0b,1.38));
    const charStation=new THREE.Group(); charStation.position.set(5.28,-1.40,1.18); root.add(charStation);
    box(charStation,0,.1,0,1.72,.18,1.18,M(0xe2e8f0,.12,.34));
    box(charStation,-.36,.58,-.05,.62,.78,.66,M(0xf8fafc,.05,.26));
    box(charStation,-.36,.82,.31,.52,.12,.08,M(0x0f172a,.18,.38));
    cyl(charStation,-.36,.42,.31,.16,.16,.16,M(0x111827,.25,.42),20);
    const plPanel=glassPlate(charStation,.48,.72,-.08,.62,.46,.04,0x93c5fd,.48);
    plPanel.rotation.y=-.18;
    for(let i=0;i<5;i++){
      const h=.12+.05*((i%3)+1);
      box(charStation,.24+i*.12,.56+h*.5,.18,.045,h,.035,M(i%2?0x7c3aed:0x06b6d4,.04,.24,.75));
    }
    led(charStation,-.62,1.05,-.36,0x10b981,.04); led(charStation,.72,.98,-.32,0x22d3ee,.035);
    cable(root,[[4.3,.65,1.12],[4.75,.56,1.45],[5.28,.42,1.18]],0x10b981);
    statusRings.push(floorRing(root,5.28,1.18,0x10b981,1.02));

    function makeArm(x,z,accent,redTool){
      const g=new THREE.Group(); g.position.set(x,.16,z); table.add(g);
      const white=M(0xf8fafc,.05,.28), joint=M(0xe2e8f0,.08,.24), blue=M(accent,.12,.38), red=M(0xef4444,.08,.34);
      box(g,0,.05,0,.78,.16,.52,blue); cyl(g,0,.25,0,.26,.32,.28,white);
      const s=new THREE.Group(); s.position.set(0,.37,0); s.rotation.z=.62; g.add(s);
      cyl(s,0,0,0,.16,.16,.24,joint); const a1=box(s,.42,.44,0,.34,1.0,.34,white); a1.rotation.z=-.16;
      const e=new THREE.Group(); e.position.set(.62,.9,0); e.rotation.z=-1.08; s.add(e);
      cyl(e,0,0,0,.17,.17,.28,joint); const a2=box(e,.42,.34,0,.3,.82,.3,white); a2.rotation.z=.05;
      const w=new THREE.Group(); w.position.set(.82,.72,0); w.rotation.z=.35; e.add(w);
      cyl(w,0,0,0,.16,.16,.24,joint); box(w,.28,.08,0,.42,.22,.24,white);
      if(redTool){ box(w,.72,.02,0,1.05,.2,.18,red); box(w,1.25,.02,.16,.54,.12,.16,red); box(w,1.25,.02,-.16,.54,.12,.16,red); }
      else{ box(w,.58,.02,0,.54,.18,.2,M(0x0f172a,.25,.42)); box(w,.88,.08,.13,.22,.08,.08,blue); box(w,.88,.08,-.13,.22,.08,.08,blue); }
      return g;
    }
    const arm01=makeArm(-.95,-.1,0x0ea5e9,false), arm02=makeArm(1.88,-.28,0x3b82f6,true);
    const sampleCurve=new THREE.CatmullRomCurve3([
      new THREE.Vector3(-3.9,1.05,1.18),
      new THREE.Vector3(-1.85,.92,1.52),
      new THREE.Vector3(.65,.78,1.34),
      new THREE.Vector3(2.95,.72,.92),
      new THREE.Vector3(4.1,.56,.62)
    ]);
    const traceMat=new THREE.LineBasicMaterial({color:0x7c3aed,transparent:true,opacity:.5});
    const trace=new THREE.Line(new THREE.BufferGeometry().setFromPoints(sampleCurve.getPoints(56)),traceMat); root.add(trace);
    const traceTube=new THREE.Mesh(new THREE.TubeGeometry(sampleCurve,64,.028,8,false),new THREE.MeshBasicMaterial({color:0x7c3aed,transparent:true,opacity:.18,depthWrite:false}));
    root.add(traceTube);
    [0,.25,.5,.75,1].forEach((u,i)=>{
      const p=sampleCurve.getPoint(u);
      const n=led(root,p.x,p.y,p.z,i===0?0x7c3aed:i===4?0x10b981:0xf59e0b,.055);
      stageNodes.push(n);
    });
    for(let i=0;i<3;i++){
      const p=sampleCurve.getPoint(i/3);
      const rp=led(root,p.x,p.y,p.z,0xfbbf24,.07); routePulses.push({mesh:rp,phase:i/3});
    }
    const sampleDot=new THREE.Mesh(new THREE.SphereGeometry(.16,18,12),new THREE.MeshStandardMaterial({color:0xf59e0b,emissive:0xfbbf24,emissiveIntensity:.95,roughness:.32}));
    sampleDot.position.copy(sampleCurve.getPoint(.52)); root.add(sampleDot);
    const cloud=new THREE.Group(); cloud.position.set(4.55,3.45,-2.7); root.add(cloud);
    const cloudMat=M(0xe0f2fe,.05,.22,.68);
    [[0,0,0,.62,.34,.28],[-.42,-.04,0,.46,.26,.24],[.42,-.02,0,.5,.3,.25],[0,.22,0,.72,.26,.22]].forEach(p=>box(cloud,p[0],p[1],p[2],p[3],p[4],p[5],cloudMat));
    const edge=new THREE.Group(); edge.position.set(.02,-.18,.05); cloud.add(edge);
    for(let i=0;i<3;i++){
      glassPlate(edge,0,.12+i*.22,0,.98,.10,.58,0xe0f2fe,.42);
      led(edge,-.38,.18+i*.22,.31,i===0?0x10b981:i===1?0x22d3ee:0xa78bfa,.028);
      box(edge,.12,.18+i*.22,.31,.34,.025,.035,M(0x2563eb,.04,.25,.58));
    }
    const shield=new THREE.Mesh(new THREE.TorusGeometry(.56,.025,8,72),new THREE.MeshBasicMaterial({color:0x22d3ee,transparent:true,opacity:.42,depthWrite:false}));
    shield.position.set(0,.48,.02); edge.add(shield);
    statusRings.push(floorRing(root,4.55,-2.7,0x22d3ee,1.05));
    const cloudLink=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(3.9,3.1,-2.35),new THREE.Vector3(2.2,2.2,-1.8),new THREE.Vector3(.8,1.4,.7)]),new THREE.LineBasicMaterial({color:0x06b6d4,transparent:true,opacity:.48}));
    root.add(cloudLink);

    const FOCUS={lab:{g:rack,ry:.5},car:{g:vehicleBrain,ry:.32},arm:{g:table,ry:-.42},xrd:{g:charStation,ry:-.55},vps:{g:cloud,ry:.18},sample:{g:sampleDot,ry:-.2}};
    let tRY=-.14,cRY=-.14,tRX=0,cRX=0,drag=false,lx=0,ly=0,idle=0,pulseG=null,pulseT=0;
    window.focus3D=function(k){ const f=FOCUS[k]; if(!f) return; tRY=f.ry; tRX=.06; idle=-8; pulseG=f.g; pulseT=1.5; toast('⬢ 3D 聚焦: '+({lab:'AI 脑机架',car:'车载脑',arm:'主机械臂工位',xrd:'XRD/PL 表征站',vps:'VPS/公网层',sample:'样品路径'})[k]); };
    cv.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;idle=0;cv.style.cursor='grabbing';});
    window.addEventListener('pointerup',()=>{drag=false;cv.style.cursor='grab';});
    window.addEventListener('pointermove',e=>{ if(!drag)return; tRY+=(e.clientX-lx)*.01; tRX=Math.max(-.45,Math.min(.5,tRX+(e.clientY-ly)*.005)); lx=e.clientX; ly=e.clientY; idle=0; });
    function resize(){ const w=host.clientWidth||W,h=Math.round(w*.62); renderer.setSize(w,h); cam.aspect=w/h; cam.updateProjectionMatrix(); }
    window.addEventListener('resize',resize);
    if(fallback) fallback.style.display='none'; cv.style.display='block';
    const hint=document.createElement('div'); hint.className='scene-hint'; hint.textContent=uiText('拖动旋转 · 真实感实验室数字孪生','Drag to rotate · realistic lab digital twin'); host.appendChild(hint);
    const focusBar=document.createElement('div'); focusBar.className='scene-focus';
    focusBar.innerHTML='<button data-i18n="home.focusLab" onclick="focus3D(&quot;lab&quot;)">AI 脑机架</button><button data-i18n="home.focusCar" onclick="focus3D(&quot;car&quot;)">车载脑</button><button data-i18n="home.focusArm" onclick="focus3D(&quot;arm&quot;)">机械臂工位</button><button onclick="focus3D(&quot;xrd&quot;)">XRD / PL</button><button data-i18n="home.focusVps" onclick="focus3D(&quot;vps&quot;)">公网 VPS 层</button><button data-i18n="home.focusSample" onclick="focus3D(&quot;sample&quot;)">样品路径</button>';
    host.appendChild(focusBar);
    try{ applyI18n(uiLang()); }catch(e){}

    let t=0;
    (function loop(){
      t+=.016; idle+=.016;
      if(!reducedMotion){
        wheels.forEach(w=>w.rotation.x-=.09); sweep.rotation.y+=.045; rackCore.rotation.x+=.01; rackCore.rotation.y+=.013;
        arm01.rotation.y=Math.sin(t*.7)*.04; arm02.rotation.y=-.06+Math.sin(t*.55+.7)*.035;
        slotLights.forEach((l,i)=>{ l.material.emissiveIntensity=1.0+.55*Math.sin(t*2.1+i*.62)*Math.sin(t*2.1+i*.62); });
        bevTiles.forEach((b,i)=>{ b.position.y=.006*Math.sin(t*1.7+i*.55); });
        traceMat.opacity=.25+.2*Math.sin(t*1.6)*Math.sin(t*1.6); traceTube.material.opacity=.13+.07*Math.sin(t*1.2)*Math.sin(t*1.2);
        routePulses.forEach((p,i)=>{ const q=sampleCurve.getPoint((p.phase+t*.085)%1); p.mesh.position.copy(q); p.mesh.scale.setScalar(.78+.32*Math.sin(t*2+i)); });
        const sp=sampleCurve.getPoint((.48+.035*Math.sin(t*.72))%1); sampleDot.position.copy(sp); sampleDot.position.y+=Math.sin(t*2.4)*.04;
        stageNodes.forEach((n,i)=>{ n.material.emissiveIntensity=.8+.55*Math.sin(t*1.45+i*.8)*Math.sin(t*1.45+i*.8); });
        statusRings.forEach((r,i)=>{ r.material.opacity=.15+.12*Math.sin(t*1.1+i*.9)*Math.sin(t*1.1+i*.9); r.rotation.z+=.002+i*.0005; });
        cloudLink.material.opacity=.32+.18*Math.sin(t*1.2)*Math.sin(t*1.2);
      }
      if(!drag&&!reducedMotion&&idle>3) tRY+=.0018;
      if(pulseG){ pulseT-=.016; if(pulseT>0) pulseG.scale.setScalar(1+.075*Math.abs(Math.sin(pulseT*9))); else{ pulseG.scale.setScalar(1); pulseG=null; } }
      cRY+=(tRY-cRY)*.08; cRX+=(tRX-cRX)*.08; root.rotation.y=cRY; root.rotation.x=cRX;
      if(cur==='home'&&!document.hidden) renderer.render(scene,cam);
      requestAnimationFrame(loop);
    })();
  }catch(e){ init3D(); }
};
var _threeSceneStarted=false;
function loadThreeSceneOnce(){
  if(_threeSceneStarted) return;
  _threeSceneStarted=true;
  var s=document.createElement('script');
  s.src='/three.min.js';
  s.async=true;
  s.onload=function(){ (window.init3DRealRig||init3D)(); };
  document.head.appendChild(s);
}

/* ---- P8: 命令面板 (Ctrl+K) ---- */
let _palDyn=[], _palSel=0, _palShown=[];
// 搜索 2.0: 模糊子序列打分 — 子串优先 (前缀最高), 否则按字符顺序子序列匹配, gap 越小分越高。CJK 逐字可用。
function _palScore(hay, term){
  if(!term) return 1;
  const i=hay.indexOf(term);
  if(i>=0) return 1000 - i*2 + (i===0?300:0) + term.length*4;   // 子串: 越靠前/越长越高
  let hi=0, sc=0, gap=0;                                          // 子序列模糊
  for(const c of term){ const f=hay.indexOf(c,hi); if(f<0) return 0; if(f===hi) sc+=6; gap+=f-hi; hi=f+1; }
  return Math.max(1, 120 - gap*2) + sc;
}
function _palMatch(it, terms){                                    // 多词 AND: 每词都须命中, 取分和
  const hay=(it.t+' '+it.k).toLowerCase(); let total=0;
  for(const t of terms){ const s=_palScore(hay, t); if(!s) return 0; total+=s; }
  return total;
}
function _palHi(title, terms){                                    // 高亮命中子串 (整词子串, 安全转义)
  let h=esc(title);
  terms.forEach(t=>{ if(!t) return; const re=new RegExp('('+t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','ig');
    h=h.replace(re,'<mark>$1</mark>'); });
  return h;
}
function _palRecentGet(){ try{ return JSON.parse(localStorage.getItem('xrd_pal_recent')||'[]'); }catch(e){ return []; } }
function _palRecentPush(t){ try{ let r=_palRecentGet().filter(x=>x!==t); r.unshift(t); r=r.slice(0,6);
  localStorage.setItem('xrd_pal_recent', JSON.stringify(r)); }catch(e){} }
function _palExec(it){ _palRecentPush(it.t); palClose(); it.f(); }

/* ---- H12 扫码跳转: BarcodeDetector 原生解码二维码 → 解析直达 (无第三方库) ---- */
let _scanStream=null, _scanRAF=0, _scanDet=null;
async function scanOpen(){
  palClose();
  const ov=document.getElementById('scanOv'); if(!ov) return;
  ov.classList.add('show');
  const msg=document.getElementById('scanMsg'); const vid=document.getElementById('scanVid');
  const hasDet = ('BarcodeDetector' in window);
  if(!hasDet){ msg.innerHTML='此浏览器不支持原生二维码识别 — 可手动输入码值或链接:'; document.getElementById('scanManual').style.display='flex'; }
  else { document.getElementById('scanManual').style.display='none'; }
  try{
    if(!hasDet) throw new Error('no detector');
    _scanDet=new BarcodeDetector({formats:['qr_code']});
    _scanStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
    vid.srcObject=_scanStream; await vid.play();
    msg.textContent='把二维码对准取景框…';
    const tick=async()=>{
      if(!document.getElementById('scanOv').classList.contains('show')) return;
      try{ const codes=await _scanDet.detect(vid); if(codes&&codes.length){ scanResolve(codes[0].rawValue); return; } }catch(e){}
      _scanRAF=requestAnimationFrame(tick);
    };
    _scanRAF=requestAnimationFrame(tick);
  }catch(e){
    msg.innerHTML='无法打开摄像头'+(hasDet?' ('+esc(e.message||'')+')':'')+' — 可手动输入码值或链接:';
    document.getElementById('scanManual').style.display='flex';
  }
}
function scanClose(){
  const ov=document.getElementById('scanOv'); if(ov) ov.classList.remove('show');
  if(_scanRAF){ cancelAnimationFrame(_scanRAF); _scanRAF=0; }
  if(_scanStream){ _scanStream.getTracks().forEach(t=>t.stop()); _scanStream=null; }
  const v=document.getElementById('scanVid'); if(v) v.srcObject=null;
}
function scanManualGo(){ const v=(document.getElementById('scanInput')||{}).value||''; if(v.trim()) scanResolve(v.trim()); }
function scanResolve(text){
  // 1) 本站 URL → 取 ?go= / ?wo= / 路径 hash; 2) WO 码 → 找工单; 3) 裸 token → 匹配面板项
  let nav=null, label=text;
  try{
    if(/^https?:\/\//i.test(text)){
      const u=new URL(text);
      const g=u.searchParams.get('go'), wo=u.searchParams.get('wo'), as=u.searchParams.get('asset');
      if(g && PAGES_ALL.includes(g)) nav=()=>go(g);
      else if(wo) nav=()=>_scanWo(wo);
      else if(as) nav=()=>go('assets');
      else if(u.hash){ const h=u.hash.replace(/^#/,''); if(PAGES_ALL.includes(h)) nav=()=>go(h); }
    }
  }catch(e){}
  if(!nav){
    const m=text.match(/\b(WO[-_ ]?\d{2,})\b/i);
    if(m) nav=()=>_scanWo(m[1].replace(/[-_ ]/g,'').toUpperCase());
  }
  if(!nav){
    const t=text.trim().toLowerCase();
    if(PAGES_ALL.includes(t)) nav=()=>go(t);
    else{ const it=_palBase().find(x=>(x.t+' '+x.k).toLowerCase().includes(t)); if(it) nav=it.f; }
  }
  scanClose();
  if(nav){ toast('扫码跳转: '+label,'ok'); nav(); }
  else toast('未识别的码: '+label,'warn');
}
function _scanWo(code){
  const it=_palDyn.find(x=>(x.k||'').toLowerCase().includes(code.toLowerCase()));
  if(it){ it.f(); } else { go('mq'); toast('未找到工单 '+code+', 已开批次队列','info'); }
}
function _iaGo(k, msg, opts){
  if(msg) toast(msg, 'info');
  go(k, opts||{});
}
function _palShellItems(){
  return [
    {ic:'IA', t:'总览控制台', k:'overview home public console shell 总览 控制台', f:()=>go('home')},
    {ic:'GB', t:'全球对标: 顶级站横向对比与证据评分卡', k:'benchmark global top site apple alphafold materials project emerald cloud lab waymo observability score gates 全球 对标 证据', f:()=>go('benchmark')},
    {ic:'RS', t:'科研 / 预测: AI 脑、队列、材料与配方构建', k:'research prediction material formula trace_id ai batch 科研 预测 材料 配方', f:()=>go('atlas')},
    {ic:'RB', t:'机器人 / 执行: 车载脑与双臂工位', k:'robot execution vehicle car arm workstation task 机器人 执行 车载脑 机械臂', f:()=>go('twin')},
    {ic:'ST', t:'运维 / 状态: 公共状态、可观测性、日志', k:'ops status live mirror replay offline observability logs traces 运维 状态 日志', f:()=>go('status')},
    {ic:'SC', t:'安全 / 设置: RBAC、发布与数据治理', k:'security settings rbac release audit data governance 安全 设置 发布 数据治理', f:()=>go('sec')},
    {ic:'Y', t:'材料搜索示例: YAG:Cr3+', k:'formula material yag cr3 y3al5o12 trace prediction 材料 搜索 示例', f:()=>_palMaterial('YAG Cr3')},
    {ic:'G', t:'材料搜索示例: GGG:Ni2+', k:'formula material ggg ni2 gd3ga5o12 prediction 材料 搜索 示例', f:()=>_palMaterial('GGG Ni2')},
    {ic:'S', t:'材料搜索示例: SYGO:Cr/Ni', k:'formula material sygo cr ni prediction 材料 搜索 示例', f:()=>_palMaterial('SYGO Cr Ni')},
  ];
}
function _palMaterial(q){
  _atF.q=q;
  if(typeof MX!=='undefined') MX.q=q;
  go('atlas',{after:()=>{
    _atF.q=q;
    if(typeof MX!=='undefined'){
      MX.q=q;
      const inp=document.getElementById('mxQ'); if(inp) inp.value=q;
      if(typeof mxFetch==='function') mxFetch();
    }
    atlasRender(); toast('Material filter: '+q,'info');
  }});
}
function _palSmart(q){
  q=(q||'').trim();
  if(!q) return [];
  const low=q.toLowerCase(), out=[];
  const isTrace=/\btrace[_-]?[a-z0-9_-]{3,}\b/i.test(q)||low.includes('trace_id');
  const isWo=/\bwo[-_ ]?\d{2,}\b/i.test(q)||low.includes('work order')||low.includes('工单');
  if(isTrace){
    out.push({ic:'TR',t:'打开链路追踪搜索: '+q,k:'trace trace_id request_id prediction reproducibility 链路 追踪 '+low,f:()=>go('traces',{after:()=>traceFill(q)})});
  }
  if(isWo){
    out.push({ic:'WO',t:'打开任务驾驶舱: '+q,k:'workorder work order batch queue tasks 工单 任务 '+low,f:()=>go('tasks')});
  }
  if(!isTrace && !isWo && (/[A-Z][a-z]?\d|:|cr3|ni2|yag|ggg|sygo/i.test(q))){
    out.push({ic:'MX',t:'搜索材料 / 配方: '+q,k:'material formula host dopant prediction 材料 配方 '+low,f:()=>_palMaterial(q)});
  }
  if(low.includes('asset')||low.includes('资产')||low.includes('x5')||low.includes('arm01')||low.includes('arm02')){
    out.push({ic:'AS',t:'搜索资产注册表: '+q,k:'asset registry equipment 资产 注册表 '+low,f:()=>_iaGo('assets','资产搜索: '+q)});
  }
  return out;
}
function _palBase(){
  const b=[
    {ic:'⊞',t:'总览',k:'home overview 0',f:()=>go('home')},
    {ic:'✨',t:'技术亮点',k:'highlight 4',f:()=>go('highlight')},
    {ic:'●',t:'公共状态 (真机 / 镜像 / 离线)',k:'status public health mirror offline uptime 状态 公网',f:()=>go('status')},
    {ic:'🔬',t:'材料图鉴 (生成式候选 Atlas)',k:'atlas 图鉴 candidates 候选 材料',f:()=>go('atlas')},
    {ic:'🧩',t:'模型注册表 (全模型清单 + 在线态)',k:'models 模型 注册表 registry llm bpu',f:()=>go('models')},
    {ic:'🧪',t:'配方构建器 (点周期表 → 真预测)',k:'build 构建器 周期表 配方 predict periodic',f:()=>go('build')},
    {ic:'🔬',t:'实测复现台 (锚点喂回引擎 · 误差/CI)',k:'repro 复现 reproduce 实测 误差 ci 锚点 bench',f:()=>go('repro')},
    {ic:'📥',t:'实测导入向导 (CSV 拖拽预览)',k:'import 导入 csv 拖拽 dropzone 回填',f:()=>go('importw')},
    {ic:'📓',t:'电子实验笔记 (ELN · 真持久化)',k:'eln 笔记 notebook 实验 记录',f:()=>go('eln')},
    {ic:'📐',t:'标准合规墙 (ISA-18.2/88 · Part 11 · SLO)',k:'standards 标准 合规 isa part11 compliance',f:()=>go('standards')},
    {ic:'💰',t:'成本能效 (BOM + 功耗)',k:'cost 成本 能效 bom 功耗 efficiency',f:()=>go('cost')},
    {ic:'🧠',t:'AI 脑 · 智能计算平台',k:'lab ai 1',f:()=>go('lab')},
    {ic:'🚗',t:'车载脑 · NavCockpit',k:'car nav 2',f:()=>go('car')},
    {ic:'🦾',t:'机械臂 · WorkCockpit',k:'arm work 3',f:()=>go('arm')},
    {ic:'🛰',t:'机群驾驶舱 (真机 / 镜像 / 离线)',k:'fleet devices robot serving mirror offline 舰队 机群',f:()=>go('fleet')},
    {ic:'🧾',t:'任务驾驶舱 (阻塞项 / 下一步)',k:'tasks workorder stage blocker next action 工单 任务',f:()=>go('tasks')},
    {ic:'📋',t:'批次工单',k:'mq workorder wo 7',f:()=>go('mq')},
    {ic:'🖥',t:'统一运营总览 NOC Wall (一屏掌控)',k:'noc 运营 总览 wall 运维墙 grafana 大屏 一屏',f:()=>go('noc')},
    {ic:'🏭',t:'OEE 生产线看板 + 安灯 (Andon)',k:'oee 生产线 andon 安灯 可用率 性能 质量 制造 manufacturing',f:()=>go('oee')},
    {ic:'🔔',t:'告警中心 (规则引擎 · 通知 · 值班)',k:'alert 告警 规则 rule 通知 notify 值班 oncall 静默 silence pagerduty',f:()=>go('alert')},
    {ic:'🧪',t:'质量中心 QMS (NCR · CAPA · COA · 族谱)',k:'qms 质量 ncr capa coa 合格证 族谱 genealogy quality part11',f:()=>go('qms')},
    {ic:'🔧',t:'维护管理 CMMS (PM · MTBF/MTTR · 备件)',k:'cmms 维护 pm 预防 mtbf mttr 备件 spares maintenance',f:()=>go('cmms')},
    {ic:'🩺',t:'平台自监控 (RED/USE · Prometheus)',k:'self 自监控 prometheus red use metrics 进程 内存 看门狗',f:()=>go('self')},
    {ic:'📑',t:'日志检索 (结构化 · 请求追踪)',k:'logs 日志 log trace 追踪 req_id elk 检索',f:()=>go('logs')},
    {ic:'🧬',t:'链路追踪 (预测 / 请求瀑布)',k:'traces trace_id req_id waterfall 追踪 瀑布 链路',f:()=>go('traces')},
    {ic:'🛡',t:'安全态势 (响应头 · RBAC · 评分)',k:'sec 安全 security posture csp hsts rbac 评分 62443 owasp',f:()=>go('sec')},
    {ic:'🗄',t:'数据治理 (留存 · 降采样 · 备份)',k:'data 数据治理 governance 留存 retention rollup 备份 backup wal',f:()=>go('data')},
    {ic:'🚀',t:'发布与配置 (变更日志 · 配置中心)',k:'release 发布 config 配置 变更 changelog rollback 回滚',f:()=>go('release')},
    {ic:'🩺',t:'运维总览 · 告警 · SLO · 报表',k:'ops alarm slo report 5',f:()=>go('ops')},
    {ic:'📈',t:'可观测性中心 (时序趋势图 · 延迟/KPI)',k:'obs 观测 observability 时序 趋势 图表 metrics grafana 延迟',f:()=>go('obs')},
    {ic:'🕸',t:'服务依赖拓扑 (链路活体地图)',k:'topo 拓扑 topology 依赖 service map 链路 datadog',f:()=>go('topo')},
    {ic:'🎯',t:'SLO 错误预算 (燃尽图 · 可用性目标)',k:'slo budget 错误预算 可用性 燃尽 sre burndown',f:()=>go('budget')},
    {ic:'🚨',t:'事故复盘 (时间线 · 根因 · 改进项)',k:'incident 事故 复盘 postmortem 时间线 pagerduty 根因',f:()=>go('inc')},
    {ic:'⏳',t:'全局时间机器 (回看历史时刻平台态)',k:'tm 时间机器 time travel 回放 replay 历史 foundry',f:()=>go('tm')},
    {ic:'🔄',t:'镜像保鲜台 (数据新鲜度 · 真机可达 · 同步)',k:'sync 镜像 保鲜 mirror freshness 同步 兜底',f:()=>go('sync')},
    {ic:'✅',t:'演示就绪预检 (GO/NO-GO · 在险册)',k:'preflight 预检 就绪 readiness go nogo',f:()=>go('preflight')},
    {ic:'🎨',t:'设计系统 (token 即文档)',k:'design 设计 token',f:()=>window.open('/design','_blank')},
    {ic:'🗂',t:'资产数字孪生',k:'assets 8',f:()=>go('assets')},
    {ic:'🌐',t:'实验室全景孪生 (3D 真运动学)',k:'twin 3d digital 9',f:()=>go('twin')},
    {ic:'🧊',t:'AR 晶体查看器 (示意晶胞 · 手机可 AR)',k:'ar 晶体 crystal 增强现实 model-viewer 3d 晶胞',f:()=>go('ar')},
    {ic:'📖',t:'项目故事',k:'story 6',f:()=>go('story')},
    {ic:'📚',t:'工程档案馆 (ADR · 踩坑 · 历程)',k:'archive 档案 adr 踩坑 历程 决策',f:()=>go('archive')},
    {ic:'📖',t:'术语百科 (跨学科白话解释)',k:'glossary 术语 百科 词典 概念',f:()=>go('glossary')},
    {ic:'🆕',t:'更新日志 (What\'s New · 演进时间线)',k:'changelog 更新 日志 whatsnew 时间线',f:()=>go('changelog')},
    {ic:'📷',t:'扫码跳转 (二维码 → 直达批次 / 页面)',k:'scan 扫码 saoma qr 二维码 扫描 camera',f:scanOpen},
    {ic:'▶',t:'一条龙演示 (剧场模式)',k:'tour theater demo',f:tourStart},
    {ic:'🖥',t:'大屏轮播模式 (展台无人值守)',k:'kiosk 大屏 轮播 daping',f:kioskStart},
    {ic:'🔔',t:(NOTIFY.on?'关闭':'开启')+'桌面告警通知',k:'notify notification 通知 告警',f:notifyToggle},
    {ic:'⬇',t:'导出工单 CSV',k:'csv export workorder',f:()=>location.href='/api/export/workorders.csv'},
    {ic:'⬇',t:'导出事件 CSV (7 天)',k:'csv export events',f:()=>location.href='/api/export/events.csv?hours=168'},
    {ic:'📜',t:'API 文档',k:'api docs openapi',f:()=>window.open('/api/docs','_blank')},
    {ic:'🛂',t:'科研护照 Research Passport',k:'research passport evidence bundle citation global 科研 护照 证据 引用',f:()=>window.open('/api/research_passport','_blank')},
    {ic:'📦',t:'下载公开证据包 Evidence Bundle',k:'evidence bundle download json txt judging offline 证据包 离线 答辩',f:()=>window.open('/api/evidence_bundle.json','_blank')},
    {ic:'⎋',t:'登出',k:'logout exit',f:()=>location.href='/logout'},
  ];
  if(window._role==='admin'){
    b.push({ic:'🛡',t:'平台管理 (admin)',k:'admin manage',f:openAdmin});
    b.push({ic:'📊',t:'立即生成今日日报 (admin)',k:'report daily',f:repGenNow});
  }
  return b.concat(_palDyn);
}
async function _palLoadDyn(){
  if(_palDyn.length) return;
  try{
    const [a,w,s]=await Promise.all([
      fetch('/api/assets',{cache:'no-store'}).then(r=>r.json()).catch(()=>({})),
      fetch('/api/workorders?limit=20',{cache:'no-store'}).then(r=>r.json()).catch(()=>({})),
      fetch('/api/search/index',{cache:'no-store'}).then(r=>r.json()).catch(()=>({}))]);
    (a.groups||[]).forEach(g=>(g.children||[]).forEach(c=>{
      _palDyn.push({ic:'🔩',t:c.name+' · 资产',k:(c.id+' '+c.kind),f:()=>go('assets')});}));
    (w.workorders||[]).forEach(x=>{
      _palDyn.push({ic:'📄',t:x.code+' '+x.formula+' · 批次档案',k:'wo '+x.code,
        f:()=>go('mq',{after:()=>woDetail(x.id)})});});
    // 跨系统: AI 脑预测 (点击进 AI 脑预测历史页)
    (s.predictions||[]).forEach(p=>{
      const dop=p.dopant?(' '+p.dopant):''; const vd=p.verdict&&p.verdict!=='—'?(' · '+p.verdict):'';
      _palDyn.push({ic:'🔬',t:p.formula+dop+vd+' · 预测',k:('pred prediction '+p.formula+' '+p.dopant+' '+p.verdict).toLowerCase(),
        f:()=>go('lab',{path:'/predictions'})});});
    // 跨系统: 车载脑语义地标 (点击进车载脑)
    (s.landmarks||[]).forEach(lm=>{
      _palDyn.push({ic:'📍',t:lm.name+' · 语义地标',k:('landmark 地标 '+lm.name).toLowerCase(),
        f:()=>go('car')});});
  }catch(e){}
}
function palOpen(){
  document.getElementById('pal').classList.add('show');
  const inp=document.getElementById('palIn'); inp.value=''; _palSel=0;
  palRender(''); inp.focus(); _palLoadDyn().then(()=>palRender(inp.value));
}
function palClose(){ document.getElementById('pal').classList.remove('show'); }
function palGroup(it){
  const k=((it&&it.k)||'').toLowerCase(), t=((it&&it.t)||'').toLowerCase();
  if(/security|安全|rbac|release|发布|data governance|备份/.test(k+t)) return '安全 / 发布';
  if(/ops|status|observability|logs|trace|slo|incident|noc|self|oee|cmms|qms|运维|状态|日志|追踪|告警/.test(k+t)) return '运维 / 状态';
  if(/robot|car|arm|fsd|replay|twin|fleet|tasks|机器人|车载脑|机械臂|孪生/.test(k+t)) return '具身 / 执行';
  if(/atlas|material|formula|prediction|lab|brain|studio|research|材料|配方|预测|科研/.test(k+t)) return '科研 / 数据';
  if(/benchmark|defense|evidence|archive|story|preflight|global|对标|答辩|证据/.test(k+t)) return '证据 / 答辩';
  return '常用';
}
function palGroupOrder(g){
  return {'科研 / 数据':1,'具身 / 执行':2,'证据 / 答辩':3,'运维 / 状态':4,'安全 / 发布':5,'常用':6}[g]||9;
}
function palMeta(it){
  const g=palGroup(it);
  const ro=/api|docs|status|benchmark|defense|atlas|trace|evidence|public|对标|答辩|状态|材料/.test(((it&&it.k)||'')+' '+((it&&it.t)||''));
  return g+' · '+(ro?'只读证据面':'工作台入口');
}
function palRender(q){
  q=(q||'').trim().toLowerCase();
  const terms=q.split(/\s+/).filter(Boolean);
  const all=_palSmart(q).concat(_palBase());
  let recCount=0;                                  // 空查询时置顶的"最近"条数 (用于画分隔)
  if(!terms.length){
    const rec=_palRecentGet().map(t=>all.find(it=>it.t===t)).filter(Boolean);
    const seen=new Set(rec.map(it=>it.t));
    _palShown=rec.concat(all.filter(it=>!seen.has(it.t))).slice(0,12);
    recCount=Math.min(rec.length, _palShown.length);
  }else{
    _palShown=all.map(it=>[it,_palMatch(it,terms)]).filter(x=>x[1]>0)
      .sort((a,b)=>b[1]-a[1]).slice(0,12).map(x=>x[0]);
  }
  if(_palSel>=_palShown.length) _palSel=Math.max(0,_palShown.length-1);
  if(!recCount){
    _palShown=_palShown.map((it,i)=>({it,i,g:palGroup(it)}))
      .sort((a,b)=>(palGroupOrder(a.g)-palGroupOrder(b.g)) || (a.i-b.i))
      .map(x=>x.it);
  }
  const list=document.getElementById('palList'); list.innerHTML='';
  const cnt=document.getElementById('palCnt'); if(cnt) cnt.textContent=_palShown.length?(_palShown.length+' 项'):'';
  if(!_palShown.length){ list.innerHTML='<div class="oe-empty">无匹配 — 试试 "工单" / "资产" / "ops" / "扫码"</div>'; return; }
  let lastGroup='';
  _palShown.forEach((it,i)=>{
    if(recCount){ if(i===0) list.insertAdjacentHTML('beforeend','<div class="pal-sep">最近</div>');
      else if(i===recCount) list.insertAdjacentHTML('beforeend','<div class="pal-sep">常用</div>'); }
    if(!recCount){
      const g=palGroup(it);
      if(g!==lastGroup){ list.insertAdjacentHTML('beforeend','<div class="pal-sep">'+esc(g)+'</div>'); lastGroup=g; }
    }
    const d=document.createElement('div'); d.className='pali'+(i===_palSel?' sel':'');
    d.innerHTML='<span class="pic">'+it.ic+'</span><span class="plt">'+(terms.length?_palHi(it.t,terms):esc(it.t))+'<span class="pmeta">'+esc(palMeta(it))+'</span></span>';
    d.onclick=()=>_palExec(it);
    d.onmouseenter=()=>{ _palSel=i; [...list.querySelectorAll('.pali')].forEach((e,j)=>e.classList.toggle('sel',j===i)); };
    list.appendChild(d);
  });
}
document.getElementById('palIn').addEventListener('input', function(){ _palSel=0; palRender(this.value); });
document.getElementById('palIn').addEventListener('keydown', function(e){
  if(e.key==='ArrowDown'){ e.preventDefault(); _palSel=Math.min(_palSel+1,_palShown.length-1); palRender(this.value); }
  else if(e.key==='ArrowUp'){ e.preventDefault(); _palSel=Math.max(_palSel-1,0); palRender(this.value); }
  else if(e.key==='Enter'){ const it=_palShown[_palSel]; if(it){ _palExec(it); } }
  else if(e.key==='Escape'){ palClose(); }
});

/* 键盘 */
window.addEventListener('keydown', e=>{
  if((e.ctrlKey||e.metaKey) && (e.key==='k'||e.key==='K')){ e.preventDefault();
    const p=document.getElementById('pal');
    if(p.classList.contains('show')) palClose(); else palOpen();
    return; }
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.isContentEditable) return;
  if((e.key==='?'||(e.key==='/'&&e.shiftKey)) && !e.ctrlKey && !e.metaKey){ e.preventDefault(); kbdToggle(); return; }
  if(_gPending){ _gPending=false; clearTimeout(_gTimer);
    const gm={h:'home',o:'ops',m:'obs',t:'topo',q:'mq',a:'lab',c:'car',r:'arm',w:'twin',s:'sync',p:'repro',l:'highlight'};
    if(gm[e.key]){ e.preventDefault(); go(gm[e.key]); return; } }
  if(e.key==='Escape'){ if(booting){ skipBoot(); return; }
    if(document.getElementById('kbdHelp').classList.contains('show')){ kbdToggle(); return; }
    if(document.getElementById('scanOv').classList.contains('show')){ scanClose(); return; }
    if(document.getElementById('moreMenu').classList.contains('show')){ moreClose(); return; }
    if(NC_OPEN){ ncToggle(); return; }
    if(CP_OPEN){ cpToggle(); return; }
    if(KIOSK.on){ kioskStop(); return; }
    if(document.getElementById('pal').classList.contains('show')){ palClose(); return; }
    if(document.getElementById('woModal').classList.contains('show')){ woClose(); return; }
    if(document.getElementById('repModal').classList.contains('show')){ repClose(); return; }
    if(document.getElementById('adminModal').classList.contains('show')){ adminClose(); return; }
    if(document.getElementById('onb').classList.contains('show')){ onbEnd(); return; }
    if(TH.on){ tourExit(); return; } }
  if(booting && (e.key==='Enter'||e.key===' ')){ skipBoot(); return; }
  if(KIOSK.on){ if(e.key===' '){ e.preventDefault(); kioskPause(); return; }
    if(e.key==='ArrowRight'){ e.preventDefault(); kioskNext(); return; } return; }
  if(TH.on && (e.key===' ')){ e.preventDefault(); theaterPlayPause(); return; }
  if(e.key==='g'&&!e.ctrlKey&&!e.metaKey){ _gPending=true; clearTimeout(_gTimer); _gTimer=setTimeout(()=>{_gPending=false;},1300); return; }
  if(e.key==='0') go('home'); if(e.key==='4') go('highlight'); if(e.key==='5') go('ops'); if(e.key==='6') go('story'); if(e.key==='7') go('mq'); if(e.key==='8') go('assets'); if(e.key==='9') go('twin');
  if(e.key==='1') go('lab'); if(e.key==='2') go('car'); if(e.key==='3') go('arm');
});
/* ---- E6 平台体验: 快捷键帮助 + 密度切换 ---- */
let _gPending=false,_gTimer=null;
const KBD_SHORTCUTS=[
  ['Ctrl / ⌘ + K','全局命令面板 (搜索 / 跳转 / 操作)'],
  ['?','打开本快捷键帮助'],
  ['Esc','关闭弹层 / 返回'],
  ['g 然后 …','快速跳转: g h 总览 · g m 可观测 · g t 拓扑 · g o 运维 · g q 队列 · g p 复现 · g s 镜像 · g a/c/r 三机 · g w 孪生'],
  ['1 / 2 / 3','AI 脑 / 车载脑 / 机械臂'],
  ['0 / 4 / 5','总览 / 亮点 / 运维'],
  ['空格 (剧场/大屏)','播放 / 暂停'],
];
function kbdToggle(){
  const m=document.getElementById('kbdHelp'); if(!m) return;
  const open=m.classList.toggle('show');
  if(open){ const b=document.getElementById('kbdBody');
    if(b) b.innerHTML=KBD_SHORTCUTS.map(([k,v])=>'<div class="kbd-row"><kbd>'+esc(k)+'</kbd><span>'+esc(v)+'</span></div>').join('')
      +'<div class="kbd-dens"><span>界面密度</span><button class="kbd-db" onclick="densToggle()" id="kbdDensBtn"></button></div>';
    _densBtnSync(); }
}
function densToggle(){ const on=document.body.classList.toggle('compact');
  try{ localStorage.setItem('xrd_density', on?'compact':'comfy'); }catch(e){}
  _densBtnSync(); toast('界面密度: '+(on?'紧凑':'舒适'),'info'); }
function _densBtnSync(){ const b=document.getElementById('kbdDensBtn');
  if(b) b.textContent=document.body.classList.contains('compact')?'紧凑 ✓ (点切舒适)':'舒适 (点切紧凑)'; }
(function(){ try{ if(localStorage.getItem('xrd_density')==='compact') document.body.classList.add('compact'); }catch(e){} })();

