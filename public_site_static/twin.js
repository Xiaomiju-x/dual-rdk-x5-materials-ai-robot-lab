/* G1 数字孪生 — 实验室全景 3D, 真尺寸真运动学, /api/twin 数据驱动.
 *
 * 真实物理写照的根据 (全部可溯源, 非美术臆造):
 *  - myCobot 280 六轴: 标准 DH 链 d1=131.22 a2=-110.4 a3=-96 d4=63.4 d5=75.05 d6=45.6 (mm)
 *    Source: Elephant Robotics myCobot 280 URDF (mycobot_ros) — 与 workstation/web/interlock.py 同一张表,
 *    关节角进来就是真实姿态 (FK 在浏览器逐帧算).
 *  - 双臂工位布局: interlock 配置默认 base_dx=400mm, base_yaw=180° (两臂对望).
 *  - 车位姿: NavCockpit /api/snapshot pose{x,y,yaw} (m/rad), 直接驱动场景小车.
 *  - 车体/雷达/相机为近似外观 (LD14 顶置 + Astra 前向 + 桅杆), 尺寸标注"近似".
 * 数据源诚实标注: 每台机器 chip 显示 真机遥测 / 演示数据(镜像 mock) / 离线.
 * GLB 钩子: /models/{mycobot,agv,x5rack}.glb 存在即换高模 (用户 CAD 导出后无缝接入).
 * 回落链: THREE 缺失/WebGL 失败 → #twFallback 文本面板, 数据表照常活.
 */
(function(){
'use strict';

/* ---- myCobot 280 标准 DH (mm), 与 interlock.py 逐行一致 ---- */
var DH = [
  /* alpha,        a,      d,      theta_offset */
  [ Math.PI/2,     0.0,    131.22, 0.0        ],
  [ 0.0,          -110.4,  0.0,   -Math.PI/2  ],
  [ 0.0,           -96.0,  0.0,    0.0        ],
  [ Math.PI/2,     0.0,    63.4,  -Math.PI/2  ],
  [-Math.PI/2,     0.0,    75.05,  Math.PI/2  ],
  [ 0.0,           0.0,    45.6,   0.0        ],
];
var MM = 0.001;                 // 场景单位 = 米
var BENCH = {x:-0.9, z:-0.55};  // 工位台中心 (场景坐标)
var BENCH_H = 0.75;             // 台面高 (近似)
var ARM_DX = 0.4;               // interlock 默认 base_dx_mm=400

function dhMat(alpha, a, d, theta){
  var ct=Math.cos(theta), st=Math.sin(theta), ca=Math.cos(alpha), sa=Math.sin(alpha);
  return [
    [ct, -st*ca,  st*sa, a*ct],
    [st,  ct*ca, -ct*sa, a*st],
    [0,       sa,     ca,    d],
  ];
}
function matMul(m1, m2){
  var out=[];
  for(var r=0;r<3;r++){ var row=[];
    for(var c=0;c<4;c++){ var v=0;
      for(var k=0;k<3;k++) v+=m1[r][k]*m2[k][c];
      if(c===3) v+=m1[r][3];
      row.push(v); }
    out.push(row); }
  return out;
}
/* 6 关节角(deg) → 7 个关节原点 (mm, 臂基座系) — interlock.fk_points 的 JS 移植 */
function fkPoints(deg){
  var m=[[1,0,0,0],[0,1,0,0],[0,0,1,0]];
  var pts=[[0,0,0]];
  for(var i=0;i<6;i++){
    var th=deg[i]*Math.PI/180 + DH[i][3];
    m=matMul(m, dhMat(DH[i][0], DH[i][1], DH[i][2], th));
    pts.push([m[0][3], m[1][3], m[2][3]]);
  }
  return pts;
}

/* ---- 模块态 ---- */
var inited=false, ok3d=false, visible=false;
var renderer, scene, cam, rootG;
var armsG={}, carG=null, carParts={}, rackCore=null, rackLight=null, furnGlow=null;
var sampleG=null, sampleTarget={x:0.8,z:0.5}, pathLine=null, xrdNode=null, plNode=null;
var trail=[], trailLine=null;
var latest=null;            // /api/twin 最新数据
var smooth={arm01:[0,0,0,0,0,0], arm02:[0,0,0,0,0,0], grip:{arm01:50,arm02:50},
            car:{x:0,y:0,yaw:0}};
var pollT=null;
var replayPct=100;
var pollSeq=0;
var camT={r:4.6, th:0.62, ph:0.85, cx:0, cy:0.45, cz:0};   // 球坐标目标
var camC={r:6.5, th:0.62, ph:0.85, cx:0, cy:0.45, cz:0};   // 当前 (lerp)
var drag=false, lx=0, ly=0, idleT=0;

function M(THREE,c,met,rou){ return new THREE.MeshStandardMaterial({color:c, metalness:met==null?0.3:met, roughness:rou==null?0.5:rou}); }

/* ---- myCobot 程序化模型: 关节球 + 链节管, 每帧按 FK 重摆 ---- */
function buildArm(THREE, name){
  var g=new THREE.Group();
  var bodyM=M(THREE,0xf4f6fb,0.25,0.4);          // myCobot 280 本体白
  var jointM=M(THREE,0xc7cdd9,0.5,0.35);
  var base=new THREE.Mesh(new THREE.CylinderGeometry(0.045,0.056,0.03,24), M(THREE,0xe2e8f0,0.4,0.4));
  base.position.y=0.015; base.castShadow=true; g.add(base);
  var segs=[], joints=[];
  for(var i=0;i<6;i++){
    var seg=new THREE.Mesh(new THREE.CylinderGeometry(0.024,0.024,1,14), bodyM);
    seg.castShadow=true; g.add(seg); segs.push(seg);
    var jr = i<3 ? 0.031 : 0.026;
    var j=new THREE.Mesh(new THREE.SphereGeometry(jr,18,14), jointM);
    j.castShadow=true; g.add(j); joints.push(j);
  }
  /* Atom 末端: LED 方块 + 双指夹爪 */
  var atom=new THREE.Group();
  var head=new THREE.Mesh(new THREE.BoxGeometry(0.034,0.02,0.034), M(THREE,0x334155,0.4,0.4));
  atom.add(head);
  var led=new THREE.Mesh(new THREE.BoxGeometry(0.02,0.004,0.02),
    new THREE.MeshStandardMaterial({color:0x22d3ee, emissive:0x22d3ee, emissiveIntensity:1.4}));
  led.position.y=0.012; atom.add(led);
  var fL=new THREE.Mesh(new THREE.BoxGeometry(0.006,0.034,0.012), jointM);
  var fR=fL.clone(); fL.position.set( 0.014,-0.026,0); fR.position.set(-0.014,-0.026,0);
  atom.add(fL); atom.add(fR); g.add(atom);
  return {g:g, segs:segs, joints:joints, atom:atom, fL:fL, fR:fR, led:led, name:name};
}

/* FK 点 (mm, 臂系: z 上) → 摆 mesh. 臂系→场景: (x,y,z)mm → (x, z, -y)m */
function poseArm(THREE, arm, deg, grip){
  var pts=fkPoints(deg);
  var v=[];
  for(var i=0;i<pts.length;i++)
    v.push(new THREE.Vector3(pts[i][0]*MM, pts[i][2]*MM, -pts[i][1]*MM));
  for(var s=0;s<6;s++){
    var a=v[s], b=v[s+1], d=a.distanceTo(b);
    var seg=arm.segs[s];
    if(d<0.004){ seg.visible=false; }
    else{
      seg.visible=true;
      seg.scale.set(1,d,1);
      seg.position.copy(a).lerp(b,0.5);
      seg.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), b.clone().sub(a).normalize());
    }
    arm.joints[s].position.copy(v[s+1]);
  }
  var tip=v[6], prev=v[5];
  arm.atom.position.copy(tip);
  arm.atom.quaternion.setFromUnitVectors(new THREE.Vector3(0,-1,0), tip.clone().sub(prev).normalize());
  var open = grip==null ? 0.5 : Math.max(0,Math.min(1,grip/100));
  arm.fL.position.x =  0.007+open*0.012;
  arm.fR.position.x = -0.007-open*0.012;
}

/* ---- 车: 近似外观 (LD14 顶置 + Astra 前向 + 桅杆), 真位姿驱动 ---- */
function buildCar(THREE){
  var g=new THREE.Group();
  var body=new THREE.Mesh(new THREE.BoxGeometry(0.30,0.10,0.24), M(THREE,0x2563eb,0.35,0.45));
  body.position.y=0.075; body.castShadow=true; g.add(body);
  var deck=new THREE.Mesh(new THREE.BoxGeometry(0.26,0.025,0.20), M(THREE,0x1e3a8a,0.4,0.4));
  deck.position.y=0.14; g.add(deck);
  var wg=new THREE.CylinderGeometry(0.04,0.04,0.025,16); wg.rotateX(Math.PI/2);
  var wm=M(THREE,0x1f2937,0.5,0.7), wheels=[];
  [[-0.10,0.04,0.125],[0.10,0.04,0.125],[-0.10,0.04,-0.125],[0.10,0.04,-0.125]].forEach(function(p){
    var w=new THREE.Mesh(wg,wm); w.position.set(p[0],p[1],p[2]); g.add(w); wheels.push(w);
  });
  var lidar=new THREE.Mesh(new THREE.CylinderGeometry(0.035,0.035,0.032,18), M(THREE,0x334155,0.45,0.4));
  lidar.position.set(0,0.175,0); g.add(lidar);                              // LD14/D300 顶置
  var swG=new THREE.CircleGeometry(0.5,26,0,Math.PI/3); swG.rotateX(-Math.PI/2);
  var sweep=new THREE.Mesh(swG, new THREE.MeshBasicMaterial({color:0x06b6d4,transparent:true,opacity:0.2,side:THREE.DoubleSide}));
  sweep.position.set(0,0.168,0); g.add(sweep);
  var astra=new THREE.Mesh(new THREE.BoxGeometry(0.025,0.03,0.16), M(THREE,0x0f172a,0.4,0.35));
  astra.position.set(0.155,0.10,0); g.add(astra);                            // Astra Pro 前向
  var mast=new THREE.Mesh(new THREE.CylinderGeometry(0.006,0.006,0.18,8), M(THREE,0x64748b,0.5,0.4));
  mast.position.set(-0.11,0.24,0); g.add(mast);
  var camHead=new THREE.Mesh(new THREE.BoxGeometry(0.03,0.022,0.03), M(THREE,0x0f172a,0.4,0.35));
  camHead.position.set(-0.11,0.34,0); g.add(camHead);                        // USB 相机桅杆
  return {g:g, wheels:wheels, sweep:sweep};
}

/* ---- 场景静物: 地板 / 工位台 / AI 脑机架 / 烧结炉 ---- */
function buildStatics(THREE){
  var ground=new THREE.Mesh(new THREE.PlaneGeometry(8,6), M(THREE,0xf0f3fa,0,0.96));
  ground.rotation.x=-Math.PI/2; ground.receiveShadow=true; rootG.add(ground);
  var grid=new THREE.GridHelper(8,32,0xc7d2fe,0xe6ebf7); grid.position.y=0.001; rootG.add(grid);

  /* 工位台 + AprilTag 贴片 */
  var bench=new THREE.Mesh(new THREE.BoxGeometry(1.5,BENCH_H,0.7), M(THREE,0xdbe3f0,0.15,0.7));
  bench.position.set(BENCH.x, BENCH_H/2, BENCH.z); bench.castShadow=true; bench.receiveShadow=true; rootG.add(bench);
  var tagM=new THREE.MeshBasicMaterial({color:0x0f172a});
  var tagW=new THREE.MeshBasicMaterial({color:0xffffff});
  [[-0.25,0.12],[0.25,0.12],[0,-0.18]].forEach(function(p){
    var w=new THREE.Mesh(new THREE.PlaneGeometry(0.06,0.06), tagW);
    w.rotation.x=-Math.PI/2; w.position.set(BENCH.x+p[0], BENCH_H+0.002, BENCH.z+p[1]); rootG.add(w);
    var b=new THREE.Mesh(new THREE.PlaneGeometry(0.044,0.044), tagM);
    b.rotation.x=-Math.PI/2; b.position.set(BENCH.x+p[0], BENCH_H+0.003, BENCH.z+p[1]); rootG.add(b);
  });

  /* AI 脑机架 (RDK X5 ×2 + 路由) */
  var rack=new THREE.Mesh(new THREE.BoxGeometry(0.4,0.86,0.34), M(THREE,0x7c3aed,0.4,0.4));
  rack.position.set(2.6,0.43,-1.6); rack.castShadow=true; rootG.add(rack);
  rackCore=new THREE.Mesh(new THREE.IcosahedronGeometry(0.11,0),
    new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xa78bfa,emissiveIntensity:1.1,metalness:0.2,roughness:0.3}));
  rackCore.position.set(2.6,0.62,-1.41); rootG.add(rackCore);
  rackLight=new THREE.PointLight(0xa78bfa,0.7,3); rackLight.position.set(2.6,0.6,-1.2); rootG.add(rackLight);

  /* 烧结炉 (马弗炉外观, 炉门琥珀热辉) */
  var furn=new THREE.Mesh(new THREE.BoxGeometry(0.5,0.62,0.5), M(THREE,0x94a3b8,0.5,0.45));
  furn.position.set(2.5,0.31,0.9); furn.castShadow=true; rootG.add(furn);
  furnGlow=new THREE.Mesh(new THREE.PlaneGeometry(0.26,0.2),
    new THREE.MeshStandardMaterial({color:0xfbbf24, emissive:0xf59e0b, emissiveIntensity:1.0}));
  furnGlow.position.set(2.5,0.30,1.151); rootG.add(furnGlow);
  var chimney=new THREE.Mesh(new THREE.CylinderGeometry(0.03,0.03,0.2,10), M(THREE,0x64748b,0.5,0.5));
  chimney.position.set(2.5,0.72,0.9); rootG.add(chimney);

  /* XRD / PL nodes + sample path. These are public context markers, not live controls. */
  xrdNode=new THREE.Mesh(new THREE.BoxGeometry(0.38,0.34,0.42), M(THREE,0x2563eb,0.35,0.42));
  xrdNode.position.set(2.1,0.17,1.65); xrdNode.castShadow=true; rootG.add(xrdNode);
  plNode=new THREE.Mesh(new THREE.CylinderGeometry(0.18,0.18,0.34,22), M(THREE,0x06b6d4,0.25,0.38));
  plNode.position.set(2.95,0.17,1.55); plNode.castShadow=true; rootG.add(plNode);
  sampleG=new THREE.Group();
  var sampleCore=new THREE.Mesh(new THREE.SphereGeometry(0.07,22,16),
    new THREE.MeshStandardMaterial({color:0xf59e0b, emissive:0xfbbf24, emissiveIntensity:0.9, roughness:0.32, metalness:0.08}));
  sampleG.add(sampleCore);
  var sampleRing=new THREE.Mesh(new THREE.TorusGeometry(0.12,0.008,8,36),
    new THREE.MeshBasicMaterial({color:0xf59e0b, transparent:true, opacity:0.55}));
  sampleRing.rotation.x=Math.PI/2; sampleG.add(sampleRing);
  sampleG.position.set(0.8,0.13,0.5); rootG.add(sampleG);
  var routePts=[new THREE.Vector3(2.6,0.08,-1.6),new THREE.Vector3(0.8,0.08,0.5),new THREE.Vector3(BENCH.x,0.1,BENCH.z),
    new THREE.Vector3(2.5,0.1,0.9),new THREE.Vector3(2.1,0.1,1.65),new THREE.Vector3(2.95,0.1,1.55)];
  pathLine=new THREE.Line(new THREE.BufferGeometry().setFromPoints(routePts),
    new THREE.LineBasicMaterial({color:0xf59e0b, transparent:true, opacity:0.55}));
  rootG.add(pathLine);
}

/* ---- GLB 钩子: /models/*.glb 存在即换高模 (CAD 导出后无缝接入) ---- */
function tryGLB(THREE){
  if(!window.XRD_TWIN_MODELS || !window.XRD_TWIN_MODELS.length) return;
  var REG=[ {url:'/models/mycobot.glb', anchor:function(o){ /* 双臂各放一份 */
              ['arm01','arm02'].forEach(function(k,i){ var c=o.clone();
                c.position.copy(armsG[k].g.position); c.rotation.copy(armsG[k].g.rotation);
                rootG.add(c); armsG[k].g.visible=false; }); }},
            {url:'/models/agv.glb', anchor:function(o){ carG.add(o); carParts.prog && (carParts.prog.visible=false); }},
            {url:'/models/x5rack.glb', anchor:function(o){ o.position.set(2.6,0,-1.6); rootG.add(o); }} ];
  function ensureLoader(cb){ if(THREE.GLTFLoader){cb();return;}
    var s=document.createElement('script'); s.src='/GLTFLoader.js'; s.onload=cb; s.onerror=function(){}; document.head.appendChild(s); }
  REG.forEach(function(r){
    fetch(r.url, {method:'HEAD'}).then(function(resp){
      if(!resp.ok) return;
      ensureLoader(function(){ if(!THREE.GLTFLoader) return;
        new THREE.GLTFLoader().load(r.url, function(g){
          var o=g.scene; o.traverse(function(m){ if(m.isMesh) m.castShadow=true; });
          r.anchor(o); window.toast && window.toast('🤖 已载入高模: '+r.url);
        }, undefined, function(){});
      });
    }).catch(function(){});
  });
}

/* ---- 初始化 ---- */
function init(){
  if(inited) return; inited=true;
  var THREE=window.THREE;
  var host=document.getElementById('twScene'), cv=document.getElementById('tw3d');
  if(!THREE || !host || !cv){ fail(); return; }
  try{
    var W=host.clientWidth||900, H=Math.max(380, Math.round(W*0.52));
    renderer=new THREE.WebGLRenderer({canvas:cv, antialias:true, alpha:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2)); renderer.setSize(W,H);
    renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    scene=new THREE.Scene();
    scene.fog=new THREE.Fog(0xeef2fb, 6.5, 16);   // U4: 远景柔化增景深
    cam=new THREE.PerspectiveCamera(42, W/H, 0.05, 60);
    scene.add(new THREE.AmbientLight(0xffffff,0.7));
    var key=new THREE.DirectionalLight(0xffffff,0.9); key.position.set(4,7,5); key.castShadow=true;
    key.shadow.mapSize.set(1024,1024);
    var sc=key.shadow.camera; sc.left=-5;sc.right=5;sc.top=5;sc.bottom=-5; scene.add(key);
    scene.add(function(){var p=new THREE.PointLight(0x7c3aed,0.35,18); p.position.set(-3,3,2); return p;}());
    scene.add(function(){var p=new THREE.PointLight(0x06b6d4,0.35,18); p.position.set(3,3,2); return p;}());
    rootG=new THREE.Group(); scene.add(rootG);

    buildStatics(THREE);

    /* 双臂上台: arm01 + arm02 相距 400mm, arm02 yaw 180° 对望 (interlock 默认布局) */
    armsG.arm01=buildArm(THREE,'arm01');
    armsG.arm01.g.position.set(BENCH.x-ARM_DX/2, BENCH_H, BENCH.z);
    rootG.add(armsG.arm01.g);
    armsG.arm02=buildArm(THREE,'arm02');
    armsG.arm02.g.position.set(BENCH.x+ARM_DX/2, BENCH_H, BENCH.z);
    armsG.arm02.g.rotation.y=Math.PI;
    rootG.add(armsG.arm02.g);

    var c=buildCar(THREE); carG=c.g; carParts=c; rootG.add(carG);
    carG.position.set(0.8,0,0.5);

    /* 车轨迹尾迹 */
    var tGeo=new THREE.BufferGeometry();
    tGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64*3),3));
    trailLine=new THREE.Line(tGeo, new THREE.LineBasicMaterial({color:0x2563eb,transparent:true,opacity:0.45}));
    rootG.add(trailLine);

    /* 交互: 拖动 orbit + 滚轮缩放 */
    cv.addEventListener('pointerdown',function(e){drag=true;lx=e.clientX;ly=e.clientY;idleT=0;cv.style.cursor='grabbing';});
    window.addEventListener('pointerup',function(){drag=false;cv.style.cursor='grab';});
    window.addEventListener('pointermove',function(e){ if(!drag)return;
      camT.th-=(e.clientX-lx)*0.008;
      camT.ph=Math.max(0.18,Math.min(1.35,camT.ph+(e.clientY-ly)*0.005));
      lx=e.clientX;ly=e.clientY;idleT=0; });
    cv.addEventListener('wheel',function(e){ e.preventDefault();
      camT.r=Math.max(1.2,Math.min(10,camT.r*(e.deltaY>0?1.08:0.92))); idleT=0; },{passive:false});
    window.addEventListener('resize',function(){
      var w=host.clientWidth||W,h=Math.max(380,Math.round(w*0.52));
      renderer.setSize(w,h); cam.aspect=w/h; cam.updateProjectionMatrix(); });

    tryGLB(THREE);
    ok3d=true;
    document.getElementById('twFallback').style.display='none';
    cv.style.display='block';
    loop();
  }catch(e){ fail(); }
}

function fail(){
  var f=document.getElementById('twFallback');
  if(f){ f.style.display='block';
    f.innerHTML='⚠ 3D 不可用 (WebGL/three.js 加载失败) — 下方数据表照常实时更新';
  }
  var cv=document.getElementById('tw3d'); if(cv) cv.style.display='none';
}

/* ---- 镜头预设 ---- */
var VIEWS={
  all:  {r:4.6, th:0.62, ph:0.85, cx:0,        cy:0.45, cz:0},
  bench:{r:1.6, th:0.45, ph:0.95, cx:BENCH.x,  cy:0.85, cz:BENCH.z},
  car:  {r:1.5, th:0.9,  ph:0.9,  cx:0.8,      cy:0.2,  cz:0.5,  follow:'car'},
  rack: {r:1.8, th:-0.5, ph:0.95, cx:2.55,     cy:0.5,  cz:-0.4},
  path: {r:3.1, th:0.62, ph:1.05, cx:1.25,     cy:0.22, cz:0.45},
  furnace:{r:1.7, th:-0.75,ph:0.9, cx:2.5,     cy:0.35, cz:0.9},
  cloud:{r:3.0, th:-0.25, ph:0.82, cx:2.7,     cy:0.85, cz:-1.05},
  fpv:  {fpv:true},   // U4: 车头第一人称 (雷达视角随车头朝向)
};
window.twView=function(k){
  var v=VIEWS[k]; if(!v) return;
  for(var key in v) if(key!=='follow'&&key!=='fpv') camT[key]=v[key];
  camT.follow = v.follow||null; camT.fpv = !!v.fpv; idleT=-20;
  document.querySelectorAll('.twvbtn').forEach(function(b){ b.classList.toggle('on', b.dataset.v===k); });
};

/* ---- 数据轮询 (仅孪生视图可见时) ---- */
function poll(){
  var seq=++pollSeq, want=replayPct;
  var url='/api/twin';
  if(want<100) url+='?replay='+encodeURIComponent(want);
  fetch(url, {cache:'no-store'}).then(function(r){return r.json();}).then(function(d){
    if(want!==replayPct) return;
    latest=d; renderChips(d); renderTable(d); renderMap(d); renderPlan(d); renderReplay(d);
  }).catch(function(){});
}
function startPoll(){ if(pollT) return; poll(); pollT=setInterval(poll, 2500); }
function stopPoll(){ if(pollT){ clearInterval(pollT); pollT=null; } }

var SRC_TXT={
  real:['live','ok'], live:['live','ok'],
  mirror:['mock','mock'], mock:['mock','mock'],
  replay:['replay','replay'],
  stale:['stale','mock'],
  down:['offline','down'], offline:['offline','down'],
  unknown:['unknown','down']
};
function renderChips(d){
  var box=document.getElementById('twChips'); if(!box) return;
  var names={lab:'🧠 AI 脑', car:'🚗 车载脑', arm:'🦾 双臂工位'};
  var h='';
  ['lab','car','arm'].forEach(function(k){
    var s=(d.source_label||{})[k] || (d.source||{})[k] || 'offline', t=SRC_TXT[s]||SRC_TXT.unknown;
    h+='<span class="twchip '+t[1]+'">'+names[k]+' · '+t[0]+'</span>';
  });
  if(d.replay) h+='<span class="twchip '+(d.replay.mode==='live'?'ok':'replay')+'">timeline · '+(d.replay.mode||'replay')+'</span>';
  box.innerHTML=h;
}
function fmt(x,n){ return (x==null||isNaN(x)) ? '—' : Number(x).toFixed(n==null?1:n); }
function twEsc(s){ return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];}); }
function renderTable(d){
  var el=document.getElementById('twData'); if(!el) return;
  var h='';
  ['arm01','arm02'].forEach(function(a){
    var j=(d.arms||{})[a];
    h+='<div class="twrow"><span class="twk">'+a+' 关节角</span><span class="twv">'+
       (j ? j.angles.map(function(x){return fmt(x,1)+'°';}).join(' · ')+' | 爪 '+fmt(j.gripper,0)+'%' : '—')+'</span></div>';
  });
  var c=d.car;
  h+='<div class="twrow"><span class="twk">车位姿 (x, y, yaw)</span><span class="twv">'+
     (c&&c.pose ? fmt(c.pose.x,2)+' m, '+fmt(c.pose.y,2)+' m, '+fmt(c.pose.yaw,2)+' rad' : '—')+'</span></div>';
  h+='<div class="twrow"><span class="twk">车速 / 电量</span><span class="twv">'+
     (c ? fmt((c.velocity||{}).linear,2)+' m/s · '+fmt(c.battery_pct,0)+'%' : '—')+'</span></div>';
  var s=d.sample||{};
  h+='<div class="twrow"><span class="twk">样品位置 / 阶段</span><span class="twv">'+
     (s.stage ? twEsc(s.stage)+' · '+fmt(s.progress_pct,1)+'%' : '—')+'</span></div>';
  el.innerHTML=h;
}
function pointAt(route,pct){
  if(!route||route.length<2) return {x:0,y:0};
  var p=Math.max(0,Math.min(100,Number(pct)||0))/100, seg=Math.min(route.length-2,Math.floor(p*(route.length-1)));
  var local=p*(route.length-1)-seg, a=route[seg], b=route[seg+1];
  return {x:a.x+(b.x-a.x)*local, y:a.y+(b.y-a.y)*local};
}
function renderMap(d){
  var box=document.getElementById('twMap'); if(!box) return;
  var m=d.map||{}, route=m.route||[], nodes=m.nodes||[], zones=m.zones||[];
  var pts=route.map(function(p){return p.x+','+p.y;}).join(' ');
  var sample=d.sample||{}, sp=sample.x!=null?sample:pointAt(route,(d.replay||{}).progress_pct||100);
  var h='<svg viewBox="0 0 100 100" role="img" aria-label="digital lab mini map">'+
    '<defs><linearGradient id="twRouteG" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#7c3aed"/><stop offset="1" stop-color="#06b6d4"/></linearGradient></defs>'+
    '<rect x="2" y="2" width="96" height="96" rx="10" fill="rgba(255,255,255,.55)" stroke="rgba(148,163,184,.35)"/>';
  zones.forEach(function(z){ h+='<rect class="tw-zone '+twEsc(z.risk||'')+'" x="'+z.x+'" y="'+z.y+'" width="'+z.w+'" height="'+z.h+'" rx="4"/>'; });
  h+='<polyline class="tw-route" points="'+pts+'"/>';
  nodes.forEach(function(n){ h+='<g class="tw-node"><circle cx="'+n.x+'" cy="'+n.y+'" r="3.4"/><text x="'+(n.x+4.5)+'" y="'+(n.y+1.8)+'">'+twEsc(n.label)+'</text></g>'; });
  h+='<g class="tw-sample"><circle cx="'+sp.x+'" cy="'+sp.y+'" r="4.6"/><text x="'+(sp.x+5.5)+'" y="'+(sp.y-4)+'">sample '+fmt(sample.progress_pct,0)+'%</text></g></svg>';
  box.innerHTML=h;
}
function renderPlan(d){
  var box=document.getElementById('twPlan'); if(!box) return;
  var plan=d.plan||[];
  box.innerHTML=plan.map(function(p){
    return '<div class="tw-plan-row"><b>'+twEsc(p.t)+'</b><span>'+twEsc(p.system)+' · '+twEsc(p.phase)+'</span><em>'+twEsc(p.source||'mock')+'</em><p>'+twEsc(p.action)+'</p></div>';
  }).join('') || '<div class="oe-empty">No replay plan available</div>';
}
function renderReplay(d){
  var lab=document.getElementById('twReplayLabel');
  if(lab) lab.textContent=(d.replay&&d.replay.mode==='live')?'Live telemetry':'Replay '+fmt((d.replay||{}).progress_pct,0)+'%';
}
window.twReplay=function(v){
  replayPct=Math.max(0,Math.min(100,Number(v)||0));
  var lab=document.getElementById('twReplayLabel'); if(lab) lab.textContent=replayPct>=100?'Live telemetry':'Replay '+fmt(replayPct,0)+'%';
  poll();
};
window.twReplayLive=function(){
  replayPct=100;
  var r=document.getElementById('twReplay'); if(r) r.value=100;
  poll();
};

/* ---- 主循环: 平滑插值 → FK → 渲染 (仅视图可见) ---- */
var tAcc=0;
function loop(){
  requestAnimationFrame(loop);
  if(!visible || !ok3d) return;
  tAcc+=0.016; idleT+=0.016;

  /* 关节角/位姿 平滑追踪到 /api/twin 最新值 */
  if(latest){
    ['arm01','arm02'].forEach(function(a){
      var j=(latest.arms||{})[a]; if(!j) return;
      for(var i=0;i<6;i++) smooth[a][i]+=(j.angles[i]-smooth[a][i])*0.06;
      if(j.gripper!=null) smooth.grip[a]+=(j.gripper-smooth.grip[a])*0.06;
    });
    var c=latest.car;
    if(c&&c.pose){
      /* NavCockpit 世界 (m) → 场景: 限制在地板活动区 (车在车区绕, 不开进台子) */
      var tx=Math.max(-0.3,Math.min(3.2, 0.8+c.pose.x*0.8));
      var tz=Math.max(-2.2,Math.min(2.2, 0.5+c.pose.y*0.8));
      smooth.car.x+=(tx-smooth.car.x)*0.05;
      smooth.car.yaw+=(((-c.pose.yaw)-smooth.car.yaw))*0.05;
      smooth.car.z=smooth.car.z==null?tz:smooth.car.z+(tz-smooth.car.z)*0.05;
    }
  }
  poseArm(window.THREE, armsG.arm01, smooth.arm01, smooth.grip.arm01);
  poseArm(window.THREE, armsG.arm02, smooth.arm02, smooth.grip.arm02);
  if(carG){
    carG.position.x=smooth.car.x; carG.position.z=(smooth.car.z!=null?smooth.car.z:0.5);
    carG.rotation.y=smooth.car.yaw;
    carParts.wheels.forEach(function(w){ w.rotation.z-=0.06; });
    carParts.sweep.rotation.y+=0.05;
    /* 尾迹 */
    if(tAcc%0.2<0.017){
      trail.push([carG.position.x, 0.02, carG.position.z]);
      if(trail.length>64) trail.shift();
      var pos=trailLine.geometry.attributes.position;
      for(var i=0;i<64;i++){ var p=trail[Math.min(i,trail.length-1)]||[0,0,0];
        pos.setXYZ(i,p[0],p[1],p[2]); }
      pos.needsUpdate=true;
      trailLine.geometry.setDrawRange(0,trail.length);
    }
  }
  if(rackCore){ rackCore.rotation.y+=0.012;
    rackCore.material.emissiveIntensity=1.0+Math.sin(tAcc*2)*0.4;
    rackLight.intensity=0.55+Math.sin(tAcc*2)*0.3; }
  if(furnGlow) furnGlow.material.emissiveIntensity=0.85+Math.sin(tAcc*1.3)*0.3;
  if(sampleG){
    if(latest && latest.sample && latest.sample.scene){
      sampleTarget.x=latest.sample.scene.x; sampleTarget.z=latest.sample.scene.z;
    }
    sampleG.position.x+=(sampleTarget.x-sampleG.position.x)*0.08;
    sampleG.position.z+=(sampleTarget.z-sampleG.position.z)*0.08;
    sampleG.position.y=0.13+Math.sin(tAcc*3)*0.018;
    sampleG.rotation.y+=0.035;
  }
  if(pathLine && pathLine.material) pathLine.material.opacity=0.38+Math.sin(tAcc*1.7)*0.12;
  if(xrdNode) xrdNode.rotation.y+=0.004;
  if(plNode) plNode.rotation.y-=0.006;

  /* U4 FPV: 车头第一人称 (相机骑在雷达上, 沿车头朝向看出去) */
  if(camT.fpv && carG){
    var yaw=carG.rotation.y, fx=Math.sin(yaw), fz=Math.cos(yaw);
    cam.position.set(carG.position.x - fx*0.05, 0.34, carG.position.z - fz*0.05);
    cam.lookAt(carG.position.x + fx*3, 0.22, carG.position.z + fz*3);
    renderer.render(scene, cam);
    return;
  }
  /* 相机: 球坐标 lerp + 跟车 + 闲转 */
  if(camT.follow==='car' && carG){ camT.cx=carG.position.x; camT.cz=carG.position.z; }
  if(!drag && idleT>6) camT.th+=0.0018;
  ['r','th','ph','cx','cy','cz'].forEach(function(k){ camC[k]+=(camT[k]-camC[k])*0.06; });
  cam.position.set(
    camC.cx + camC.r*Math.sin(camC.ph)*Math.sin(camC.th),
    camC.cy + camC.r*Math.cos(camC.ph),
    camC.cz + camC.r*Math.sin(camC.ph)*Math.cos(camC.th));
  cam.lookAt(camC.cx, camC.cy, camC.cz);
  renderer.render(scene, cam);
}

/* ---- 对外: revealView('twin') 时调 ---- */
window.twinShow=function(){
  visible=true;
  startPoll();
  if(window.THREE){ init(); }
  else{ /* three.min.js 还在路上 (app.js 已发起加载), 等它 */
    var n=0, t=setInterval(function(){
      if(window.THREE){ clearInterval(t); init(); }
      else if(++n>50){ clearInterval(t); fail(); }
    }, 200);
  }
};
window.twinHide=function(){ visible=false; stopPoll(); };
})();
