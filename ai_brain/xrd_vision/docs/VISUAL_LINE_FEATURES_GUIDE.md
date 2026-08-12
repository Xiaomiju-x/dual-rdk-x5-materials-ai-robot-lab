# 视觉线新功能移植指南 — 数值线参考

> 供数值线(`web_demo.py`)Claude Code参考，描述视觉线已实现的AI Agent、知识图谱、3D晶体可视化功能。

---

## 1. AI科学家Agent (双LLM协作: 千问VL + DeepSeek-R1)

### 架构

```
视觉线: 千问VL(看图→视觉特征) → DeepSeek-R1(推理+工具调用) → 结果
数值线: MLP分类(BPU→分类结果) → DeepSeek-R1(推理+工具调用) → 结果
```

数值线不需要千问VL(没有图像)，直接将MLP分类结果+峰位数据作为DeepSeek-R1的输入。

### DeepSeek-R1 API配置

```python
DEEPSEEK_R1_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_R1_KEY = "<API_KEY_FROM_ENVIRONMENT>"
DEEPSEEK_R1_MODEL = "deepseek-reasoner"  # R1推理模型
```

### API调用方式

```python
def call_deepseek_r1(messages, tools=None):
    """调用DeepSeek-R1, 支持thinking + tool-calling"""
    import requests
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_R1_KEY}",
    }
    payload = {
        "model": DEEPSEEK_R1_MODEL,
        "messages": messages,
        "max_tokens": 4000,
    }
    if tools:
        payload["tools"] = tools
    resp = requests.post(DEEPSEEK_R1_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]
    return {
        "content": msg.get("content", ""),
        "reasoning_content": msg.get("reasoning_content", ""),
        "tool_calls": msg.get("tool_calls", []),
    }
```

### 工具定义 (数值线版本)

```python
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_rag_knowledge",
            "description": "从197篇论文知识库中语义检索相关段落",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "match_pdf_card",
            "description": "在晶体学PDF卡片数据库中匹配衍射峰位置",
            "parameters": {
                "type": "object",
                "properties": {
                    "peak_positions": {"type": "string", "description": "主要峰位, 如'33.0,36.2,42.5'"},
                    "crystal_system": {"type": "string", "description": "晶系, 如'cubic'或'monoclinic'"}
                },
                "required": ["peak_positions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_next_experiment",
            "description": "基于分析结果建议下一步实验方向",
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {"type": "string", "description": "已识别的材料体系"},
                    "current_findings": {"type": "string", "description": "当前分析发现"}
                },
                "required": ["material"]
            }
        }
    }
]
```

### 工具执行 (数值线需适配)

```python
def execute_tool(name, args):
    if name == "query_rag_knowledge":
        return _rag.retrieve(args["query"], top_k=3)
    elif name == "match_pdf_card":
        # 数值线已有peak_matcher, 直接调用
        peaks = [float(p) for p in args["peak_positions"].split(",")]
        matcher = _load_peak_matcher()
        if matcher:
            results = matcher.match(peaks, category_hint=args.get("crystal_system", ""))
            if results:
                best = results[0]
                return f"匹配: {best['display_name']} ({best['space_group']}), 得分={best['score']:.3f}"
        return "未找到匹配"
    elif name == "suggest_next_experiment":
        return f"基于{args['material']}，建议: 1)变温XRD 2)调整掺杂浓度 3)光谱表征"
    return "未知工具"
```

### Agent ReAct循环

```python
def run_agent(input_description):
    """DeepSeek-R1 Agent循环: 思考→工具→观察→结论"""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": input_description}
    ]
    full_thinking = ""
    full_response = ""
    max_rounds = 2

    for round_i in range(max_rounds + 1):
        use_tools = AGENT_TOOLS if round_i < max_rounds else None
        resp = call_deepseek_r1(messages, tools=use_tools)

        thinking = resp.get("reasoning_content", "")
        if thinking:
            full_thinking += f"\n🤔 推理第{round_i+1}轮:\n{thinking}\n"

        tool_calls = resp.get("tool_calls", [])
        content = resp.get("content", "")

        if not tool_calls:
            full_response = content
            break

        # 执行工具
        assistant_msg = {"role": "assistant", "content": content or ""}
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        for tc in tool_calls:
            func = tc.get("function", {})
            func_name = func.get("name", "")
            func_args = json.loads(func.get("arguments", "{}"))
            full_thinking += f"🔧 调用工具: {func_name}({func_args})\n"
            result = execute_tool(func_name, func_args)
            full_thinking += f"📋 结果: {result[:300]}\n"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })

    if not full_response:
        full_response = full_thinking

    return full_thinking, full_response
```

### 数值线的Agent输入构造

```python
# 在run_pipeline()的报告生成阶段替换call_deepseek()
input_desc = f"""MLP分类结果: {pred_label} (置信度{pred_conf:.2%}, 模式={mode})
峰位数据: {', '.join(f'{p:.1f}°' for p in peak_positions[:10])}
峰位匹配: {match_data['display_name'] if match_data else '未匹配'} ({match_data['space_group'] if match_data else ''})

请基于以上数据，自主调用工具进行深度分析。"""

thinking, response = run_agent(input_desc)
report = response  # 最终报告
```

### System Prompt (数值线版)

```python
AGENT_SYSTEM_PROMPT = (
    "你是一个自主推理的AI晶体学家，部署在RDK X5嵌入式平台上。\n\n"
    "MLP分类器(BPU加速)已经给出了初步分类结果，你的任务是进行深度验证和分析。\n\n"
    "你拥有以下工具:\n"
    "- query_rag_knowledge: 检索197篇论文知识库\n"
    "- match_pdf_card: 验证峰位与标准卡片匹配\n"
    "- suggest_next_experiment: 建议下一步实验\n\n"
    "工作流程:\n"
    "1. 接受MLP分类结果和峰位数据\n"
    "2. 调用query_rag_knowledge获取该材料详细知识\n"
    "3. 调用match_pdf_card验证峰位\n"
    "4. 输出结构化报告:\n"
    "**材料判定**: 材料名称和化学式\n"
    "**晶体结构**: 晶系、空间群、离子占位\n"
    "**峰位分析**: 关键峰位和晶面指数\n"
    "**应用价值**: 科研意义\n"
    "引用工具返回的[Ref.N]。控制在300字以内。"
)
```

### 降级策略

```python
try:
    thinking, response = run_agent(input_desc)
    report_mode = "AI Agent(DeepSeek-R1)"
except Exception:
    # 降级回原来的call_deepseek()
    report = call_deepseek(prompt)
    report_mode = "在线(DeepSeek降级)"
```

---

## 2. 知识图谱 (带动画效果)

### 实现方式

不使用WebGL 3D库(兼容性差)，用纯HTML/CSS实现分组卡片式知识图谱，带丰富动画效果。零外部依赖。

### 后端API

```python
@app.route('/api/knowledge_graph')
def api_knowledge_graph():
    """知识图谱数据"""
    # 从chunks.json的元数据构建图谱
    # 节点类型: material, crystal, structure, dopant, property, tech, paper, detected
    # 每个分析完成后动态添加detected节点
    # paper节点按连接数排序只保留Top40, 避免图谱过大
    # 返回 {"nodes": [...], "links": [...]}
```

视觉线的完整后端实现在 `deploy_xrd_system.py` 的 `_build_knowledge_graph()` 函数。

### 前端JS渲染函数 (完整代码，可直接复用)

```javascript
function renderKGHtml(data){
    const cm={crystal:'#3b82f6',material:'#f59e0b',property:'#8b5cf6',
              dopant:'#10b981',tech:'#ef4444',detected:'#f97316',
              structure:'#06b6d4',paper:'#94a3b8'};
    const gnames={crystal:'💎 晶系',material:'🧪 材料',property:'✨ 性能',
                  dopant:'⚛ 掺杂离子',tech:'🔧 技术',detected:'📊 分析结果',
                  structure:'🔬 结构类型',paper:'📄 论文'};
    // 每种分组对应不同动画类
    const ganims={material:'kg-pulse',structure:'kg-glow',crystal:'kg-spin-y',
                  dopant:'kg-bounce',property:'kg-shimmer',tech:'kg-pulse',
                  detected:'kg-pop',paper:''};
    // 按group分组
    const groups={};
    data.nodes.forEach(n=>{
        if(!groups[n.group]) groups[n.group]=[];
        groups[n.group].push(n);
    });
    // 计算连接数(决定节点大小)
    const linkCount={};
    data.links.forEach(l=>{
        linkCount[l.source]=(linkCount[l.source]||0)+1;
        linkCount[l.target]=(linkCount[l.target]||0)+1;
    });
    let html='<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;perspective:800px;">';
    const order=['material','structure','crystal','dopant','property','tech','detected','paper'];
    order.forEach((g,gi)=>{
        if(!groups[g]) return;
        const nodes=groups[g];
        // paper组只显示连接数>2的前12个
        const show=g==='paper'?nodes.filter(n=>(linkCount[n.id]||0)>2).slice(0,12):nodes;
        if(show.length===0) return;
        const c=cm[g]||'#94a3b8';
        const anim=ganims[g]||'';
        // 分组卡片: 渐变背景 + 入场动画(依次延迟淡入)
        html+='<div class="kg-group" style="background:linear-gradient(135deg,#ffffff,'+c+'08);'
            +'border-radius:12px;padding:10px 12px;min-width:110px;max-width:200px;'
            +'border:1.5px solid '+c+'30;box-shadow:0 2px 8px '+c+'15;'
            +'animation:kg-fadein 0.5s ease '+(gi*0.1)+'s both;">';
        html+='<div style="font-size:11px;font-weight:700;color:'+c+';margin-bottom:6px;text-align:center;'
            +'border-bottom:1px solid '+c+'20;padding-bottom:4px;">'
            +(gnames[g]||g)+'<span style="font-weight:400;opacity:0.6;margin-left:4px;">('+nodes.length+')</span></div>';
        html+='<div style="display:flex;flex-wrap:wrap;gap:4px;justify-content:center;">';
        show.forEach((n,ni)=>{
            const lc=linkCount[n.id]||0;
            const sz=Math.max(9,Math.min(13,9+lc));
            const delay=(ni*0.05+gi*0.1).toFixed(2);
            // 每个节点: 圆角标签 + 动画 + hover放大光晕
            html+='<span class="kg-node '+(anim||'')+'" style="display:inline-block;padding:3px 8px;'
                +'border-radius:12px;font-size:'+sz+'px;font-weight:600;'
                +'background:'+c+'15;color:'+c+';border:1px solid '+c+'35;'
                +'white-space:nowrap;cursor:default;transition:all .2s;'
                +'animation-delay:'+delay+'s;" '
                +'title="'+n.name+' (连接:'+lc+')" '
                +'onmouseover="this.style.transform=\'scale(1.15)\';this.style.boxShadow=\'0 0 12px '+c+'50\'" '
                +'onmouseout="this.style.transform=\'scale(1)\';this.style.boxShadow=\'none\'">'
                +n.name+'</span>';
        });
        html+='</div></div>';
    });
    html+='</div>';
    // 底部彩虹流动线 + 统计
    html+='<div style="margin-top:10px;text-align:center;">';
    html+='<div class="kg-flow-line"></div>';
    html+='<div style="font-size:11px;color:#94a3b8;margin-top:6px;">'
        +'<span class="icon-spin" style="font-size:13px;">🌐</span> '
        +data.nodes.length+' 个实体 · '+data.links.length+' 条关系 · 197篇论文语义知识库</div>';
    html+='</div>';
    document.getElementById('knowledgeGraph').innerHTML=html;
}
```

### CSS动画 (完整版，可直接复用)

```css
/* ===== 知识图谱动画 ===== */

/* 入场渐入 */
@keyframes kg-fadein{
    from{opacity:0;transform:translateY(10px) scale(0.95)}
    to{opacity:1;transform:translateY(0) scale(1)}
}

/* 材料节点: 脉冲呼吸光晕 */
@keyframes kg-pulse-anim{
    0%,100%{box-shadow:0 0 0 0 currentColor}
    50%{box-shadow:0 0 8px 2px currentColor}
}
.kg-pulse{animation:kg-pulse-anim 2.5s ease-in-out infinite;}

/* 结构类型: 文字发光 */
@keyframes kg-glow-anim{
    0%,100%{opacity:0.85}
    50%{opacity:1;text-shadow:0 0 6px currentColor}
}
.kg-glow{animation:kg-glow-anim 3s ease-in-out infinite;}

/* 掺杂离子: 上下弹跳 */
@keyframes kg-bounce-anim{
    0%,100%{transform:translateY(0)}
    50%{transform:translateY(-2px)}
}
.kg-bounce{animation:kg-bounce-anim 2s ease-in-out infinite;}

/* 性能: 光泽扫过 */
@keyframes kg-shimmer-anim{
    0%{background-position:-100%}
    100%{background-position:200%}
}
.kg-shimmer{
    background:linear-gradient(90deg,transparent 30%,rgba(255,255,255,0.3) 50%,transparent 70%);
    background-size:200% 100%;
    animation:kg-shimmer-anim 3s linear infinite;
}

/* 晶系: Y轴3D旋转 */
@keyframes kg-spin-y-anim{
    from{transform:rotateY(0)}
    to{transform:rotateY(360deg)}
}
.kg-spin-y{animation:kg-spin-y-anim 6s linear infinite;display:inline-block;}

/* 分析结果: 缩放弹出 */
@keyframes kg-pop-anim{
    0%,100%{transform:scale(1)}
    50%{transform:scale(1.08)}
}
.kg-pop{animation:kg-pop-anim 2s ease-in-out infinite;}

/* 底部彩虹流动线 */
@keyframes kg-flow{
    0%{background-position:0% 50%}
    100%{background-position:200% 50%}
}
.kg-flow-line{
    height:3px;border-radius:2px;
    background:linear-gradient(90deg,#3b82f6,#8b5cf6,#10b981,#f59e0b,#ef4444,#3b82f6);
    background-size:200% 100%;
    animation:kg-flow 3s linear infinite;
    margin:0 auto;width:80%;opacity:0.6;
}

/* ===== 全局装饰动画 (可用于其他卡片) ===== */

/* 慢速旋转图标 */
@keyframes spin-slow{from{transform:rotate(0)}to{transform:rotate(360deg)}}
.icon-spin{display:inline-block;animation:spin-slow 4s linear infinite;}

/* 呼吸脉动 */
@keyframes pulse-dot{0%,100%{opacity:0.6}50%{opacity:1}}
.icon-pulse{display:inline-block;animation:pulse-dot 2s ease-in-out infinite;}

/* 上下浮动 */
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}
.icon-float{display:inline-block;animation:float 3s ease-in-out infinite;}
```

### 动画效果说明

| 节点类型 | 动画效果 | CSS类名 | 视觉效果 |
|---------|---------|---------|---------|
| 🧪 材料 | 脉冲呼吸 | `kg-pulse` | 光晕周期性扩散 |
| 🔬 结构类型 | 文字发光 | `kg-glow` | 文字周期性发亮 |
| 💎 晶系 | Y轴旋转 | `kg-spin-y` | 3D翻转效果 |
| ⚛ 掺杂离子 | 上下弹跳 | `kg-bounce` | 轻微跳动 |
| ✨ 性能 | 光泽扫过 | `kg-shimmer` | 高光从左到右扫过 |
| 📊 分析结果 | 缩放弹出 | `kg-pop` | 周期性放大缩小 |
| 📄 论文 | 无动画 | — | 静态显示 |
| 分组卡片 | 入场渐入 | `kg-fadein` | 从下方淡入, 每组延迟0.1s |
| 所有节点 | hover放大 | inline style | 悬停放大1.15倍+光晕 |
| 底部线条 | 彩虹流动 | `kg-flow-line` | 蓝→紫→绿→金→红循环流动 |
| 🌐 图标 | 慢速旋转 | `icon-spin` | 4秒一圈匀速旋转 |

---

## 3. 3D晶体结构可视化

### CDN

```html
<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
```

### CIF文件

两个CIF文件已准备好，需要复制到数值线目录:
- `crystal_data/SYGO.cif` — 单斜C2, 40个原子(12Sr+4Y+8Ga+40O), 层状钙钛矿
- `crystal_data/YCAS.cif` — 立方Ia-3d, 119个原子(24Y+12Ca+20Al+12Si+48O), 石榴石

### 后端路由

```python
@app.route('/api/crystal/<name>')
def api_crystal(name):
    safe_name = name.replace('/', '').replace('\\', '').replace('..', '')
    cif_path = os.path.join(_SCRIPT_DIR, "crystal_data", f"{safe_name}.cif")
    if not os.path.exists(cif_path):
        return jsonify({"error": "未找到"}), 404
    with open(cif_path, 'r', encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/plain')
```

### 前端HTML卡片

```html
<div class="card" id="crystalCard" style="display:none;">
    <div class="card-hd blue">晶体结构3D可视化
        <span id="crystalLabel" style="margin-left:auto;"></span>
    </div>
    <div class="card-bd">
        <div id="crystal3d" style="width:100%;height:400px;"></div>
        <div id="crystalInfo"></div>
    </div>
</div>
```

### 前端JS

```javascript
function show3DCrystal(material_label){
    if(typeof $3Dmol === 'undefined') return;
    let mat = '';
    if(/SYGO|Sr₃Y|单斜/.test(material_label)) mat = 'SYGO';
    else if(/YCAS|石榴石|garnet/i.test(material_label)) mat = 'YCAS';
    if(!mat) return;

    document.getElementById('crystalCard').style.display='block';
    document.getElementById('crystalLabel').textContent=mat;

    fetch('/api/crystal/'+mat).then(r=>r.text()).then(cif=>{
        const el = document.getElementById('crystal3d');
        el.innerHTML = '';
        let viewer = $3Dmol.createViewer(el, {backgroundColor:'#ffffff'});
        viewer.addModel(cif, 'cif', {doAssembly:true, duplicateAssemblyAtoms:true});
        viewer.setStyle({}, {
            sphere:{radius:0.35, colorscheme:'Jmol'},
            stick:{radius:0.12, colorscheme:'Jmol'}
        });
        viewer.addUnitCell({box:{color:'#94a3b8'}});
        viewer.zoomTo();
        viewer.render();
        viewer.spin('y', 0.5);  // 自动旋转
    });
}
```

### 数值线触发时机

```python
# 分析完成后, 根据分类结果显示对应晶体
# 在displayResult()的JS中:
if(cls.final_label === 'garnet' || cls.final_label === 'YCAS')
    show3DCrystal('YCAS');
else if(cls.final_label === 'non_garnet' || cls.final_label === 'SYGO')
    show3DCrystal('SYGO');
```

---

## 4. QR码分享 (附赠)

### CDN

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
```

### 后端

```python
@app.route('/api/report_view')
def api_report_view():
    """在线查看报告(QR扫码用, 不下载)"""
    # 和/api/export相同内容, 但不设Content-Disposition
    return _generate_report(history)
```

### 前端

```javascript
new QRCode(document.getElementById('qrcode'), {
    text: location.origin + '/api/report_view',
    width: 100, height: 100
});
```

---

## 5. 需要传的文件清单

### 已有(RAG相关, 之前已传):
- `rag_engine.py` → `~/xrd1/`
- `xrd_knowledge/embeddings/chunks.json` → `~/xrd1/xrd_knowledge/embeddings/`
- `xrd_knowledge/embeddings/vectors.npy` → `~/xrd1/xrd_knowledge/embeddings/`

### 新增(本次需传):
- `crystal_data/SYGO.cif` → `~/xrd1/crystal_data/`
- `crystal_data/YCAS.cif` → `~/xrd1/crystal_data/`

### 参考文档(给Claude Code看):
- `RAG_REFERENCE.md` — RAG引擎接入指南
- `VOICE_INTERACTION_GUIDE.md` — 语音交互移植指南
- `VISUAL_LINE_FEATURES_GUIDE.md` — 本文档(AI Agent+知识图谱+3D晶体)
