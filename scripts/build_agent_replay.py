"""Build a self-contained backup demo from selected REAL Agent traces, no API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Graph Agent 真实日志回放</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0c1422;color:#e6edf7;font:22px/1.5 'Microsoft YaHei',sans-serif}
main{width:1280px;height:720px;margin:auto;padding:30px 48px;overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155;padding-bottom:16px}
h1{font-size:32px;margin:0}header span{font-size:17px;color:#9cbed0}h2{font-size:29px;margin:18px 0 10px;color:#8ad7ed}p{margin:8px 0}.question{font-size:24px;min-height:68px}
.meta{color:#9db2cb;font-size:18px}.stage{color:#f7cf84;font-size:19px}.answer{white-space:pre-wrap;font-size:22px;max-height:144px;overflow:auto;padding:16px;background:#16253a;border-left:4px solid #58bfdc}
.evidence{font:17px/1.5 Consolas,'Microsoft YaHei',monospace;white-space:pre-wrap;max-height:120px;overflow:auto;color:#bed2e8}footer{position:absolute;bottom:24px;width:1184px;display:flex;justify-content:space-between;color:#9db2cb;font-size:16px}button{font-size:17px;background:#20364c;color:#e6edf7;border:1px solid #4b6c87;padding:8px 18px;cursor:pointer}nav{display:flex;gap:10px}.metric{display:flex;gap:40px;margin:20px 0;color:#8ad7ed;font-size:25px}.note{font-size:18px;color:#b5c4d6}
</style><main><header><h1>Graph Agent · 真实日志回放</h1><span>离线备份演示 / 非实时 API 调用</span></header>
<h2 id="heading"></h2><p class="question" id="question"></p><p class="meta" id="meta"></p><div class="metric" id="metrics"></div>
<p class="stage" id="stage"></p><div class="answer" id="answer"></div><p class="note" id="note"></p><div class="evidence" id="evidence"></div>
<footer><span id="source"></span><nav><button onclick="show(current-1)">上一页</button><button id="play" onclick="toggle()">自动播放</button><button onclick="show(current+1)">下一页</button></nav><span id="page"></span></footer></main>
<script type="application/json" id="records">__DATA__</script><script>
const screens=JSON.parse(document.getElementById('records').textContent);let current=0,timer=null;
function put(id,text){document.getElementById(id).textContent=text}
function show(index){current=(index+screens.length)%screens.length;const s=screens[current];put('heading',s.title);put('question',s.question);put('meta',s.meta);put('metrics',s.metrics);put('stage',s.stage);put('answer',s.answer);put('evidence',s.evidence);put('note',s.note);put('source','真实运行 '+s.run_id+' · '+s.question_id);put('page',(current+1)+' / '+screens.length);document.getElementById('answer').scrollTop=0;}
function toggle(){if(timer){clearInterval(timer);timer=null;put('play','自动播放')}else{timer=setInterval(()=>show(current+1),6000);put('play','暂停')}}
window.show=show;window.screenCount=screens.length;document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')show(current+1);if(e.key==='ArrowLeft')show(current-1)});show(0);
</script></html>'''


def screens_for_trace(trace: dict) -> list[dict]:
    answer_events = [event for event in trace["events"] if event["stage"] == "answer"]
    prompt = json.loads(answer_events[-1]["messages"][1]["content"])
    prediction = trace["prediction"]
    retrieval = [event for event in trace.get("shared_route_events", []) if event["stage"] == "retrieval"]
    stats = retrieval[-1].get("stats") or {} if retrieval else {}
    mode = trace["mode"]
    base = {"question": prompt["question"], "question_id": trace["question_id"], "run_id": trace["run_id"],
            "meta": f"模式 {mode} | 响应模型 {answer_events[-1].get('served_model') or '见原日志'} | 证据策略 {trace.get('evidence_policy', '无图谱证据')}",
            "metrics": f"图谱查询 {prediction['tool_calls']} 次     输入 {prediction['input_tokens']} tokens     输出 {prediction['output_tokens']} tokens",
            "note": "此处展示保存的实际模型输出。示例及路由检查不代表完整测试集成绩。"}
    decision = "不查询图谱，直接进入回答" if not prediction["tool_calls"] else "模型选择 query_graph，程序执行 B 的检索工具"
    route = {**base, "title": "自主决策与证据准备", "stage": decision,
             "answer": f"检索节点 {stats.get('retrieved_nodes', 0)}，边 {stats.get('retrieved_edges', 0)}\n上下文候选 {len(trace.get('context_candidate_ids', []))}，证据 {trace.get('context_utf8_bytes', 0)} 字节\n路由缓存命中：{trace.get('route_cache_hit', False)}",
             "evidence": "候选实体 ID：" + ", ".join(trace.get("context_candidate_ids", []))}
    final = {**base, "title": "结构化回答与真实证据", "stage": "模型回答与引用（保留原始边方向）",
             "answer": prediction["answer_text"],
             "evidence": "答案 ID：" + ", ".join(prediction["answer_ids"]) + "\n" + "\n".join(
                 f"{triple['head']} --{triple['relation']}--> {triple['tail']}" for triple in prediction["cited_evidence"])}
    return [route, final]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--include", action="append", required=True, help="Explicit question IDs for the demonstration")
    parser.add_argument("--output", type=Path, default=Path("outputs/agent-final/deliverables/Agent真实日志回放.html"))
    args = parser.parse_args()
    chosen = {}
    for directory in args.trace_dir:
        for path in directory.rglob("*.json"):
            trace = json.loads(path.read_text(encoding="utf-8"))
            if trace.get("question_id") in args.include and trace.get("status") == "ok":
                key = (trace["question_id"], trace["mode"])
                if key not in chosen or path.stat().st_mtime > chosen[key][0]:
                    chosen[key] = (path.stat().st_mtime, trace)
    screens = []
    for question_id in args.include:
        found = False
        for mode in ("llm", "text", "gnn"):
            if (question_id, mode) in chosen:
                found = True
                screens.extend(screens_for_trace(chosen[(question_id, mode)][1]))
        if not found:
            parser.error(f"No successful real trace for {question_id}")
    payload = json.dumps(screens, ensure_ascii=False).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    print(f"Replay: {args.output}; screens={len(screens)}; no API call made")


if __name__ == "__main__":
    main()
