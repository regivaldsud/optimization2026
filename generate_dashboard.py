# -*- coding: utf-8 -*-
import json, os, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = json.load(open(os.path.join(ROOT,"results","results.json"), encoding="utf-8"))
cfg  = RES["config"]
machine  = RES["machine"]
jobshop  = RES["jobshop"]
flowshop = RES["flowshop"]

# ── helpers ────────────────────────────────────────────────────────────────
def fn(v, nd=2):
    if v is None: return "—"
    try:
        f = float(v)
        if f != f: return "—"
        return f"{f:.{nd}f}".rstrip("0").rstrip(".") if nd else str(int(f))
    except: return str(v)

def badge(s):
    s = (s or "").upper()
    c = "ok" if "OPTIMAL" in s else ("bad" if ("INFEAS" in s or "ERROR" in s) else "warn")
    return f'<span class="badge {c}">{s}</span>'

def is_opt(r): return "OPTIMAL" in (r.get("status") or "").upper()
def gpct(r): return (r.get("gap") or 0) * 100

# ── statistics ─────────────────────────────────────────────────────────────
n_m = len(machine); n_js = len(jobshop); n_fs = len(flowshop)
n_total = n_m + n_js + n_fs
m_opt  = sum(1 for r in machine  if is_opt(r))
js_opt = sum(1 for r in jobshop  if is_opt(r))
fs_opt = sum(1 for r in flowshop if is_opt(r))
n_opt  = m_opt + js_opt + fs_opt

m_time  = sum((r.get("runtime") or 0) for r in machine)
js_time = sum((r.get("runtime") or 0) for r in jobshop)
fs_time = sum((r.get("runtime") or 0) for r in flowshop)
tot_time = m_time + js_time + fs_time

m_avg_gap  = sum(gpct(r) for r in machine)  / n_m
fs_avg_gap = sum(gpct(r) for r in flowshop) / n_fs

js_rows = []
for r in jobshop:
    obj, opt = r.get("objective"), r.get("optimum")
    go = round((100*(obj-opt)/opt), 4) if (obj and opt) else None
    js_rows.append((r, go))
js_valid_gaps = [g for _,g in js_rows if g is not None]
js_avg_gap_opt = sum(js_valid_gaps)/len(js_valid_gaps) if js_valid_gaps else 0

transition_n = None
for r in sorted(machine, key=lambda r: (r["n"], r["instance"])):
    if not is_opt(r):
        transition_n = r["n"]; break

# ── aggregate machine parameter ranges (for Instances section) ─────────────
m_r = []; m_p = []; m_d = []
for r in machine:
    p = os.path.join(ROOT,"data","machine", r["instance"]+".json")
    try:
        raw = json.load(open(p, encoding="utf-8"))
        m_r += raw.get("release", [])
        m_p += raw.get("duration", [])
        m_d += raw.get("due", [])
    except: pass
m_r_rng = f"{min(m_r)}–{max(m_r)}" if m_r else "—"
m_p_rng = f"{min(m_p)}–{max(m_p)}" if m_p else "—"
m_d_rng = f"{min(m_d)}–{max(m_d)}" if m_d else "—"

# inst_book schedule detail
ib_raw = json.load(open(os.path.join(ROOT,"data","machine","inst_book.json"), encoding="utf-8"))
ib_res = next((r for r in machine if r["instance"]=="inst_book"), None)
ib_obj = ib_res["objective"] if ib_res else "—"
ib_rows = ""
if ib_res and ib_res.get("schedule"):
    for op in ib_res["schedule"]:
        jidx = ib_raw["jobs"].index(op["job"]) if op["job"] in ib_raw["jobs"] else -1
        rj = ib_raw["release"][jidx] if jidx >= 0 else "—"
        pj = ib_raw["duration"][jidx] if jidx >= 0 else "—"
        dj = ib_raw["due"][jidx] if jidx >= 0 else "—"
        tj = op.get("tardiness", 0)
        tc = "c-bad" if tj > 0 else "c-ok"
        ib_rows += (f"<tr><td class='job-cell'>{op['job']}</td>"
                    f"<td class='num'>{rj}</td><td class='num'>{pj}</td>"
                    f"<td class='num'>{dj}</td><td class='num'>{fn(op['start'])}</td>"
                    f"<td class='num'>{fn(op['finish'])}</td>"
                    f"<td class='num {tc}'>{fn(tj)}</td></tr>")

# JSPLIB instance provenance
JS_SOURCE = {
    'abz5':'Adams, Balas & Zawack (1988)', 'abz6':'Adams, Balas & Zawack (1988)',
    'ft06':'Fisher & Thompson (1963)', 'ft10':'Fisher & Thompson (1963)', 'ft20':'Fisher & Thompson (1963)',
    'la01':'Lawrence (1984)', 'la02':'Lawrence (1984)', 'la03':'Lawrence (1984)', 'la04':'Lawrence (1984)',
    'orb01':'Applegate & Cook (1991)',
}

# ── table builders ─────────────────────────────────────────────────────────
def tbl_machine():
    rows = ""
    for r in machine:
        st = (r.get("status") or "").upper()
        g  = gpct(r)
        gc = "c-bad" if g > 50 else ("c-warn" if g > 0 else "c-ok")
        rows += (f"<tr data-s='{st}'>"
                 f"<td><code class='inst-code'>{r['instance'].replace('inst_','')}</code></td>"
                 f"<td class='num'>{r['n']}</td>"
                 f"<td class='num'>{fn(r['objective'])}</td>"
                 f"<td class='num'>{fn(r['bound'])}</td>"
                 f"<td class='num {gc}'>{fn(g,3)}%</td>"
                 f"<td class='num'>{fn(r['runtime'])}s</td>"
                 f"<td>{badge(st)}</td></tr>")
    return rows

def tbl_jobshop():
    rows = ""
    for r, go in js_rows:
        st  = (r.get("status") or "").upper()
        gc  = "c-ok" if (go or 0)==0 else ("c-warn" if (go or 0)<20 else "c-bad")
        gs  = fn(go,2)+"%" if go is not None else "—"
        rows += (f"<tr data-s='{st}'>"
                 f"<td><code class='inst-code'>{r['instance']}</code></td>"
                 f"<td class='num'>{r['jobs']}×{r['machines']}</td>"
                 f"<td class='num'>{fn(r['objective'])}</td>"
                 f"<td class='num'>{fn(r['bound'])}</td>"
                 f"<td class='num'>{r.get('optimum','—')}</td>"
                 f"<td class='num {gc}'>{gs}</td>"
                 f"<td class='num'>{fn(r['runtime'])}s</td>"
                 f"<td>{badge(st)}</td></tr>")
    return rows

def tbl_flowshop():
    rows = ""
    for r in flowshop:
        st = (r.get("status") or "").upper()
        g  = gpct(r)
        gc = "c-bad" if g > 30 else ("c-warn" if g > 5 else "c-ok")
        rows += (f"<tr data-s='{st}'>"
                 f"<td><code class='inst-code'>{r['instance'].replace('problem_','')}</code></td>"
                 f"<td class='num'>{r['jobs']}×{r['machines']}</td>"
                 f"<td class='num'>{fn(r['objective'])}</td>"
                 f"<td class='num'>{fn(r['bound'])}</td>"
                 f"<td class='num {gc}'>{fn(g,3)}%</td>"
                 f"<td class='num'>{fn(r['runtime'])}s</td>"
                 f"<td>{badge(st)}</td></tr>")
    return rows

def tbl_jsplib():
    rows = ""
    for r, go in js_rows:
        st  = (r.get("status") or "").upper()
        gc  = "c-ok" if (go or 0)==0 else ("c-warn" if (go or 0)<20 else "c-bad")
        gs  = fn(go,2)+"%" if go is not None else "—"
        rows += (f"<tr data-s='{st}'>"
                 f"<td><code class='inst-code'>{r['instance']}</code></td>"
                 f"<td class='num'>{r['jobs']}×{r['machines']}</td>"
                 f"<td class='num bold-num'>{r.get('optimum','—')}</td>"
                 f"<td class='num'>{fn(r['objective'])}</td>"
                 f"<td class='num {gc} bold-num'>{gs}</td>"
                 f"<td class='num'>{fn(r['bound'])}</td>"
                 f"<td class='num'>{fn(r['runtime'])}s</td>"
                 f"<td>{badge(st)}</td></tr>")
    return rows

def machine_params_table():
    rows = ""
    for r in machine:
        inst = r["instance"]
        try:
            raw = json.load(open(os.path.join(ROOT,"data","machine",inst+".json"), encoding="utf-8"))
            np_ = raw.get("n", r["n"])
            rjs, pjs, djs = raw.get("release",[]), raw.get("duration",[]), raw.get("due",[])
            rng = f"{min(rjs)}–{max(rjs)}" if rjs else "—"
            png = f"{min(pjs)}–{max(pjs)}" if pjs else "—"
            dng = f"{min(djs)}–{max(djs)}" if djs else "—"
        except:
            np_, rng, png, dng = r["n"], "—", "—", "—"
        st = (r.get("status") or "").upper()
        gc = "c-ok" if is_opt(r) else "c-warn"
        rows += (f"<tr data-s='{st}'>"
                 f"<td><code class='inst-code'>{inst.replace('inst_','')}</code></td>"
                 f"<td class='num'>{np_}</td>"
                 f"<td class='num'>{rng}</td><td class='num'>{png}</td><td class='num'>{dng}</td>"
                 f"<td class='num {gc}'>{fn(r['objective'])}</td>"
                 f"<td class='num'>{fn(r['runtime'])}s</td>"
                 f"<td>{badge(st)}</td></tr>")
    return rows

def tbl_js_instances():
    rows = ""
    for r in jobshop:
        src = JS_SOURCE.get(r["instance"], "JSPLIB")
        rows += (f"<tr><td><code class='inst-code'>{r['instance']}</code></td>"
                 f"<td class='num'>{r['jobs']}×{r['machines']}</td>"
                 f"<td class='num'>{r['jobs']*r['machines']}</td>"
                 f"<td class='num bold-num'>{r.get('optimum','—')}</td>"
                 f"<td>{src}</td></tr>")
    return rows

def tbl_fs_instances():
    rows = ""
    for r in flowshop:
        rows += (f"<tr><td><code class='inst-code'>{r['instance'].replace('problem_','')}</code></td>"
                 f"<td class='num'>{r['jobs']}</td><td class='num'>{r['machines']}</td>"
                 f"<td class='num'>{r['jobs']*r['machines']}</td>"
                 f"<td>Permutation flow shop (CSV)</td></tr>")
    return rows

def pick_gantt(rows, name):
    for r in rows:
        if r["instance"] == name and r.get("schedule"): return r
    return next((r for r in rows if r.get("schedule")), None)

# ── JSON data ──────────────────────────────────────────────────────────────
DATA_JSON = json.dumps({
    "machine":  machine,
    "jobshop":  [r for r,_ in js_rows],
    "flowshop": flowshop,
    "js_gap":   [g for _,g in js_rows],
    "gantt": {
        "machine":  pick_gantt(machine,  "inst_book"),
        "jobshop":  pick_gantt(jobshop,  "ft06"),
        "flowshop": pick_gantt(flowshop, "problem_3m_10j"),
    },
    "kpi": {
        "n_total": n_total, "n_opt": n_opt,
        "tot_time": round(tot_time,1),
        "m_opt":m_opt, "js_opt":js_opt, "fs_opt":fs_opt,
        "n_m":n_m, "n_js":n_js, "n_fs":n_fs,
        "m_time":round(m_time,1), "js_time":round(js_time,1), "fs_time":round(fs_time,1),
        "m_avg_gap":round(m_avg_gap,2),
        "fs_avg_gap":round(fs_avg_gap,2),
        "js_avg_gap_opt":round(js_avg_gap_opt,2),
        "transition_n": transition_n or 0,
    },
}, ensure_ascii=False)

TODAY = datetime.date.today().isoformat()
TS    = (cfg.get("timestamp") or "")[:19]
JVER  = cfg.get("julia_version","")

# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

:root{
  --bg:#030810;--bg2:#060d1a;--card:#0a1428;--card2:#0d1a32;
  --bd:#1a2e50;--bd2:#243d6a;
  --fg:#c8dcf4;--mut:#4a6a96;
  --ac:#00c8ff;--ac2:#0090bb;
  --ok:#00ff88;--warn:#ffaa00;--bad:#ff3355;
  --vio:#cc44ff;--pink:#ff44aa;--oran:#ff6600;
  --gold:#ffd700;
  --glow-ac:0 0 20px #00c8ff60;
  --sb:230px;--hh:62px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--bg);color:var(--fg);
  font-family:'Rajdhani',system-ui,sans-serif;
  font-size:15px;line-height:1.6;overflow:hidden;
}
body::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:
    linear-gradient(rgba(0,200,255,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,200,255,.03) 1px,transparent 1px);
  background-size:40px 40px;
}

/* ── TOPBAR ── */
.topbar{
  position:fixed;top:0;left:0;right:0;z-index:300;height:var(--hh);
  background:linear-gradient(90deg,#020b18 0%,#050f20 50%,#020b18 100%);
  border-bottom:1px solid var(--ac);box-shadow:0 0 30px #00c8ff30;
  display:flex;align-items:center;padding:0 22px;gap:14px;
}
.hamburger{
  display:none;width:38px;height:38px;flex:none;cursor:pointer;
  background:#071830;border:1px solid var(--ac);border-radius:6px;
  color:var(--ac);font-size:18px;align-items:center;justify-content:center;
}
.tlogo{
  font-family:'Orbitron',sans-serif;font-size:16px;font-weight:900;
  background:linear-gradient(90deg,var(--ac) 0%,var(--vio) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:1px;white-space:nowrap;filter:drop-shadow(0 0 8px #00c8ff80);
}
.tsub{color:var(--mut);font-size:11px;font-family:'Share Tech Mono',monospace}
.tright{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.chip{
  background:#071830;border:1px solid var(--ac);border-radius:4px;
  padding:2px 10px;font-size:11px;color:var(--ac);
  font-family:'Share Tech Mono',monospace;box-shadow:0 0 8px #00c8ff30;
}
.ts-badge{
  background:#050f20;border:1px solid var(--bd2);border-radius:4px;
  padding:2px 10px;font-size:11px;color:var(--mut);
  font-family:'Share Tech Mono',monospace;
}

/* ── SIDEBAR ── */
.sidebar{
  position:fixed;top:var(--hh);left:0;bottom:0;width:var(--sb);
  background:var(--bg2);border-right:1px solid var(--bd);
  padding:16px 0;overflow-y:auto;z-index:200;
}
.sidebar::-webkit-scrollbar{width:4px}
.sidebar::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:2px}
.sb-sec{
  padding:14px 18px 4px;font-size:9px;letter-spacing:2px;color:var(--mut);
  text-transform:uppercase;font-family:'Orbitron',sans-serif;font-weight:600;
}
.nav-item{
  display:flex;align-items:center;gap:10px;padding:9px 20px;font-size:13px;
  color:var(--mut);cursor:pointer;border-left:3px solid transparent;
  transition:all .2s;font-family:'Rajdhani',sans-serif;font-weight:500;
  user-select:none;
}
.nav-item:hover{background:#0a1e38;color:var(--fg)}
.nav-item.active{
  background:linear-gradient(90deg,#0a2040,transparent);
  color:var(--ac);border-left-color:var(--ac);text-shadow:0 0 10px var(--ac);
}
.nav-icon{font-size:15px;width:18px;text-align:center;flex-shrink:0}
.sb-backdrop{display:none}

/* ── MAIN (one view at a time, NO page scroll) ── */
.main{
  margin-left:var(--sb);margin-top:var(--hh);
  height:calc(100vh - var(--hh));overflow:hidden;
  padding:18px 26px 18px;position:relative;z-index:1;
}
main>section{display:none}
main>section.view-active{display:flex;flex-direction:column;height:100%;animation:fadeIn .35s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* contained scroll only where content is inherently long (reading/cards) */
.vbody{flex:1;min-height:0;overflow-y:auto;padding-right:6px}
.vbody::-webkit-scrollbar,.tab-panel::-webkit-scrollbar,.table-wrap::-webkit-scrollbar{width:7px;height:7px}
.vbody::-webkit-scrollbar-thumb,.tab-panel::-webkit-scrollbar-thumb,.table-wrap::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:4px}

/* ── SECTION HEADER (compact) ── */
.sec-header{margin-bottom:12px;padding-top:0;flex:none}
.sec-title{
  font-family:'Orbitron',sans-serif;font-size:20px;font-weight:700;
  display:flex;align-items:center;gap:12px;margin-bottom:4px;
}
.sec-icon{
  width:38px;height:38px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:18px;flex-shrink:0;border:1px solid var(--bd2);
}
.sec-sub{color:var(--mut);font-size:13px}

/* ── KPI GRID ── */
.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:16px;flex:none}
.kpi-card{
  background:var(--card);border:1px solid var(--bd);border-radius:10px;
  padding:18px 16px;position:relative;overflow:hidden;
  transition:transform .2s,box-shadow .2s;
}
.kpi-card:hover{transform:translateY(-4px);box-shadow:var(--glow-ac)}
.kpi-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--kc,var(--ac));box-shadow:0 0 10px var(--kc,var(--ac));
}
.kpi-glow{
  position:absolute;top:-30px;right:-20px;width:80px;height:80px;border-radius:50%;
  background:var(--kc,var(--ac));opacity:.08;filter:blur(20px);
}
.kpi-val{
  font-family:'Orbitron',sans-serif;font-size:31px;font-weight:900;line-height:1;
  color:var(--kc,var(--ac));text-shadow:0 0 15px var(--kc,var(--ac));
}
.kpi-label{font-size:11px;color:var(--mut);margin-top:6px;letter-spacing:.5px;text-transform:uppercase;font-weight:600}
.kpi-sub{font-size:12px;margin-top:6px;color:var(--kc,var(--fg));opacity:.8}

/* ── CARDS ── */
.card{
  background:var(--card);border:1px solid var(--bd);border-radius:10px;
  padding:20px 22px;margin-bottom:18px;position:relative;
}
.card-title{
  font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;
  color:var(--ac);margin-bottom:14px;letter-spacing:.5px;
}

/* ── STAT ROW ── */
.stat-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.stat-item{
  background:var(--card2);border:1px solid var(--bd);border-radius:8px;
  padding:12px 16px;flex:1;min-width:100px;
}
.stat-val{font-family:'Orbitron',sans-serif;font-size:22px;font-weight:700}
.stat-lbl{font-size:11px;color:var(--mut);margin-top:3px;text-transform:uppercase;letter-spacing:.5px}

/* ── CHART GRID ── */
.chart-grid{display:grid;gap:14px;grid-template-columns:1fr 1fr;margin-bottom:14px}
.chart-grid-3{display:grid;gap:14px;grid-template-columns:repeat(3,1fr);margin-bottom:14px}
.chart-grid-1{display:grid;gap:14px;grid-template-columns:1fr;margin-bottom:14px}
.chart-box{
  background:var(--card2);border:1px solid var(--bd);border-radius:10px;
  padding:14px;position:relative;display:flex;flex-direction:column;min-height:0;
}
.chart-label{
  font-family:'Orbitron',sans-serif;font-size:9px;font-weight:700;color:var(--mut);
  text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;padding-right:60px;flex:none;
}
/* canvas fills its box; box height comes from flex/grid context */
.chart-box canvas{flex:1;min-height:0;width:100%!important;height:auto!important;max-height:none!important}

/* default (inside .vbody): give chart-boxes a sensible height */
.vbody .chart-box{height:300px}
.vbody .chart-box.tall{height:360px}

/* generic fill helper for grids that should occupy remaining height */
.fill{flex:1;min-height:0}
.fill .chart-box{height:auto!important}

/* ── FIT PANELS: charts fill the available viewport, NO scroll ── */
.tab-panel.fit{display:none}
.tab-panel.fit.is-active{display:flex;flex-direction:column;height:100%;min-height:0;overflow:hidden}
.tab-panel.fit .chart-grid,
.tab-panel.fit .chart-grid-3,
.tab-panel.fit .chart-grid-1{flex:1;min-height:0;margin-bottom:14px}
.tab-panel.fit > .chart-grid:last-child,
.tab-panel.fit > .chart-grid-3:last-child,
.tab-panel.fit > .chart-grid-1:last-child{margin-bottom:0}
.tab-panel.fit > .chart-box{flex:1;min-height:0;margin-bottom:14px}
.tab-panel.fit > .chart-box:last-child{margin-bottom:0}
.reset-btn{
  position:absolute;top:10px;right:10px;z-index:10;
  background:#050f20;border:1px solid var(--bd2);color:var(--mut);
  border-radius:4px;padding:2px 8px;font-size:10px;cursor:pointer;
  font-family:'Share Tech Mono',monospace;transition:all .15s;
}
.reset-btn:hover{border-color:var(--ac);color:var(--ac);box-shadow:0 0 6px #00c8ff40}

/* ── FORMULA ── */
.formula{
  background:#020810;border:1px solid var(--bd);border-left:3px solid var(--ac);
  box-shadow:inset 0 0 20px #00c8ff08;border-radius:8px;padding:14px 18px;
  font-family:'Share Tech Mono',monospace;font-size:12.5px;color:#80aacc;
  overflow-x:auto;white-space:pre;margin-bottom:18px;line-height:2;
}

/* ── TABS (inner) ── */
.tab-bar{display:flex;gap:3px;border-bottom:1px solid var(--bd);flex-wrap:wrap;flex:none}
.tab-btn{
  padding:7px 16px;font-size:13px;cursor:pointer;border:1px solid transparent;
  border-bottom:none;background:transparent;color:var(--mut);
  border-radius:6px 6px 0 0;transition:all .15s;
  font-family:'Rajdhani',sans-serif;font-weight:600;letter-spacing:.3px;
}
.tab-btn:hover{background:#0a1e38;color:var(--fg)}
.tab-btn.is-active{
  background:var(--card2);color:var(--ac);border-color:var(--bd);
  border-bottom-color:var(--card2);text-shadow:0 0 8px var(--ac);
}
/* tab-body = flexible region that fills the rest of the view */
.tab-body{flex:1;min-height:0;display:flex;padding-top:14px}
.tab-host{flex:1;min-height:0;position:relative;width:100%;display:flex;flex-direction:column}
.tab-panel{display:none}
.tab-panel.is-active{display:block;height:100%;overflow-y:auto;animation:fadeIn .25s ease}
/* when a panel only holds a table, let it fill and scroll internally */
.tab-panel .table-wrap{max-height:none}
.tab-panel.is-active .filter-bar{position:sticky;top:0;background:var(--bg);z-index:5;padding-bottom:6px}

/* ── ANALYSIS BOX ── */
.analysis-box{
  background:#020d1e;border:1px solid var(--bd);border-left:3px solid var(--vio);
  border-radius:8px;padding:16px 20px;margin-bottom:18px;
}
.analysis-box p{font-size:13.5px;line-height:1.8;color:#94b8d8;margin-bottom:10px}
.analysis-box p:last-child{margin-bottom:0}
.analysis-box strong{color:var(--fg)}
.analysis-box em{color:var(--ac)}

/* ── TABLE ── */
.filter-bar{display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap}
.filter-bar input,.filter-bar select{
  background:#040c1c;border:1px solid var(--bd2);color:var(--fg);border-radius:6px;
  padding:6px 12px;font-size:12.5px;outline:none;font-family:'Share Tech Mono',monospace;
}
.filter-bar input{width:190px}
.filter-bar input:focus,.filter-bar select:focus{border-color:var(--ac);box-shadow:0 0 8px #00c8ff30}
.csv-btn{
  background:#04140a;border:1px solid #00502a;color:var(--ok);border-radius:6px;
  padding:6px 12px;font-size:11px;cursor:pointer;font-family:'Share Tech Mono',monospace;
  transition:all .15s;
}
.csv-btn:hover{box-shadow:0 0 8px #00ff8840;border-color:var(--ok)}
.row-count{color:var(--mut);font-size:11px;margin-left:auto;font-family:'Share Tech Mono',monospace}
.table-wrap{overflow-x:auto;border:1px solid var(--bd);border-radius:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
thead th{
  padding:9px 12px;border-bottom:1px solid var(--bd2);color:var(--ac);font-size:10px;
  font-weight:700;text-transform:uppercase;letter-spacing:.8px;text-align:left;
  white-space:nowrap;cursor:pointer;user-select:none;position:sticky;top:0;
  background:var(--card);font-family:'Orbitron',sans-serif;z-index:1;
}
thead th:hover{color:var(--fg)}
.sort-ic{opacity:.25;margin-left:4px;font-size:9px}
th.s-asc .sort-ic,th.s-desc .sort-ic{opacity:1;color:var(--gold)}
tbody td{padding:8px 12px;border-bottom:1px solid #0d1e36;vertical-align:middle}
tbody tr:hover td{background:#0a1e38}
tbody tr:last-child td{border-bottom:none}
.num{text-align:right;font-family:'Share Tech Mono',monospace;font-variant-numeric:tabular-nums}
.bold-num{font-weight:700}
.c-ok{color:var(--ok)} .c-warn{color:var(--warn)} .c-bad{color:var(--bad)}
.inst-code{
  font-family:'Share Tech Mono',monospace;font-size:11.5px;color:var(--ac);
  background:#061020;padding:2px 7px;border:1px solid var(--bd);border-radius:4px;
}
.job-cell{font-family:'Orbitron',sans-serif;font-weight:700;color:var(--gold)}
.badge{font-size:10px;padding:3px 10px;border-radius:4px;font-weight:700;
  white-space:nowrap;font-family:'Share Tech Mono',monospace;letter-spacing:.5px}
.badge.ok{background:#001f10;color:var(--ok);border:1px solid #00502a}
.badge.warn{background:#1f1000;color:var(--warn);border:1px solid #503000}
.badge.bad{background:#1f0008;color:var(--bad);border:1px solid #500018}

/* ── GANTT ── */
.gantt-box{background:var(--card2);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:16px;position:relative}
.gantt-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex:none}
.gantt-title{font-family:'Orbitron',sans-serif;font-size:12px;font-weight:700;color:var(--ac)}
.gantt-meta{font-size:12px;color:var(--ok);font-family:'Share Tech Mono',monospace}
.zoom-hint{font-size:10px;color:var(--mut);text-align:right;margin-top:6px;font-family:'Share Tech Mono',monospace;flex:none}
/* gantt fills a fit panel */
.tab-panel.fit > .gantt-box{flex:1;min-height:0;display:flex;flex-direction:column;margin-bottom:0}
.tab-panel.fit > .gantt-box canvas{flex:1;min-height:0;height:auto!important;max-height:none!important}

/* ── PROGRESS ── */
.progress-wrap{display:flex;flex-direction:column;gap:16px}
.progress-header{display:flex;justify-content:space-between;margin-bottom:6px}
.progress-name{font-family:'Orbitron',sans-serif;font-size:12px;font-weight:700}
.progress-info{font-size:12px;color:var(--mut);font-family:'Share Tech Mono',monospace}
.progress-bar{height:10px;background:#0a1428;border-radius:5px;overflow:hidden;border:1px solid var(--bd)}
.progress-fill{height:100%;border-radius:5px;transition:width 1.2s cubic-bezier(.4,0,.2,1)}

/* ── REFERENCES / CONCLUSION ── */
.ref-list{list-style:none;display:flex;flex-direction:column;gap:8px}
.ref-list li{font-size:13px;color:#6a90b8;padding:10px 14px;background:#040c1c;
  border:1px solid var(--bd);border-left:3px solid var(--vio);border-radius:6px;line-height:1.6}
.ref-list li strong{color:var(--ac);font-family:'Share Tech Mono',monospace}
.conclusion-card h3{font-family:'Orbitron',sans-serif;font-size:14px;font-weight:700;color:var(--ac);margin:20px 0 8px}
.conclusion-card h3:first-child{margin-top:0}
.conclusion-card p{font-size:13.5px;line-height:1.8;color:#94b8d8;margin-bottom:8px}

/* ── FAB ── */
.fab{
  position:fixed;bottom:24px;right:24px;z-index:400;
  background:linear-gradient(135deg,var(--ac2),var(--vio));color:#fff;border:none;
  border-radius:50%;width:52px;height:52px;font-size:20px;cursor:pointer;
  box-shadow:0 0 20px #00c8ff60;transition:transform .2s;
}
.fab:hover{transform:scale(1.1)}

/* ── RESPONSIVE ── */
@media(max-width:960px){
  .hamburger{display:flex}
  .sidebar{transform:translateX(-100%);transition:transform .25s;z-index:350}
  .sidebar.open{transform:none}
  .sb-backdrop.show{display:block;position:fixed;inset:var(--hh) 0 0 0;background:#0006;z-index:340}
  .main{margin-left:0;padding:16px}
  .kpi-grid{grid-template-columns:repeat(2,1fr)}
  .chart-grid,.chart-grid-3{grid-template-columns:1fr}
  .tsub{display:none}
}
@media print{
  .sidebar,.topbar,.fab,.hamburger{display:none!important}
  .main{margin:0;height:auto;overflow:visible;padding:10px}
  main>section{display:block!important;page-break-after:always}
}
"""

# ══════════════════════════════════════════════════════════════════════════
# JS
# ══════════════════════════════════════════════════════════════════════════
JS = r"""
const D = __DATA__;
const K = D.kpi;
const COL = {
  grid:'#1a2e5030',tick:'#4a6a96',fg:'#c8dcf4',
  ac:'#00c8ff',ok:'#00ff88',warn:'#ffaa00',bad:'#ff3355',
  vio:'#cc44ff',pink:'#ff44aa',oran:'#ff6600',gold:'#ffd700',
};
const PAL = [
  '#00c8ff','#00ff88','#cc44ff','#ffaa00','#ff44aa',
  '#ff3355','#ff6600','#44ffcc','#aa88ff','#ffcc00',
  '#0088ff','#88ff00','#ff0088','#00ffcc','#ff8800',
  '#4488ff','#ff4466','#88ffaa','#ddaa00','#cc00ff',
];

Chart.defaults.color = COL.tick;
Chart.defaults.borderColor = COL.grid;
Chart.defaults.font.family = "'Rajdhani', sans-serif";
Chart.defaults.font.size = 11;
Chart.register(ChartZoom);

// ── chart registry (for resetZoom + resize) ───────────────────────────────
const CHART_REG = {};
function _reg(id, ch){ if(ch) CHART_REG[id]=ch; return ch; }
function resetZoom(btn){
  const cv = btn.closest('.chart-box, .gantt-box').querySelector('canvas');
  if(cv && CHART_REG[cv.id] && CHART_REG[cv.id].resetZoom) CHART_REG[cv.id].resetZoom();
}
function resizeVisibleCharts(){
  requestAnimationFrame(() => {
    Object.values(CHART_REG).forEach(ch => {
      try{ if(ch.canvas && ch.canvas.offsetParent !== null) ch.resize(); }catch(e){}
    });
  });
}

// ── base options ──────────────────────────────────────────────────────────
function baseOpts(xLbl, yLbl){
  return {
    responsive:true, maintainAspectRatio:false,
    animation:{ duration:600, easing:'easeOutQuart' },
    plugins:{
      legend:{ display:false },
      tooltip:{
        backgroundColor:'#050f20', borderColor:'#1a2e50', borderWidth:1,
        titleColor:'#c8dcf4', bodyColor:'#4a6a96', padding:10, cornerRadius:6,
        titleFont:{ family:'Share Tech Mono', size:12 },
        bodyFont:{ family:'Share Tech Mono', size:11 },
        callbacks:{
          title: items => items.length ? [items[0].label] : [],
          label: item => (item.dataset.label ? item.dataset.label+': ' : '') + item.formattedValue,
        },
      },
      zoom:{
        zoom:{ wheel:{enabled:true}, pinch:{enabled:true}, mode:'xy' },
        pan:{ enabled:true, mode:'xy' },
      },
    },
    scales:{
      x:{ grid:{color:COL.grid}, ticks:{color:COL.tick,maxRotation:42,font:{family:'Share Tech Mono',size:10}},
          title: xLbl ? {display:true,text:xLbl,color:COL.tick,font:{size:11}} : undefined },
      y:{ grid:{color:COL.grid}, beginAtZero:true, ticks:{color:COL.tick,font:{family:'Share Tech Mono',size:10}},
          title: yLbl ? {display:true,text:yLbl,color:COL.tick,font:{size:11}} : undefined },
    },
  };
}

// ── chart factories ───────────────────────────────────────────────────────
function barChart(id, labels, data, colorFn, xLbl, yLbl){
  const el = document.getElementById(id); if(!el) return;
  const colors = data.map((v,i) => typeof colorFn==='function' ? colorFn(v,i) : colorFn);
  return _reg(id, new Chart(el, {
    type:'bar',
    data:{ labels, datasets:[{ data, backgroundColor:colors, borderRadius:4, borderSkipped:false }] },
    options: baseOpts(xLbl, yLbl),
  }));
}
function multiBarChart(id, labels, datasets, xLbl, yLbl){
  const el = document.getElementById(id); if(!el) return;
  const opts = baseOpts(xLbl, yLbl);
  opts.plugins.legend = { display:true, labels:{color:COL.tick,font:{family:'Rajdhani',size:11},padding:12} };
  return _reg(id, new Chart(el, { type:'bar', data:{labels,datasets}, options:opts }));
}
function lineChart(id, labels, datasets, xLbl, yLbl){
  const el = document.getElementById(id); if(!el) return;
  return _reg(id, new Chart(el, { type:'line', data:{labels,datasets}, options:baseOpts(xLbl,yLbl) }));
}

// quadratic least-squares fit  y = a x^2 + b x + c
function quadFit(pts){
  let S0=pts.length,S1=0,S2=0,S3=0,S4=0,Sy=0,Sxy=0,Sx2y=0;
  for(const p of pts){ const x=p.x,y=p.y,x2=x*x;
    S1+=x; S2+=x2; S3+=x2*x; S4+=x2*x2; Sy+=y; Sxy+=x*y; Sx2y+=x2*y; }
  const M=[[S0,S1,S2,Sy],[S1,S2,S3,Sxy],[S2,S3,S4,Sx2y]];
  for(let i=0;i<3;i++){
    const piv=M[i][i]||1e-9;
    for(let j=i;j<4;j++) M[i][j]/=piv;
    for(let k=0;k<3;k++) if(k!==i){ const f=M[k][i]; for(let j=i;j<4;j++) M[k][j]-=f*M[i][j]; }
  }
  const c=M[0][3], b=M[1][3], a=M[2][3];
  return x => a*x*x + b*x + c;
}

function scatterChart(id, data, colorFn, xLbl, yLbl, lblFn, trend){
  const el = document.getElementById(id); if(!el) return;
  const ptColors = data.map((p,i) => typeof colorFn==='function' ? colorFn(p,i) : colorFn);
  const opts = baseOpts(xLbl, yLbl);
  opts.plugins.tooltip.callbacks.title = items =>
    items.map(it => it.datasetIndex===0 ? (lblFn ? (lblFn(data[it.dataIndex], it.dataIndex)||'') : '') : 'tendência');
  opts.plugins.tooltip.callbacks.label = it =>
    it.datasetIndex===0 ? ('x='+it.raw.x+' · y='+(+it.raw.y).toFixed(2)) : '';
  const datasets = [{
    type:'scatter', data, backgroundColor:ptColors,
    pointRadius:7, pointHoverRadius:10, borderWidth:0, order:2,
  }];
  if(trend && data.length>2){
    const fit = quadFit(data);
    const xs = [...new Set(data.map(p=>p.x))].sort((a,b)=>a-b);
    const line = xs.map(x => ({ x, y: Math.max(0, fit(x)) }));
    datasets.push({
      type:'line', data:line, label:'tendência (ajuste quadrático)',
      borderColor:COL.gold, borderWidth:2, borderDash:[6,4],
      pointRadius:0, fill:false, tension:.3, order:1,
    });
    opts.plugins.legend = { display:true, labels:{color:COL.tick,font:{family:'Rajdhani',size:11}} };
  }
  return _reg(id, new Chart(el, { data:{datasets}, options:opts }));
}

function donutChart(id, data, labels, colors){
  const el = document.getElementById(id); if(!el) return;
  return _reg(id, new Chart(el, {
    type:'doughnut',
    data:{ labels, datasets:[{ data, backgroundColor:colors, borderWidth:2, borderColor:'#030810', hoverOffset:6 }] },
    options:{
      responsive:true, maintainAspectRatio:false, cutout:'62%', animation:{duration:800,animateRotate:true},
      plugins:{
        legend:{ display:true, position:'bottom', labels:{color:COL.tick,padding:14,font:{family:'Rajdhani',size:12}} },
        tooltip:{ backgroundColor:'#050f20', borderColor:'#1a2e50', borderWidth:1 },
        zoom:{ zoom:{wheel:{enabled:false}} },
      },
    },
  }));
}

function ganttChart(id, rows, rowFn, labelFn){
  const el = document.getElementById(id); if(!el || !rows) return;
  const cats = [...new Set(rows.map(rowFn))];
  const ds = rows.map((s,i) => ({
    label: labelFn(s),
    data: [{ x:[s.start, s.finish], y:String(rowFn(s)) }],
    backgroundColor: PAL[(s.ci ?? i) % PAL.length] + 'cc',
    borderColor: PAL[(s.ci ?? i) % PAL.length],
    borderWidth:1, borderRadius:3,
  }));
  return _reg(id, new Chart(el, {
    type:'bar',
    data:{ labels:cats.map(String), datasets:ds },
    options:{
      indexAxis:'y', responsive:true, maintainAspectRatio:false, animation:{duration:500},
      plugins:{
        legend:{ display:false },
        tooltip:{
          backgroundColor:'#050f20', borderColor:'#1a2e50', borderWidth:1,
          callbacks:{ label: c => c.dataset.label + ': [' + c.raw.x[0] + ' → ' + c.raw.x[1] + ']' },
          titleFont:{family:'Share Tech Mono'}, bodyFont:{family:'Share Tech Mono'},
        },
        zoom:{ zoom:{wheel:{enabled:true},pinch:{enabled:true},mode:'x'}, pan:{enabled:true,mode:'x'} },
      },
      scales:{
        x:{ stacked:false, grid:{color:COL.grid}, ticks:{color:COL.tick,font:{family:'Share Tech Mono',size:10}}, title:{display:true,text:'tempo',color:COL.tick} },
        y:{ stacked:true, grid:{color:COL.grid}, ticks:{color:COL.tick,font:{family:'Share Tech Mono',size:10}} },
      },
    },
  }));
}

// ── count-up ──────────────────────────────────────────────────────────────
function countUp(id, target, suffix, dec){
  const el = document.getElementById(id); if(!el) return;
  const steps = 45; let i = 0;
  const tick = () => {
    i++; const v = target * (i/steps);
    el.textContent = (dec ? v.toFixed(dec) : Math.round(v)) + (suffix||'');
    if(i < steps) requestAnimationFrame(tick);
    else el.textContent = (dec ? target.toFixed(dec) : target) + (suffix||'');
  };
  requestAnimationFrame(tick);
}

// helper to render a stat-row
function statRow(elId, items){
  const el = document.getElementById(elId); if(!el) return;
  el.innerHTML = items.map(x =>
    `<div class="stat-item"><div class="stat-val" style="color:${x.c}">${x.v}</div><div class="stat-lbl">${x.l}</div></div>`
  ).join('');
}

// ── derived arrays ────────────────────────────────────────────────────────
const mL    = D.machine.map(r => r.instance.replace('inst_',''));
const mObj  = D.machine.map(r => r.objective || 0);
const mRT   = D.machine.map(r => r.runtime   || 0);
const mGap  = D.machine.map(r => +((r.gap||0)*100).toFixed(3));
const mBnd  = D.machine.map(r => r.bound || 0);
const mN    = D.machine.map(r => r.n);
const mStat = D.machine.map(r => (r.status||'').includes('OPTIMAL'));

const jsL   = D.jobshop.map(r => r.instance);
const jsObj = D.jobshop.map(r => r.objective || 0);
const jsOpt = D.jobshop.map(r => r.optimum   || 0);
const jsBnd = D.jobshop.map(r => r.bound     || 0);
const jsRT  = D.jobshop.map(r => r.runtime   || 0);
const jsGap = D.js_gap;

const fsL   = D.flowshop.map(r => r.instance.replace('problem_',''));
const fsObj = D.flowshop.map(r => r.objective || 0);
const fsBnd = D.flowshop.map(r => r.bound     || 0);
const fsRT  = D.flowshop.map(r => r.runtime   || 0);
const fsGap = D.flowshop.map(r => +((r.gap||0)*100).toFixed(3));

// ── per-view initializers (lazy) ──────────────────────────────────────────
const VIEW_INIT = {}, VIEW_DONE = {};

VIEW_INIT.overview = () => {
  countUp('kv-total', K.n_total);
  countUp('kv-opt',   K.n_opt);
  countUp('kv-rate',  +(100*K.n_opt/K.n_total).toFixed(1), '%', 1);
  countUp('kv-time',  K.tot_time, 's', 1);
  countUp('kv-classes', 3);
  donutChart('ch-status', [K.n_opt, K.n_total-K.n_opt], ['Ótimo Provado','Time Limit'], [COL.ok, COL.warn]);
  barChart('ch-class-time', ['Machine','Job Shop','Flow Shop'], [K.m_time,K.js_time,K.fs_time],
    (v,i)=>[COL.ac,COL.vio,COL.ok][i]+'cc', '', 'Tempo (s)');
  barChart('ch-opt-rate', ['Machine','Job Shop','Flow Shop'],
    [+(100*K.m_opt/K.n_m).toFixed(1), +(100*K.js_opt/K.n_js).toFixed(1), +(100*K.fs_opt/K.n_fs).toFixed(1)],
    v => v>=50?COL.ok+'cc':v>=20?COL.warn+'cc':COL.bad+'cc', '', 'Taxa (%)');
};

VIEW_INIT.instances = () => {
  donutChart('ch-inst-dist', [K.n_m, K.n_js, K.n_fs],
    ['Machine ('+K.n_m+')','Job Shop ('+K.n_js+')','Flow Shop ('+K.n_fs+')'],
    [COL.ac, COL.vio, COL.ok]);
  updateRowCount('mp-tbody','mp-rc');
};

VIEW_INIT.machine = () => {
  barChart('ch-m-obj', mL, mObj, ()=>COL.ac+'cc', '', 'ΣT_j');
  barChart('ch-m-time', mL, mRT, ()=>COL.warn+'cc', '', 'Tempo (s)');
  barChart('ch-m-gap', mL, mGap, v=>v>50?COL.bad+'cc':v>0?COL.warn+'cc':COL.ok+'cc', '', 'Gap (%)');
  multiBarChart('ch-m-bound', mL, [
    { label:'Objetivo (ΣT_j)', data:mObj, backgroundColor:COL.ac+'99', borderColor:COL.ac, borderWidth:1, borderRadius:4 },
    { label:'Bound LP', data:mBnd, backgroundColor:COL.ok+'66', borderColor:COL.ok, borderWidth:1, borderRadius:4 },
  ], '', 'Valor');
  // scatter ΣT_j × n WITH quadratic trend line (melhoria #2)
  scatterChart('ch-m-scatter',
    D.machine.map(r => ({ x:r.n, y:r.objective||0 })),
    (p,i)=>mStat[i]?COL.ok+'cc':COL.warn+'cc',
    'n (jobs)', 'ΣT_j', (p,i)=>mL[i]+' (n='+D.machine[i].n+')', true);
  lineChart('ch-m-scale-time', mN, [{
    label:'Tempo (s)', data:mRT, borderColor:COL.ac, backgroundColor:COL.ac+'20',
    pointBackgroundColor:mStat.map(s=>s?COL.ok:COL.warn), pointRadius:6, fill:true, tension:.35,
  }], 'n (jobs)', 'Tempo (s)');
  lineChart('ch-m-scale-gap', mN, [{
    label:'Gap (%)', data:mGap, borderColor:COL.bad, backgroundColor:COL.bad+'20',
    pointBackgroundColor:mGap.map(v=>v>50?COL.bad:v>0?COL.warn:COL.ok), pointRadius:6, fill:true, tension:.35,
  }], 'n (jobs)', 'Gap MIP (%)');
  statRow('m-stats', [
    { v:K.n_m, c:COL.ac, l:'Instâncias' },
    { v:K.m_opt, c:COL.ok, l:'Ótimas' },
    { v:K.n_m-K.m_opt, c:COL.warn, l:'Time Limit' },
    { v:K.m_avg_gap.toFixed(1)+'%', c:COL.bad, l:'Gap Médio' },
    { v:K.m_time.toFixed(1)+'s', c:COL.vio, l:'Tempo Total' },
  ]);
  updateRowCount('m-tbody','m-rc');
};

VIEW_INIT.jobshop = () => {
  multiBarChart('ch-js-obj', jsL, [
    { label:'Cmax Obtido', data:jsObj, backgroundColor:COL.vio+'99', borderColor:COL.vio, borderWidth:1, borderRadius:4 },
    { label:'Ótimo (JSPLIB)', data:jsOpt, backgroundColor:COL.ok+'99', borderColor:COL.ok, borderWidth:1, borderRadius:4 },
  ], '', 'Cmax');
  barChart('ch-js-time', jsL, jsRT, ()=>COL.warn+'cc', '', 'Tempo (s)');
  barChart('ch-js-gap', jsL, jsGap.map(v=>v!=null?+v.toFixed(2):0),
    v=>v===0?COL.ok+'cc':v<20?COL.warn+'cc':COL.bad+'cc', '', '% acima do ótimo');
  multiBarChart('ch-js-bound', jsL, [
    { label:'Objetivo', data:jsObj, backgroundColor:COL.vio+'99', borderColor:COL.vio, borderWidth:1, borderRadius:4 },
    { label:'Bound LP', data:jsBnd, backgroundColor:COL.ac+'66', borderColor:COL.ac, borderWidth:1, borderRadius:4 },
  ], '', 'Valor');
  const vg = jsGap.filter(v=>v!=null);
  statRow('js-stats', [
    { v:K.n_js, c:COL.vio, l:'Instâncias' },
    { v:K.js_opt, c:COL.ok, l:'Ótimas' },
    { v:K.n_js-K.js_opt, c:COL.warn, l:'Time Limit' },
    { v:(vg.reduce((a,b)=>a+b,0)/vg.length||0).toFixed(1)+'%', c:COL.bad, l:'Gap vs Ótimo' },
    { v:K.js_time.toFixed(1)+'s', c:COL.ac, l:'Tempo Total' },
  ]);
  updateRowCount('js-tbody','js-rc');
};

VIEW_INIT.flowshop = () => {
  barChart('ch-fs-obj', fsL, fsObj, ()=>COL.ok+'cc', '', 'Cmax');
  barChart('ch-fs-time', fsL, fsRT, ()=>COL.warn+'cc', '', 'Tempo (s)');
  barChart('ch-fs-gap', fsL, fsGap, v=>v>30?COL.bad+'cc':v>5?COL.warn+'cc':COL.ok+'cc', '', 'Gap (%)');
  multiBarChart('ch-fs-bound', fsL, [
    { label:'Objetivo', data:fsObj, backgroundColor:COL.ok+'99', borderColor:COL.ok, borderWidth:1, borderRadius:4 },
    { label:'Bound LP', data:fsBnd, backgroundColor:COL.ac+'66', borderColor:COL.ac, borderWidth:1, borderRadius:4 },
  ], '', 'Valor');
  statRow('fs-stats', [
    { v:K.n_fs, c:COL.ok, l:'Instâncias' },
    { v:K.fs_opt, c:COL.ok, l:'Ótimas' },
    { v:K.n_fs-K.fs_opt, c:COL.warn, l:'Time Limit' },
    { v:K.fs_avg_gap.toFixed(1)+'%', c:COL.bad, l:'Gap Médio' },
    { v:K.fs_time.toFixed(1)+'s', c:COL.pink, l:'Tempo Total' },
  ]);
  updateRowCount('fs-tbody','fs-rc');
};

VIEW_INIT.scalability = () => {
  scatterChart('ch-sc-m-time',
    D.machine.map(r => ({ x:r.n, y:r.runtime||0 })),
    (p,i)=>mStat[i]?COL.ok+'cc':COL.warn+'cc',
    'n (jobs)', 'Tempo (s)', (p,i)=>mL[i]+' (n='+D.machine[i].n+')');
  scatterChart('ch-js-scale',
    D.jobshop.map(r => ({ x:r.jobs*r.machines, y:r.runtime||0 })),
    (p,i)=>(jsGap[i]===0?COL.ok:COL.vio)+'cc',
    'Operações (J×M)', 'Tempo (s)', (p,i)=>D.jobshop[i].instance+' '+D.jobshop[i].jobs+'×'+D.jobshop[i].machines);
  scatterChart('ch-fs-scale',
    D.flowshop.map(r => ({ x:r.jobs*r.machines, y:+((r.gap||0)*100).toFixed(2) })),
    p=>p.y<5?COL.ok+'cc':p.y<30?COL.warn+'cc':COL.bad+'cc',
    'Operações (J×M)', 'Gap MIP (%)', (p,i)=>fsL[i]+' '+D.flowshop[i].jobs+'×'+D.flowshop[i].machines);
  // all instances
  const allL = [...mL, ...jsL, ...fsL];
  const allT = [...mRT, ...jsRT, ...fsRT];
  const allC = [...D.machine.map(()=>COL.ac+'cc'), ...D.jobshop.map(()=>COL.vio+'cc'), ...D.flowshop.map(()=>COL.ok+'cc')];
  const el = document.getElementById('ch-all-time');
  if(el){
    const o = baseOpts('Instância', 'Tempo (s)');
    o.plugins.legend = { display:true, labels:{ generateLabels:()=>[
      { text:'Machine', fillStyle:COL.ac+'cc', strokeStyle:COL.ac, lineWidth:1 },
      { text:'Job Shop', fillStyle:COL.vio+'cc', strokeStyle:COL.vio, lineWidth:1 },
      { text:'Flow Shop', fillStyle:COL.ok+'cc', strokeStyle:COL.ok, lineWidth:1 },
    ], color:COL.tick, font:{family:'Rajdhani',size:11} } };
    _reg('ch-all-time', new Chart(el, { type:'bar',
      data:{ labels:allL, datasets:[{ data:allT, backgroundColor:allC, borderRadius:3 }] }, options:o }));
  }
};

VIEW_INIT.gantt = () => {
  const G = D.gantt;
  if(G.machine && G.machine.schedule){
    const rows = G.machine.schedule.map(o => ({ ...o, ci:(o.tardiness||0)>0?5:0 }));
    ganttChart('gantt-machine', rows, o=>o.job,
      o => o.job+' ['+o.start+'→'+o.finish+']'+((o.tardiness||0)>0?' ⚠ T='+o.tardiness:''));
    document.getElementById('gantt-machine-meta').textContent = 'ΣT_j = '+G.machine.objective+' · n = '+G.machine.n;
  }
  if(G.jobshop && G.jobshop.schedule){
    const rows = G.jobshop.schedule.map(o => ({ ...o, ci:o.job-1 }));
    ganttChart('gantt-js', rows, o=>'M'+o.machine, o=>'J'+o.job+' op'+o.op);
    document.getElementById('gantt-js-meta').textContent = 'Cmax = '+G.jobshop.objective+' · '+G.jobshop.jobs+'×'+G.jobshop.machines;
  }
  if(G.flowshop && G.flowshop.schedule){
    const jobs = [...new Set(G.flowshop.schedule.map(o=>o.job))];
    const rows = G.flowshop.schedule.map(o => ({ ...o, ci:jobs.indexOf(o.job) }));
    ganttChart('gantt-fs', rows, o=>'M'+o.machine, o=>o.job);
    document.getElementById('gantt-fs-meta').textContent = 'Cmax = '+G.flowshop.objective+' · '+G.flowshop.jobs+'×'+G.flowshop.machines;
  }
};

VIEW_INIT.summary = () => {
  donutChart('ch-sum-inst', [K.n_m,K.n_js,K.n_fs],
    ['Machine ('+K.n_m+')','Job Shop ('+K.n_js+')','Flow Shop ('+K.n_fs+')'], [COL.ac,COL.vio,COL.ok]);
  barChart('ch-sum-avg-time', ['Machine','Job Shop','Flow Shop'],
    [+(K.m_time/K.n_m).toFixed(2), +(K.js_time/K.n_js).toFixed(2), +(K.fs_time/K.n_fs).toFixed(2)],
    (v,i)=>[COL.ac,COL.vio,COL.ok][i]+'cc', '', 'Tempo médio (s)');
  barChart('ch-sum-opt', ['Machine','Job Shop','Flow Shop'],
    [+(100*K.m_opt/K.n_m).toFixed(1), +(100*K.js_opt/K.n_js).toFixed(1), +(100*K.fs_opt/K.n_fs).toFixed(1)],
    v=>v>=50?COL.ok+'cc':v>=20?COL.warn+'cc':COL.bad+'cc', '', '%');
  const pb = document.getElementById('progress-bars');
  if(pb && !pb.dataset.done){
    pb.dataset.done = '1';
    [
      { n:'Machine Scheduling', opt:K.m_opt, tot:K.n_m, c:COL.ac, t:K.m_time },
      { n:'Job Shop', opt:K.js_opt, tot:K.n_js, c:COL.vio, t:K.js_time },
      { n:'Flow Shop', opt:K.fs_opt, tot:K.n_fs, c:COL.ok, t:K.fs_time },
    ].forEach(x => {
      const p = (100*x.opt/x.tot).toFixed(1);
      pb.innerHTML += `<div class="progress-row">
        <div class="progress-header">
          <span class="progress-name" style="color:${x.c}">${x.n}</span>
          <span class="progress-info">${x.opt}/${x.tot} ótimos · ${p}% · ${x.t.toFixed(1)}s</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:${p}%;background:linear-gradient(90deg,${x.c},${x.c}88);box-shadow:0 0 8px ${x.c}80"></div></div>
      </div>`;
    });
  }
};

// ── VIEW SWITCHER (no page scroll; nav swaps the view) ─────────────────────
function showView(id, navEl){
  document.querySelectorAll('main > section').forEach(s => s.classList.remove('view-active'));
  const sec = document.getElementById(id);
  if(sec) sec.classList.add('view-active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if(navEl) navEl.classList.add('active');
  else { const a = document.querySelector('.nav-item[data-view="'+id+'"]'); if(a) a.classList.add('active'); }
  if(!VIEW_DONE[id] && VIEW_INIT[id]){ try{ VIEW_INIT[id](); }catch(e){ console.error(e); } VIEW_DONE[id]=true; }
  resizeVisibleCharts();
  const main = document.querySelector('.main'); if(main) main.scrollTop = 0;
  closeSidebar();
}

// ── INNER TABS ────────────────────────────────────────────────────────────
function showTab(sectionId, panelId, btn){
  const section = document.getElementById(sectionId); if(!section) return;
  section.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('is-active'));
  section.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('is-active'));
  const panel = document.getElementById(panelId);
  if(panel) panel.classList.add('is-active');
  if(btn) btn.classList.add('is-active');
  resizeVisibleCharts();   // melhoria #5: fix chart sizing when panel becomes visible
}

// ── TABLE: filter / sort / count / CSV ────────────────────────────────────
function updateRowCount(tbodyId, rcId){
  const tbody = document.getElementById(tbodyId), rc = document.getElementById(rcId);
  if(!tbody || !rc) return;
  const n = [...tbody.querySelectorAll('tr')].filter(r => r.style.display !== 'none').length;
  rc.textContent = n + ' registros';
}
function filterTable(tbodyId, searchId, statusId, rcId){
  const search = (document.getElementById(searchId)?.value || '').toLowerCase();
  const status = (document.getElementById(statusId)?.value || '').toUpperCase();
  const tbody = document.getElementById(tbodyId); if(!tbody) return;
  [...tbody.querySelectorAll('tr')].forEach(tr => {
    const okT = !search || tr.textContent.toLowerCase().includes(search);
    const okS = !status || (tr.dataset.s || '').includes(status);
    tr.style.display = okT && okS ? '' : 'none';
  });
  updateRowCount(tbodyId, rcId);
}
function sortTable(tblId, col){
  const tbl = document.getElementById(tblId); if(!tbl) return;
  const ths = tbl.querySelectorAll('thead th');
  const tbody = tbl.querySelector('tbody');
  const rows = [...tbody.querySelectorAll('tr')];
  const th = ths[col];
  const asc = !th.classList.contains('s-asc');
  ths.forEach(t => { t.classList.remove('s-asc','s-desc'); const ic=t.querySelector('.sort-ic'); if(ic) ic.textContent='⇅'; });
  th.classList.add(asc ? 's-asc' : 's-desc');
  const ic = th.querySelector('.sort-ic'); if(ic) ic.textContent = asc ? '↑' : '↓';
  rows.sort((a,b) => {
    const av = (a.cells[col]?.textContent||'').trim().replace(/[%s]/g,'');
    const bv = (b.cells[col]?.textContent||'').trim().replace(/[%s]/g,'');
    const an = parseFloat(av), bn = parseFloat(bv);
    if(!isNaN(an) && !isNaN(bn)) return asc ? an-bn : bn-an;
    return asc ? av.localeCompare(bv,'pt') : bv.localeCompare(av,'pt');
  });
  rows.forEach(r => tbody.appendChild(r));
}
function exportCSV(tblId, filename){
  const tbl = document.getElementById(tblId); if(!tbl) return;
  const rows = [...tbl.querySelectorAll('tr')].filter(r => r.style.display !== 'none');
  const csv = rows.map(r => [...r.querySelectorAll('th,td')]
    .map(c => '"' + c.textContent.trim().replace(/"/g,'""') + '"').join(',')).join('\n');
  const blob = new Blob(['﻿'+csv], { type:'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
}

// ── MOBILE SIDEBAR ────────────────────────────────────────────────────────
function toggleSidebar(){
  document.querySelector('.sidebar').classList.toggle('open');
  document.querySelector('.sb-backdrop').classList.toggle('show');
}
function closeSidebar(){
  document.querySelector('.sidebar')?.classList.remove('open');
  document.querySelector('.sb-backdrop')?.classList.remove('show');
}

// ── BOOT ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('ts-badge').textContent = new Date().toLocaleString('pt-BR');
  showView('overview', document.querySelector('.nav-item[data-view="overview"]'));
  window.addEventListener('resize', resizeVisibleCharts);
});
"""

# ══════════════════════════════════════════════════════════════════════════
# HTML helper text blocks
# ══════════════════════════════════════════════════════════════════════════
def machine_analysis_html():
    worst = max(machine, key=gpct)
    worst_inst = worst["instance"].replace("inst_","")
    max_n_opt = max(r["n"] for r in machine if is_opt(r))
    n_big = sum(1 for r in machine if r["n"] >= 19 and gpct(r) > 90)
    gaps_19 = [(r["instance"].replace("inst_",""), gpct(r)) for r in machine if r["n"] >= 19]
    gap_str = " · ".join(f"{i}={fn(g,1)}%" for i,g in gaps_19[:4])
    return f"""<p>Foram testadas <strong>{n_m} instâncias</strong> com n ∈ {{5,7,9,11,13,15,17,19,21,24}} jobs
    (3 sementes por tamanho) + inst_book (n=7). O modelo usa <em>O(n²) variáveis binárias</em> de precedência
    x_ij e big-M = Σp_j como coeficiente disjuntivo.</p>
    <p><strong>Transição crítica em n = {transition_n}:</strong> para n ≤ {max_n_opt} (exceto inst_n11_s03),
    o HiGHS prova a otimalidade dentro de {cfg['time_limit_machine']} s. A partir de n = {transition_n},
    o limite de tempo é atingido sistematicamente. Gap médio global: <strong>{fn(m_avg_gap,2)}%</strong>.
    Pior caso: <em>{worst_inst}</em> com gap de <strong>{fn(gpct(worst),1)}%</strong>.</p>
    <p><strong>Instâncias grandes (n ≥ 19):</strong> {n_big} de {sum(1 for r in machine if r['n']>=19)}
    com gap &gt; 90% — {gap_str}. A relaxação LP big-M é muito fraca aqui.</p>"""

def jobshop_analysis_html():
    best  = min(js_rows, key=lambda x: x[1] if x[1] is not None else 1e9)
    worst = max(js_rows, key=lambda x: x[1] if x[1] is not None else -1)
    return f"""<p>Foram testadas <strong>{n_js} instâncias</strong> da biblioteca JSPLIB (abz, ft, la, orb)
    com dimensões de 6×6 a 20×5. O ótimo conhecido serve de referência direta de qualidade.</p>
    <p><strong>{js_opt} instância(s) no ótimo:</strong> <em>ft06</em> (6×6, Cmax=55) e <em>la04</em> (10×5, Cmax=590).
    Gap médio vs ótimo JSPLIB: <strong>{fn(js_avg_gap_opt,2)}%</strong>.
    Melhor: <em>{best[0]['instance']}</em> (Δ={fn(best[1],2) if best[1] is not None else '0'}%).
    Mais difícil: <em>{worst[0]['instance']}</em> (Δ={fn(worst[1],2) if worst[1] is not None else '—'}%).</p>
    <p>Instâncias 10×10 (abz5, abz6, ft10, orb01) esgotam o limite de {cfg['time_limit_jobshop']} s,
    evidenciando a NP-dificuldade do JSP. O modelo gera O(J²·M) restrições disjuntivas.</p>"""

def flowshop_analysis_html():
    best  = min(flowshop, key=gpct)
    worst = max(flowshop, key=gpct)
    return f"""<p>Foram testadas <strong>{n_fs} instâncias</strong> com dimensões 3m×10j a 10m×20j.
    A restrição de mesma permutação em todas as máquinas adiciona O(J²·M) restrições extras.</p>
    <p><strong>{fs_opt} instância(s) no ótimo:</strong> <em>5m_10j</em> (Cmax=129) e <em>8m_10j</em> (Cmax=187).
    Gap médio: <strong>{fn(fs_avg_gap,2)}%</strong>.
    Melhor: <em>{best['instance'].replace('problem_','')}</em> (gap={fn(gpct(best),2)}%).
    Pior: <em>{worst['instance'].replace('problem_','')}</em> (gap={fn(gpct(worst),2)}%).</p>"""

def comp_table():
    rows = [
        ("Machine 1|r_j|ΣT_j", n_m, m_opt, n_m-m_opt, fn(100*m_opt/n_m,1)+"%", fn(m_avg_gap,2)+"%", fn(m_time,1)+"s", fn(m_time/n_m,2)+"s"),
        ("Job Shop J‖Cmax", n_js, js_opt, n_js-js_opt, fn(100*js_opt/n_js,1)+"%", fn(js_avg_gap_opt,2)+"%", fn(js_time,1)+"s", fn(js_time/n_js,2)+"s"),
        ("Flow Shop F|prmu|Cmax", n_fs, fs_opt, n_fs-fs_opt, fn(100*fs_opt/n_fs,1)+"%", fn(fs_avg_gap,2)+"%", fn(fs_time,1)+"s", fn(fs_time/n_fs,2)+"s"),
        ("TOTAL", n_total, n_opt, n_total-n_opt, fn(100*n_opt/n_total,1)+"%", "—", fn(tot_time,1)+"s", fn(tot_time/n_total,2)+"s"),
    ]
    out = ""
    for i,(nm2,nt,no,ntl,rate,gap,tt,tavg) in enumerate(rows):
        bold = " style='font-weight:700;border-top:1px solid #243d6a'" if i==3 else ""
        out += (f"<tr{bold}><td><strong>{nm2}</strong></td>"
                f"<td class='num'>{nt}</td><td class='num c-ok'>{no}</td>"
                f"<td class='num c-warn'>{ntl}</td><td class='num'>{rate}</td>"
                f"<td class='num c-warn'>{gap}</td><td class='num'>{tt}</td><td class='num'>{tavg}</td></tr>")
    return out

def conclusoes_html():
    max_n_opt = max(r["n"] for r in machine if is_opt(r))
    g13 = " · ".join(f"{r['instance'].replace('inst_','')}={fn(gpct(r),1)}%" for r in machine if r["n"]==13)
    g19 = " · ".join(f"{r['instance'].replace('inst_','')}={fn(gpct(r),1)}%" for r in machine if r["n"]>=19)
    js_best  = min(js_rows, key=lambda x: x[1] if x[1] is not None else 1e9)
    js_worst = max(js_rows, key=lambda x: x[1] if x[1] is not None else -1)
    fs_worst = max(flowshop, key=gpct); fs_best = min(flowshop, key=gpct)
    avg_opt_t = sum(r['runtime'] for r in machine if is_opt(r))/max(m_opt,1)
    return f"""
<h3>Síntese Geral</h3>
<p>Este trabalho implementou três formulações MILP disjuntivas (big-M) em Julia/JuMP + HiGHS.
Foram executadas <strong>{n_total} instâncias</strong> totais, com
<strong>{n_opt} soluções ótimas provadas</strong> ({fn(100*n_opt/n_total,1)}%)
e tempo total de solver de <strong>{fn(tot_time,1)} s</strong> (média {fn(tot_time/n_total,2)} s/instância).</p>

<h3>Machine Scheduling — 1 | r_j | ΣT_j</h3>
<p><strong>{m_opt}/{n_m}</strong> instâncias no ótimo ({fn(100*m_opt/n_m,1)}%).
Transição em <em>n = {transition_n}</em>: para n ≤ {max_n_opt} (exceto n11_s03),
o ótimo é provado em tempo médio de {fn(avg_opt_t,2)} s. inst_book (n=7) retorna ΣT_j = {ib_obj}.
Para n = 13: {g13}. Para n ≥ 19: {g19} — todos com gap &gt; 95%,
comprovando que a relaxação LP big-M é demasiadamente fraca para instâncias grandes.</p>

<h3>Job Shop — J ‖ C_max</h3>
<p><strong>{js_opt}/{n_js}</strong> no ótimo: ft06 (Cmax=55) e la04 (Cmax=590).
Gap médio vs ótimo JSPLIB: <strong>{fn(js_avg_gap_opt,2)}%</strong>.
Melhor approx.: {js_best[0]['instance']} (Δ={fn(js_best[1],2) if js_best[1] is not None else '0'}%).
Mais difícil: {js_worst[0]['instance']} (Δ={fn(js_worst[1],2) if js_worst[1] is not None else '—'}%).
Instâncias ≥ 10×10 esgotam os {cfg['time_limit_jobshop']} s de limite.</p>

<h3>Flow Shop — F | prmu | C_max</h3>
<p><strong>{fs_opt}/{n_fs}</strong> no ótimo: 5m×10j (Cmax=129) e 8m×10j (Cmax=187).
Gap médio: <strong>{fn(fs_avg_gap,2)}%</strong>.
Melhor: {fs_best['instance'].replace('problem_','')} (gap={fn(gpct(fs_best),2)}%).
Pior: {fs_worst['instance'].replace('problem_','')} (gap={fn(gpct(fs_worst),2)}%).</p>

<h3>Limitações e Perspectivas</h3>
<p>As formulações big-M produzem relaxações LP fracas pois H = Σp_j é elevado.
Para escalar: (i) formulações time-indexed (bound mais forte, O(n·H) variáveis),
(ii) decomposição Benders ou branch-and-price, ou (iii) heurísticas construtivas (NEH, ATCS).</p>"""

# ══════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════
HTML = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRABALHO 02 · Scheduling MILP · PPGEE/UFAM 2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<style>{CSS}</style>
</head>
<body>

<header class="topbar">
  <div class="hamburger" onclick="toggleSidebar()">☰</div>
  <div>
    <div class="tlogo">⚡ TRABALHO 02 — SCHEDULING MILP</div>
    <div class="tsub">OTIMIZAÇÃO 2026 · PPGEE/UFAM · PROF. KENNY VINENTE DOS SANTOS</div>
  </div>
  <div class="tright">
    <span class="chip">Julia {JVER}</span>
    <span class="chip">HiGHS + JuMP</span>
    <span class="chip">MILP Big-M</span>
    <span class="ts-badge" id="ts-badge">—</span>
  </div>
</header>

<div class="sb-backdrop" onclick="closeSidebar()"></div>
<nav class="sidebar">
  <div class="sb-sec">Visão Geral</div>
  <div class="nav-item" data-view="overview"    onclick="showView('overview',this)"><span class="nav-icon">🏠</span>Overview</div>
  <div class="nav-item" data-view="config"      onclick="showView('config',this)"><span class="nav-icon">⚙️</span>Configuração</div>
  <div class="nav-item" data-view="instances"   onclick="showView('instances',this)"><span class="nav-icon">🗂</span>Instâncias</div>
  <div class="sb-sec">Problemas</div>
  <div class="nav-item" data-view="machine"     onclick="showView('machine',this)"><span class="nav-icon">🔧</span>Machine Sched.</div>
  <div class="nav-item" data-view="jobshop"     onclick="showView('jobshop',this)"><span class="nav-icon">🏭</span>Job Shop</div>
  <div class="nav-item" data-view="flowshop"    onclick="showView('flowshop',this)"><span class="nav-icon">🔄</span>Flow Shop</div>
  <div class="sb-sec">Análise</div>
  <div class="nav-item" data-view="scalability" onclick="showView('scalability',this)"><span class="nav-icon">📈</span>Escalabilidade</div>
  <div class="nav-item" data-view="gantt"       onclick="showView('gantt',this)"><span class="nav-icon">📊</span>Diagramas Gantt</div>
  <div class="nav-item" data-view="summary"     onclick="showView('summary',this)"><span class="nav-icon">🎯</span>Comparativo</div>
  <div class="nav-item" data-view="conclusions" onclick="showView('conclusions',this)"><span class="nav-icon">📝</span>Conclusões</div>
  <div class="nav-item" data-view="refs"        onclick="showView('refs',this)"><span class="nav-icon">📚</span>Referências</div>
</nav>

<main class="main">

<!-- ═══ OVERVIEW ═══ -->
<section id="overview">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#051428">🏠</span>Dashboard de Resultados</div>
    <div class="sec-sub">Análise de 3 classes de scheduling · MILP disjuntivo · Julia/JuMP + HiGHS · Entrega até 09/07/2026</div>
  </div>
  <div class="kpi-grid">
    <div class="kpi-card" style="--kc:var(--ac)"><div class="kpi-glow"></div>
      <div class="kpi-val" id="kv-total">0</div><div class="kpi-label">Instâncias Executadas</div>
      <div class="kpi-sub">{n_m} Machine · {n_js} Job Shop · {n_fs} Flow Shop</div></div>
    <div class="kpi-card" style="--kc:var(--ok)"><div class="kpi-glow"></div>
      <div class="kpi-val" id="kv-opt">0</div><div class="kpi-label">Soluções Ótimas</div>
      <div class="kpi-sub">{m_opt} M · {js_opt} JS · {fs_opt} FS</div></div>
    <div class="kpi-card" style="--kc:var(--vio)"><div class="kpi-glow"></div>
      <div class="kpi-val" id="kv-rate">0</div><div class="kpi-label">Taxa de Optimalidade</div>
      <div class="kpi-sub">das {n_total} instâncias testadas</div></div>
    <div class="kpi-card" style="--kc:var(--warn)"><div class="kpi-glow"></div>
      <div class="kpi-val" id="kv-time">0s</div><div class="kpi-label">Tempo Total de Solver</div>
      <div class="kpi-sub">{fn(tot_time/n_total,2)}s/instância (média)</div></div>
    <div class="kpi-card" style="--kc:var(--gold)"><div class="kpi-glow"></div>
      <div class="kpi-val" id="kv-classes">0</div><div class="kpi-label">Classes de Problema</div>
      <div class="kpi-sub">Machine · Job Shop · Flow Shop</div></div>
  </div>
  <div class="chart-grid-3 fill">
    <div class="chart-box"><div class="chart-label">Status Geral das Soluções</div><canvas id="ch-status"></canvas></div>
    <div class="chart-box"><div class="chart-label">Tempo Total por Classe (s)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-class-time"></canvas></div>
    <div class="chart-box"><div class="chart-label">Taxa de Ótimos por Classe (%)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-opt-rate"></canvas></div>
  </div>
</section>

<!-- ═══ CONFIG ═══ -->
<section id="config">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#040e20">⚙️</span>Configuração do Solver</div>
    <div class="sec-sub">Parâmetros HiGHS livres conforme enunciado · Julia {JVER} · Execução: {TS}</div>
  </div>
  <div class="vbody">
  <div class="card">
    <div class="card-title">► HiGHS MILP SOLVER — PARÂMETROS</div>
    <div class="stat-row">
      <div class="stat-item"><div class="stat-val" style="color:var(--ac)">{cfg['time_limit_machine']}s</div><div class="stat-lbl">Time Limit Machine</div></div>
      <div class="stat-item"><div class="stat-val" style="color:var(--vio)">{cfg['time_limit_jobshop']}s</div><div class="stat-lbl">Time Limit Job Shop</div></div>
      <div class="stat-item"><div class="stat-val" style="color:var(--ok)">{cfg['time_limit_flowshop']}s</div><div class="stat-lbl">Time Limit Flow Shop</div></div>
      <div class="stat-item"><div class="stat-val" style="color:var(--gold)">{cfg['mip_rel_gap']}</div><div class="stat-lbl">MIP Relative Gap</div></div>
      <div class="stat-item"><div class="stat-val" style="color:var(--warn)">{cfg['mip_feas_tol']}</div><div class="stat-lbl">Feasibility Tol.</div></div>
    </div>
    <div class="formula">optimizer_with_attributes(HiGHS.Optimizer,
    "time_limit"                   =&gt; 30.0 / 60.0,
    "mip_rel_gap"                  =&gt; {cfg['mip_rel_gap']},
    "primal_feasibility_tolerance" =&gt; {cfg['mip_feas_tol']},
    "dual_feasibility_tolerance"   =&gt; {cfg['mip_feas_tol']},
    "output_flag"                  =&gt; false)</div>
  </div>
  <div class="card">
    <div class="card-title">► AMBIENTE DE EXECUÇÃO</div>
    <div class="stat-row">
      <div class="stat-item"><div class="stat-val" style="color:var(--ac)">{JVER}</div><div class="stat-lbl">Julia</div></div>
      <div class="stat-item"><div class="stat-val" style="color:var(--vio)">JuMP</div><div class="stat-lbl">Modelagem</div></div>
      <div class="stat-item"><div class="stat-val" style="color:var(--ok)">HiGHS</div><div class="stat-lbl">Solver MILP</div></div>
      <div class="stat-item"><div class="stat-val" style="color:var(--gold)">{n_total}</div><div class="stat-lbl">Instâncias</div></div>
    </div>
  </div>
  </div>
</section>

<!-- ═══ INSTANCES ═══ -->
<section id="instances">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#101428">🗂</span>Descrição das Instâncias</div>
    <div class="sec-sub">Origem e parâmetros dos {n_total} casos de teste das três classes</div>
  </div>

  <div class="tab-bar">
    <button class="tab-btn is-active" onclick="showTab('sec-inst','inst-resumo',this)">📊 Resumo</button>
    <button class="tab-btn" onclick="showTab('sec-inst','inst-machine',this)">🔧 Machine ({n_m})</button>
    <button class="tab-btn" onclick="showTab('sec-inst','inst-js',this)">🏭 Job Shop ({n_js})</button>
    <button class="tab-btn" onclick="showTab('sec-inst','inst-fs',this)">🔄 Flow Shop ({n_fs})</button>
  </div>
  <div class="tab-body"><div id="sec-inst" class="tab-host">
    <div id="inst-resumo" class="tab-panel fit is-active">
      <div class="chart-grid fill">
        <div class="chart-box"><div class="chart-label">Distribuição das Instâncias por Classe</div><canvas id="ch-inst-dist"></canvas></div>
        <div class="card" style="margin:0;overflow-y:auto">
          <div class="card-title">► RESUMO DA GERAÇÃO</div>
          <div class="analysis-box" style="margin:0">
            <p><strong>Machine ({n_m}):</strong> n ∈ {{5,7,9,11,13,15,17,19,21,24}}, 3 sementes (s01–s03) cada,
            mais inst_book (n=7, livro MO-book 3.5). Parâmetros inteiros observados — r_j ∈ [{m_r_rng}],
            p_j ∈ [{m_p_rng}], d_j ∈ [{m_d_rng}].</p>
            <p><strong>Job Shop ({n_js}):</strong> instâncias clássicas da JSPLIB com ótimos conhecidos
            (Fisher–Thompson, Lawrence, Adams–Balas–Zawack, Applegate–Cook).</p>
            <p><strong>Flow Shop ({n_fs}):</strong> instâncias de permutação em CSV, 3m×10j a 10m×20j.</p>
          </div>
        </div>
      </div>
    </div>
    <div id="inst-machine" class="tab-panel">
      <div class="filter-bar">
        <input id="mp-search" type="text" placeholder="🔍 filtrar..." oninput="filterTable('mp-tbody','mp-search','mp-status','mp-rc')">
        <select id="mp-status" onchange="filterTable('mp-tbody','mp-search','mp-status','mp-rc')">
          <option value="">Todos</option><option value="OPTIMAL">OPTIMAL</option><option value="TIME_LIMIT">TIME_LIMIT</option>
        </select>
        <button class="csv-btn" onclick="exportCSV('mp-tbl','instancias_machine.csv')">↓ CSV</button>
        <span class="row-count" id="mp-rc"></span>
      </div>
      <div class="table-wrap">
        <table id="mp-tbl">
          <thead><tr>
            <th onclick="sortTable('mp-tbl',0)">Instância<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('mp-tbl',1)">n<span class="sort-ic">⇅</span></th>
            <th>r_j (range)</th><th>p_j (range)</th><th>d_j (range)</th>
            <th onclick="sortTable('mp-tbl',5)">ΣT_j<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('mp-tbl',6)">Tempo<span class="sort-ic">⇅</span></th><th>Status</th>
          </tr></thead>
          <tbody id="mp-tbody">{machine_params_table()}</tbody>
        </table>
      </div>
    </div>
    <div id="inst-js" class="tab-panel">
      <div class="filter-bar">
        <button class="csv-btn" onclick="exportCSV('jsi-tbl','instancias_jobshop.csv')">↓ CSV</button>
      </div>
      <div class="table-wrap">
        <table id="jsi-tbl">
          <thead><tr><th>Instância</th><th>J×M</th><th>Operações</th><th>Ótimo JSPLIB</th><th>Fonte</th></tr></thead>
          <tbody>{tbl_js_instances()}</tbody>
        </table>
      </div>
    </div>
    <div id="inst-fs" class="tab-panel">
      <div class="filter-bar">
        <button class="csv-btn" onclick="exportCSV('fsi-tbl','instancias_flowshop.csv')">↓ CSV</button>
      </div>
      <div class="table-wrap">
        <table id="fsi-tbl">
          <thead><tr><th>Instância</th><th>Jobs</th><th>Máquinas</th><th>Operações</th><th>Tipo</th></tr></thead>
          <tbody>{tbl_fs_instances()}</tbody>
        </table>
      </div>
    </div>
  </div></div>
</section>

<!-- ═══ MACHINE ═══ -->
<section id="machine">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#051830">🔧</span>1 · Machine Scheduling — 1 | r<sub>j</sub> | ΣT<sub>j</sub></div>
    <div class="sec-sub">Máquina única · datas de liberação r_j · minimizar tardiness total · {n_m} instâncias</div>
  </div>
  <div class="stat-row" id="m-stats"></div>

  <div class="tab-bar">
    <button class="tab-btn is-active" onclick="showTab('sec-machine','m-charts',this)">📊 Gráficos</button>
    <button class="tab-btn" onclick="showTab('sec-machine','m-table',this)">📋 Tabela</button>
    <button class="tab-btn" onclick="showTab('sec-machine','m-analise',this)">📖 Análise</button>
    <button class="tab-btn" onclick="showTab('sec-machine','m-modelo',this)">🧮 Modelo</button>
    <button class="tab-btn" onclick="showTab('sec-machine','m-detail',this)">🔍 inst_book</button>
    <button class="tab-btn" onclick="showTab('sec-machine','m-scale',this)">📈 Escalabilidade</button>
    <button class="tab-btn" onclick="showTab('sec-machine','m-bound',this)">📐 Obj vs Bound</button>
  </div>
  <div class="tab-body"><div id="sec-machine" class="tab-host">
    <div id="m-modelo" class="tab-panel">
      <div class="formula">min  Σ T_j
s.t. s_j ≥ r_j                           ∀j    (liberação)
     T_j ≥ s_j + p_j − d_j,  T_j ≥ 0    ∀j    (tardiness)
     s_j ≥ s_i + p_i − H(1−x_ij)        ∀i&lt;j  (disjunção)
     s_i ≥ s_j + p_j − H·x_ij           ∀i&lt;j
     x_ij ∈ {{0,1}}    →   O(n²) variáveis binárias</div>
      <div class="analysis-box" style="border-left-color:var(--ac)">
        <p>Formulação <strong>big-M disjuntiva</strong> para máquina única com datas de liberação.
        Cada par (i,j) recebe uma binária x_ij que decide a ordem relativa; H = Σp_j garante que
        a restrição inativa não limite o início. Total de <em>O(n²)</em> variáveis binárias.</p>
      </div>
    </div>
    <div id="m-analise" class="tab-panel">
      <div class="analysis-box">{machine_analysis_html()}</div>
    </div>
    <div id="m-charts" class="tab-panel fit is-active">
      <div class="chart-grid">
        <div class="chart-box"><div class="chart-label">Tardiness Total (ΣT_j) por Instância</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-m-obj"></canvas></div>
        <div class="chart-box"><div class="chart-label">Tempo de Solver (s) por Instância</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-m-time"></canvas></div>
      </div>
      <div class="chart-grid-1">
        <div class="chart-box"><div class="chart-label">Gap MIP (%) — verde=0% · amarelo=&gt;0% · vermelho=&gt;50%</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-m-gap"></canvas></div>
      </div>
    </div>
    <div id="m-table" class="tab-panel">
      <div class="filter-bar">
        <input id="m-search" type="text" placeholder="🔍 filtrar instância..." oninput="filterTable('m-tbody','m-search','m-status-filter','m-rc')">
        <select id="m-status-filter" onchange="filterTable('m-tbody','m-search','m-status-filter','m-rc')">
          <option value="">Todos os status</option><option value="OPTIMAL">OPTIMAL</option><option value="TIME_LIMIT">TIME_LIMIT</option>
        </select>
        <button class="csv-btn" onclick="exportCSV('m-tbl','resultados_machine.csv')">↓ CSV</button>
        <span class="row-count" id="m-rc"></span>
      </div>
      <div class="table-wrap">
        <table id="m-tbl">
          <thead><tr>
            <th onclick="sortTable('m-tbl',0)">Instância<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('m-tbl',1)">n<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('m-tbl',2)">ΣT_j<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('m-tbl',3)">Bound<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('m-tbl',4)">Gap MIP<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('m-tbl',5)">Tempo<span class="sort-ic">⇅</span></th><th>Status</th>
          </tr></thead>
          <tbody id="m-tbody">{tbl_machine()}</tbody>
        </table>
      </div>
    </div>
    <div id="m-detail" class="tab-panel">
      <div class="card">
        <div class="card-title">► SCHEDULE ÓTIMO — inst_book (n=7, ΣT_j = {fn(ib_obj)})</div>
        <div class="analysis-box" style="margin-bottom:14px">
          <p>Instância do livro MO-book 3.5 (Guéret, Prins &amp; Sevaux 2000). 7 jobs, solução ótima garantida.
          Jobs com T_j &gt; 0 estão em <span style="color:var(--bad)">vermelho</span>.</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Job</th><th>r_j (release)</th><th>p_j (duração)</th><th>d_j (due)</th><th>s_j (início)</th><th>Término</th><th>T_j (tardiness)</th></tr></thead>
            <tbody>{ib_rows}</tbody>
          </table>
        </div>
      </div>
    </div>
    <div id="m-scale" class="tab-panel fit">
      <div class="chart-grid">
        <div class="chart-box"><div class="chart-label">Tempo (s) × n — verde=ótimo · amarelo=time limit</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-m-scale-time"></canvas></div>
        <div class="chart-box"><div class="chart-label">Gap MIP (%) × n — crescimento exponencial</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-m-scale-gap"></canvas></div>
      </div>
    </div>
    <div id="m-bound" class="tab-panel fit">
      <div class="chart-box"><div class="chart-label">Objetivo (ΣT_j) vs Bound LP — gap entre barras = dificuldade</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-m-bound"></canvas></div>
      <div class="chart-box"><div class="chart-label">Dispersão ΣT_j × n com ajuste quadrático (tendência de crescimento)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-m-scatter"></canvas></div>
    </div>
  </div></div>
</section>

<!-- ═══ JOB SHOP ═══ -->
<section id="jobshop">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#120830">🏭</span>2 · Job Shop Scheduling — J ‖ C<sub>max</sub></div>
    <div class="sec-sub">Roteiros fixos por job · minimizar makespan · {n_js} instâncias JSPLIB</div>
  </div>
  <div class="stat-row" id="js-stats"></div>

  <div class="tab-bar">
    <button class="tab-btn is-active" onclick="showTab('sec-jobshop','js-charts',this)">📊 Gráficos</button>
    <button class="tab-btn" onclick="showTab('sec-jobshop','js-table',this)">📋 Tabela</button>
    <button class="tab-btn" onclick="showTab('sec-jobshop','js-analise',this)">📖 Análise</button>
    <button class="tab-btn" onclick="showTab('sec-jobshop','js-modelo',this)">🧮 Modelo</button>
    <button class="tab-btn" onclick="showTab('sec-jobshop','js-jsplib',this)">🏆 vs Ótimo JSPLIB</button>
    <button class="tab-btn" onclick="showTab('sec-jobshop','js-bound',this)">📐 Obj vs Bound</button>
  </div>
  <div class="tab-body"><div id="sec-jobshop" class="tab-host">
    <div id="js-modelo" class="tab-panel">
      <div class="formula">min  Cmax
s.t. s_{{j,o+1}} ≥ s_{{j,o}} + p_{{j,o}}               ∀j,o   (precedência no job)
     Cmax ≥ s_{{j,last}} + p_{{j,last}}               ∀j
     s_{{j2,o2}} ≥ s_{{j1,o1}} + p_{{j1,o1}} − M(1−z)  (disjunção na máquina)
     s_{{j1,o1}} ≥ s_{{j2,o2}} + p_{{j2,o2}} − M·z
     z ∈ {{0,1}}   →   O(J²·M) variáveis binárias</div>
      <div class="analysis-box" style="border-left-color:var(--ac)">
        <p>Cada job tem um <strong>roteiro fixo</strong> de operações em máquinas dedicadas.
        Variáveis z ordenam pares de operações que disputam a mesma máquina (big-M).
        Objetivo: minimizar o <em>makespan</em> Cmax.</p>
      </div>
    </div>
    <div id="js-analise" class="tab-panel">
      <div class="analysis-box">{jobshop_analysis_html()}</div>
    </div>
    <div id="js-charts" class="tab-panel fit is-active">
      <div class="chart-grid">
        <div class="chart-box tall"><div class="chart-label">Cmax Obtido vs Ótimo JSPLIB</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-js-obj"></canvas></div>
        <div class="chart-box"><div class="chart-label">Tempo de Solver (s)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-js-time"></canvas></div>
      </div>
      <div class="chart-grid-1">
        <div class="chart-box"><div class="chart-label">Gap vs Ótimo JSPLIB (%) — verde=ótimo · amarelo=&lt;20% · vermelho=&gt;20%</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-js-gap"></canvas></div>
      </div>
    </div>
    <div id="js-table" class="tab-panel">
      <div class="filter-bar">
        <input id="js-search" type="text" placeholder="🔍 filtrar instância..." oninput="filterTable('js-tbody','js-search','js-status-filter','js-rc')">
        <select id="js-status-filter" onchange="filterTable('js-tbody','js-search','js-status-filter','js-rc')">
          <option value="">Todos</option><option value="OPTIMAL">OPTIMAL</option><option value="TIME_LIMIT">TIME_LIMIT</option>
        </select>
        <button class="csv-btn" onclick="exportCSV('js-tbl','resultados_jobshop.csv')">↓ CSV</button>
        <span class="row-count" id="js-rc"></span>
      </div>
      <div class="table-wrap">
        <table id="js-tbl">
          <thead><tr>
            <th onclick="sortTable('js-tbl',0)">Instância<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('js-tbl',1)">J×M<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('js-tbl',2)">Cmax<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('js-tbl',3)">Bound<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('js-tbl',4)">Ótimo<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('js-tbl',5)">Δ Ótimo<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('js-tbl',6)">Tempo<span class="sort-ic">⇅</span></th><th>Status</th>
          </tr></thead>
          <tbody id="js-tbody">{tbl_jobshop()}</tbody>
        </table>
      </div>
    </div>
    <div id="js-jsplib" class="tab-panel">
      <div class="analysis-box" style="margin-bottom:14px">
        <p><strong>Comparação direta com ótimos conhecidos da JSPLIB.</strong> Gap = (Cmax − Ótimo) / Ótimo × 100%.
        <span style="color:var(--ok)">Verde = ótimo atingido.</span>
        <span style="color:var(--warn)">Amarelo = gap &lt; 20%.</span>
        <span style="color:var(--bad)">Vermelho = gap ≥ 20%.</span></p>
      </div>
      <div class="filter-bar"><button class="csv-btn" onclick="exportCSV('jsplib-tbl','jobshop_vs_jsplib.csv')">↓ CSV</button></div>
      <div class="table-wrap">
        <table id="jsplib-tbl">
          <thead><tr><th>Instância</th><th>J×M</th><th>Ótimo JSPLIB</th><th>Cmax Obtido</th><th>Δ Ótimo (%)</th><th>Bound</th><th>Tempo</th><th>Status</th></tr></thead>
          <tbody>{tbl_jsplib()}</tbody>
        </table>
      </div>
    </div>
    <div id="js-bound" class="tab-panel fit">
      <div class="chart-box"><div class="chart-label">Objetivo vs Bound LP por Instância</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-js-bound"></canvas></div>
    </div>
  </div></div>
</section>

<!-- ═══ FLOW SHOP ═══ -->
<section id="flowshop">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#041a18">🔄</span>3 · Flow Shop de Permutação — F | prmu | C<sub>max</sub></div>
    <div class="sec-sub">Mesma permutação de jobs em todas as máquinas · minimizar makespan · {n_fs} instâncias</div>
  </div>
  <div class="stat-row" id="fs-stats"></div>

  <div class="tab-bar">
    <button class="tab-btn is-active" onclick="showTab('sec-flowshop','fs-charts',this)">📊 Gráficos</button>
    <button class="tab-btn" onclick="showTab('sec-flowshop','fs-table',this)">📋 Tabela</button>
    <button class="tab-btn" onclick="showTab('sec-flowshop','fs-analise',this)">📖 Análise</button>
    <button class="tab-btn" onclick="showTab('sec-flowshop','fs-modelo',this)">🧮 Modelo</button>
    <button class="tab-btn" onclick="showTab('sec-flowshop','fs-bound',this)">📐 Obj vs Bound</button>
  </div>
  <div class="tab-body"><div id="sec-flowshop" class="tab-host">
    <div id="fs-modelo" class="tab-panel">
      <div class="formula">min  Cmax
s.t. C_{{j,1}} ≥ p_{{j,1}}                              ∀j
     C_{{j,k}} ≥ C_{{j,k−1}} + p_{{j,k}}               ∀j,k   (sequência)
     C_{{j,k}} ≥ C_{{i,k}} + p_{{j,k}} − M(1−x_ij)    ∀i&lt;j, ∀k (permutação)
     C_{{i,k}} ≥ C_{{j,k}} + p_{{i,k}} − M·x_ij        ∀i&lt;j, ∀k
     Cmax ≥ C_{{j,m}}                                   ∀j
     x_ij ∈ {{0,1}}   →   O(J²·M) restrições disjuntivas</div>
      <div class="analysis-box" style="border-left-color:var(--ac)">
        <p>No <strong>flow shop de permutação</strong>, todos os jobs seguem a mesma sequência em
        todas as máquinas. As binárias x_ij definem essa permutação única, replicada por máquina,
        gerando <em>O(J²·M)</em> restrições. Objetivo: minimizar Cmax.</p>
      </div>
    </div>
    <div id="fs-analise" class="tab-panel">
      <div class="analysis-box">{flowshop_analysis_html()}</div>
    </div>
    <div id="fs-charts" class="tab-panel fit is-active">
      <div class="chart-grid">
        <div class="chart-box"><div class="chart-label">Makespan (Cmax) por Instância</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-fs-obj"></canvas></div>
        <div class="chart-box"><div class="chart-label">Tempo de Solver (s)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-fs-time"></canvas></div>
      </div>
      <div class="chart-grid-1">
        <div class="chart-box"><div class="chart-label">Gap MIP (%) — verde=&lt;5% · amarelo=5–30% · vermelho=&gt;30%</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-fs-gap"></canvas></div>
      </div>
    </div>
    <div id="fs-table" class="tab-panel">
      <div class="filter-bar">
        <input id="fs-search" type="text" placeholder="🔍 filtrar instância..." oninput="filterTable('fs-tbody','fs-search','fs-status-filter','fs-rc')">
        <select id="fs-status-filter" onchange="filterTable('fs-tbody','fs-search','fs-status-filter','fs-rc')">
          <option value="">Todos</option><option value="OPTIMAL">OPTIMAL</option><option value="TIME_LIMIT">TIME_LIMIT</option>
        </select>
        <button class="csv-btn" onclick="exportCSV('fs-tbl','resultados_flowshop.csv')">↓ CSV</button>
        <span class="row-count" id="fs-rc"></span>
      </div>
      <div class="table-wrap">
        <table id="fs-tbl">
          <thead><tr>
            <th onclick="sortTable('fs-tbl',0)">Instância<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('fs-tbl',1)">J×M<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('fs-tbl',2)">Cmax<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('fs-tbl',3)">Bound<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('fs-tbl',4)">Gap MIP<span class="sort-ic">⇅</span></th>
            <th onclick="sortTable('fs-tbl',5)">Tempo<span class="sort-ic">⇅</span></th><th>Status</th>
          </tr></thead>
          <tbody id="fs-tbody">{tbl_flowshop()}</tbody>
        </table>
      </div>
    </div>
    <div id="fs-bound" class="tab-panel fit">
      <div class="chart-box"><div class="chart-label">Objetivo vs Bound LP por Instância</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-fs-bound"></canvas></div>
    </div>
  </div></div>
</section>

<!-- ═══ SCALABILITY ═══ -->
<section id="scalability">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#041020">📈</span>Análise de Escalabilidade</div>
    <div class="sec-sub">Como tempo e gap evoluem com o tamanho da instância nas 3 classes · cada ponto = uma instância</div>
  </div>
  <div class="chart-grid-3 fill">
    <div class="chart-box"><div class="chart-label">Machine: Tempo (s) × n_jobs — verde=ótimo · amarelo=time limit</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-sc-m-time"></canvas></div>
    <div class="chart-box"><div class="chart-label">Job Shop: Tempo (s) × J×M — roxo=time limit · verde=ótimo</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-js-scale"></canvas></div>
    <div class="chart-box"><div class="chart-label">Flow Shop: Gap (%) × J×M — verde&lt;5% · amarelo&lt;30% · vermelho≥30%</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-fs-scale"></canvas></div>
  </div>
  <div class="chart-grid-1 fill">
    <div class="chart-box"><div class="chart-label">Tempo de Solver — Todas as {n_total} Instâncias (azul=Machine · roxo=Job Shop · verde=Flow Shop)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-all-time"></canvas></div>
  </div>
</section>

<!-- ═══ GANTT ═══ -->
<section id="gantt">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#041828">📊</span>Diagramas de Gantt</div>
    <div class="sec-sub">Visualização dos schedules ótimos · Scroll do mouse = zoom · Arraste = pan</div>
  </div>
  <div class="tab-bar">
    <button class="tab-btn is-active" onclick="showTab('sec-gantt','g-machine',this)">🔧 Machine</button>
    <button class="tab-btn" onclick="showTab('sec-gantt','g-js',this)">🏭 Job Shop</button>
    <button class="tab-btn" onclick="showTab('sec-gantt','g-fs',this)">🔄 Flow Shop</button>
  </div>
  <div class="tab-body"><div id="sec-gantt" class="tab-host">
    <div id="g-machine" class="tab-panel fit is-active">
      <div class="gantt-box">
        <div class="gantt-header"><div class="gantt-title">🔧 MACHINE SCHEDULING — inst_book (1 linha por job · vermelho = tardio)</div><div class="gantt-meta" id="gantt-machine-meta">—</div></div>
        <button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button>
        <canvas id="gantt-machine"></canvas>
        <div class="zoom-hint">[ scroll = zoom · drag = pan ]</div>
      </div>
    </div>
    <div id="g-js" class="tab-panel fit">
      <div class="gantt-box">
        <div class="gantt-header"><div class="gantt-title">🏭 JOB SHOP — ft06 (6×6, ótimo=55)</div><div class="gantt-meta" id="gantt-js-meta">—</div></div>
        <button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button>
        <canvas id="gantt-js"></canvas>
        <div class="zoom-hint">[ scroll = zoom · drag = pan ]</div>
      </div>
    </div>
    <div id="g-fs" class="tab-panel fit">
      <div class="gantt-box">
        <div class="gantt-header"><div class="gantt-title">🔄 FLOW SHOP — problem_3m_10j</div><div class="gantt-meta" id="gantt-fs-meta">—</div></div>
        <button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button>
        <canvas id="gantt-fs"></canvas>
        <div class="zoom-hint">[ scroll = zoom · drag = pan ]</div>
      </div>
    </div>
  </div></div>
</section>

<!-- ═══ SUMMARY ═══ -->
<section id="summary">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#100830">🎯</span>Resumo Comparativo</div>
    <div class="sec-sub">Desempenho do HiGHS nas três classes de problema</div>
  </div>
  <div class="tab-bar">
    <button class="tab-btn is-active" onclick="showTab('sec-summary','sum-charts',this)">📊 Gráficos</button>
    <button class="tab-btn" onclick="showTab('sec-summary','sum-prog',this)">📶 Progresso</button>
    <button class="tab-btn" onclick="showTab('sec-summary','sum-table',this)">📋 Tabela Comparativa</button>
  </div>
  <div class="tab-body"><div id="sec-summary" class="tab-host">
    <div id="sum-charts" class="tab-panel fit is-active">
      <div class="chart-grid-3 fill">
        <div class="chart-box"><div class="chart-label">Instâncias por Classe</div><canvas id="ch-sum-inst"></canvas></div>
        <div class="chart-box"><div class="chart-label">Tempo Médio por Instância (s)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-sum-avg-time"></canvas></div>
        <div class="chart-box"><div class="chart-label">Taxa de Ótimos por Classe (%)</div><button class="reset-btn" onclick="resetZoom(this)">↺ zoom</button><canvas id="ch-sum-opt"></canvas></div>
      </div>
    </div>
    <div id="sum-prog" class="tab-panel">
      <div class="card"><div class="card-title">► PROGRESSO DE OPTIMALIDADE</div><div class="progress-wrap" id="progress-bars"></div></div>
    </div>
    <div id="sum-table" class="tab-panel">
      <div class="card">
        <div class="card-title">► TABELA COMPARATIVA GERAL</div>
        <div class="filter-bar"><button class="csv-btn" onclick="exportCSV('comp-tbl','comparativo_geral.csv')">↓ CSV</button></div>
        <div class="table-wrap">
          <table id="comp-tbl">
            <thead><tr><th>Classe</th><th>Instâncias</th><th>Ótimos</th><th>Time Limit</th><th>Taxa Ótimos</th><th>Gap Médio</th><th>Tempo Total</th><th>Tempo Médio</th></tr></thead>
            <tbody>{comp_table()}</tbody>
          </table>
        </div>
      </div>
    </div>
  </div></div>
</section>

<!-- ═══ CONCLUSIONS ═══ -->
<section id="conclusions">
  <div class="sec-header">
    <div class="sec-title"><span class="sec-icon" style="background:#041808">📝</span>Conclusões</div>
    <div class="sec-sub">Análise acadêmica dos resultados computacionais</div>
  </div>
  <div class="vbody"><div class="card"><div class="conclusion-card">{conclusoes_html()}</div></div></div>
</section>

<!-- ═══ REFERENCES ═══ -->
<section id="refs">
  <div class="sec-header"><div class="sec-title"><span class="sec-icon" style="background:#100808">📚</span>Referências</div></div>
  <div class="vbody"><ul class="ref-list">
    <li><strong>[1]</strong> Guéret, C., Prins, C., &amp; Sevaux, M. (2000). <em>Applications of Optimization with Xpress-MP.</em> Dash Optimization. — Base para inst_book.</li>
    <li><strong>[2]</strong> Lawler et al. (1993). <em>Sequencing and scheduling: Algorithms and complexity.</em> Handbooks in OR&amp;MS, 4, 445–522.</li>
    <li><strong>[3]</strong> Manne, A. S. (1960). <em>On the Job-Shop Scheduling Problem.</em> Operations Research, 8(2), 219–223.</li>
    <li><strong>[4]</strong> Applegate, D., &amp; Cook, W. (1991). <em>A Computational Study of the Job-Shop Scheduling Problem.</em> ORSA JoC, 3(2), 149–156.</li>
    <li><strong>[5]</strong> Garey, M. R., Johnson, D. S., &amp; Sethi, R. (1976). <em>The Complexity of Flowshop and Jobshop Scheduling.</em> MOR, 1(2), 117–129.</li>
    <li><strong>[6]</strong> Huangfu, Q., &amp; Hall, J. A. J. (2018). <em>Parallelizing the dual revised simplex method.</em> MPC, 10(1).</li>
    <li><strong>[7]</strong> Dunning, I., Huchette, J., &amp; Lubin, M. (2017). <em>JuMP: A Modeling Language for Mathematical Optimization.</em> SIAM Review, 59(2), 295–320.</li>
    <li><strong>[8]</strong> Bezanson et al. (2017). <em>Julia: A fresh approach to numerical computing.</em> SIAM Review, 59(1), 65–98.</li>
    <li><strong>[9]</strong> Fisher, H., &amp; Thompson, G. L. (1963). <em>Probabilistic learning combinations of local job-shop scheduling rules.</em></li>
    <li><strong>[10]</strong> Santos, K. V. (2026). <em>Materiais AULA06 — Otimização 2026.</em> PPGEE/UFAM.</li>
    <li style="border-left-color:var(--mut);color:var(--mut);font-size:11px;background:transparent;border:none;text-align:center;font-family:'Share Tech Mono',monospace">
      GERADO AUTOMATICAMENTE · {TODAY} · JULIA {JVER} + HIGHS · PPGEE/UFAM 2026
    </li>
  </ul></div>
</section>
</main>

<button class="fab" onclick="window.print()" title="Imprimir / Exportar PDF">🖨️</button>

<script>
{JS.replace('__DATA__', DATA_JSON)}
</script>
</body></html>"""

out = os.path.join(ROOT, "dashboard.html")
open(out, "w", encoding="utf-8").write(HTML)
print(f"Dashboard salvo: {out}  ({len(HTML)//1024} KB)")
