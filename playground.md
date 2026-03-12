### Cell 2
Add the repo root to sys.path by locating the top-level core/ directory in the current layout.
```python
!pip -q install pytest numpy matplotlib
```

### Cell 3
Run the pytest suite from the notebook as a quick smoke test.
```python
!python3 -m pytest -q
```

### Cell 4
Build ipsundrum stages A-D, run impulse episodes, plot Ns/Ne traces, and sweep g to estimate phenomenal duration.
```python
import matplotlib.pyplot as plt
from core.ipsundrum_model import Builder, LoopParams
from core.evaluation import run_episode, phenomenal_duration
from core.recon_core import ScriptState

def impulse(t): return 1.0 if t == 0 else 0.0

# A
b = Builder()
netA, _ = b.stage_A()
netA.start_root(True)
trA = run_episode(netA, impulse, steps=40)

# B
netB, _ = b.stage_B()
netB.start_root(True)
trB = run_episode(netB, impulse, steps=60)

# C (damped)
bC = Builder(params=LoopParams(g=0.8, h=1.0))
netC, _ = bC.stage_C(cycles_required=10)
netC.start_root(True)
trC = run_episode(netC, impulse, steps=120)

# D (self-sustaining)
bD = Builder(params=LoopParams(g=1.2, h=1.0))
netD, _ = bD.stage_D(efference_threshold=0.05)
netD.start_root(True)
trD = run_episode(netD, impulse, steps=200)

plt.figure(); plt.plot(trA.Ns); plt.title("Stage A: Ns"); plt.show()
plt.figure(); plt.plot(trB.Ns, label="Ns"); plt.plot(trB.Ne, label="Ne"); plt.legend(); plt.title("Stage B"); plt.show()
plt.figure(); plt.plot(trC.Ns, label="Ns"); plt.plot(trC.Ne, label="Ne"); plt.legend(); plt.title("Stage C"); plt.show()
plt.figure(); plt.plot(trD.Ns, label="Ns"); plt.plot(trD.Ne, label="Ne"); plt.legend(); plt.title("Stage D"); plt.show()

# Time dilation curve vs gain
g_values = [0.1*i for i in range(1, 16)]
dur = []
for g in g_values:
    bb = Builder(params=LoopParams(g=g, h=1.0))
    nn, _ = bb.stage_C(cycles_required=10)
    nn.start_root(True)
    tt = run_episode(nn, impulse, steps=200)
    dur.append(phenomenal_duration(tt))

plt.figure()
plt.plot(g_values, dur, marker="o")
plt.title("Time dilation proxy (duration) vs gain g (h=1)")
plt.xlabel("g")
plt.ylabel("duration (Ns>=0.5)")
plt.show()
```

### Cell 5
Render ReCoN graphs with NetworkX (sub/por/sur/ret/aux edges), reload ipsundrum_model, and visualize stages A-D.
```python
import networkx as nx
import matplotlib.pyplot as plt
from core.recon_core import NodeKind
from core.ipsundrum_model import Builder, LoopParams, AffectParams

AUX_NODES = {"Ni","Nv","Na"}   # affect nodes (only present when affect is enabled)

def build_nx_from_recon(net, add_aux_edges=True, add_reverse_edges=True):
    G = nx.DiGraph()

    # nodes
    for nid, n in net.nodes.items():
        kind = n.kind.name
        state = n.script_state.name if n.kind == NodeKind.SCRIPT else n.terminal_state.name
        G.add_node(nid, kind=kind, state=state)

    # ReCoN control edges: parent->child (sub) and child->parent (sur)
    for nid, n in net.nodes.items():
        for c in getattr(n, "children", []):
            if c in net.nodes:
                G.add_edge(nid, c, etype="sub")
                if add_reverse_edges:
                    G.add_edge(c, nid, etype="sur")

    # por edges (predecessor->successor) and ret edges (successor->predecessor)
    for nid, n in net.nodes.items():
        suc = getattr(n, "successor", None)
        if suc is not None and suc in net.nodes:
            G.add_edge(nid, suc, etype="por")
        pred = getattr(n, "predecessor", None)
        if add_reverse_edges and pred is not None and pred in net.nodes:
            G.add_edge(nid, pred, etype="ret")

    # Optional: add "monitor" edges so physiology nodes appear connected *visually*
    # without pretending they are ReCoN children.
    if add_aux_edges and "Root" in net.nodes:
        for nid in AUX_NODES:
            if nid in net.nodes:
                # only add if orphan (parent=None) so we don't duplicate real sub edges
                if getattr(net.get(nid), "parent", None) is None:
                    G.add_edge("Root", nid, etype="aux")

    return G

def hierarchy_pos(G, root="Root"):
    # hierarchy from sub edges only
    sub_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("etype")=="sub"]
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(sub_edges)

    depth = {root: 0}
    q = [root]
    while q:
        u = q.pop(0)
        for v in H.successors(u):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)

    # put aux nodes at depth 1 if they exist
    for n in G.nodes():
        if depth.get(n, 999) == 999:
            # keep truly unreachable nodes low, but bring our aux nodes up a bit
            if n in AUX_NODES:
                depth[n] = 1
            else:
                depth[n] = 999

    levels = {}
    for n in G.nodes():
        levels.setdefault(depth[n], []).append(n)

    pos = {}
    max_depth = max(levels.keys()) if levels else 0
    for d in sorted(levels.keys()):
        nodes = sorted(levels[d])
        for i, n in enumerate(nodes):
            x = (i + 1) / (len(nodes) + 1)
            y = 1.0 - (d / (max_depth + 1 if max_depth >= 0 else 1))
            pos[n] = (x, y)

    return pos

def draw_recon(net, title="", add_aux_edges=True, add_reverse_edges=True):
    G = build_nx_from_recon(net, add_aux_edges=add_aux_edges, add_reverse_edges=add_reverse_edges)
    pos = hierarchy_pos(G, root="Root") if "Root" in G.nodes else nx.spring_layout(G, seed=0)

    labels = {n: f"{n}\n{G.nodes[n]['kind']}\n{G.nodes[n]['state']}" for n in G.nodes()}

    plt.figure(figsize=(11, 6))
    nx.draw_networkx_nodes(G, pos, node_size=1400)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    sub_edges = {(u,v) for u,v,d in G.edges(data=True) if d.get("etype")=="sub"}
    por_edges = {(u,v) for u,v,d in G.edges(data=True) if d.get("etype")=="por"}
    sur_edges = {(u,v) for u,v,d in G.edges(data=True) if d.get("etype")=="sur"}
    ret_edges = {(u,v) for u,v,d in G.edges(data=True) if d.get("etype")=="ret"}
    aux_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("etype")=="aux"]

    # Draw bidirectional edges once, with visible arrowheads.
    sub_sur = {tuple(sorted((u, v))) for (u, v) in sub_edges if (v, u) in sur_edges}
    por_ret = {tuple(sorted((u, v))) for (u, v) in por_edges if (v, u) in ret_edges}
    edge_kw = dict(arrows=True, arrowsize=18, min_source_margin=12, min_target_margin=12)

    if sub_sur:
        nx.draw_networkx_edges(G, pos, edgelist=sorted(sub_sur), arrowstyle="<|-|>", **edge_kw)
    if por_ret:
        nx.draw_networkx_edges(G, pos, edgelist=sorted(por_ret), arrowstyle="<|-|>", style="dashed", **edge_kw)
    if aux_edges:
        nx.draw_networkx_edges(G, pos, edgelist=aux_edges, arrows=False, style="dotted")

    elabels = {e: "sub/sur" for e in sub_sur}
    elabels.update({e: "por/ret" for e in por_ret})
    elabels.update({e: "aux" for e in aux_edges})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.show()


# ---- demo: visualize all stages ----
import importlib
import core.ipsundrum_model as im

importlib.reload(im)          # ensures you get the latest file after edits
Builder = im.Builder          # bring into local scope

# optional: also pull params types
LoopParams = im.LoopParams
AffectParams = im.AffectParams

b = Builder()

netA, _ = b.stage_A()
draw_recon(netA, "Stage A network")

netB, _ = b.stage_B()
draw_recon(netB, "Stage B network")

netC, _ = b.stage_C(cycles_required=10)
draw_recon(netC, "Stage C network")

netD, _ = b.stage_D(efference_threshold=0.05)
draw_recon(netD, "Stage D network")
```

### Cell 6
Print the Stage C node inventory with kinds, parents, and children for quick structure inspection.
```python
def dump_net(net):
    print("NODES:", sorted(net.nodes.keys()))
    for nid in sorted(net.nodes.keys()):
        n = net.get(nid)
        kind = n.kind.name
        parent = n.parent
        children = list(n.children)
        print(f"{nid:>3} kind={kind:<8} parent={parent} children={children}")

# rebuild + dump
import importlib
import core.ipsundrum_model as im
importlib.reload(im)

b = im.Builder()
netC, _ = b.stage_C(cycles_required=10)
dump_net(netC)
```

### Cell 7
Define TraceFull and run_episode_full to log ipsundrum loop + affect diagnostics.
```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from core.recon_core import ScriptState
from core.ipsundrum_model import Builder, LoopParams, AffectParams

@dataclass
class TraceFull:
    t: list
    I: list

    # Ipsundrum loop observables
    Ns: list
    E_prev: list
    E: list
    X: list
    M: list
    Ne_node: list

    # Affect (Barrett-style)
    bb_true: list
    bb_model: list
    pe: list
    Nv: list
    Na: list

    # Couplings / diagnostics
    g_base: list
    g_eff: list
    precision: list
    alpha_eff: list

    # Script state (top-down gating)
    P_state: list


def run_episode_full(net, stimulus, steps, seed=0, builder=None):
    rng = np.random.default_rng(seed)
    tr = TraceFull(
        t=[], I=[],
        Ns=[], E_prev=[], E=[], X=[], M=[], Ne_node=[],
        bb_true=[], bb_model=[], pe=[], Nv=[], Na=[],
        g_base=[], g_eff=[], precision=[], alpha_eff=[],
        P_state=[]
    )

    E_last = 0.0

    for _ in range(steps):
        t = net.tick
        I = float(stimulus(t))

        # tick-driven physiology update
        if hasattr(net, "_update_ipsundrum_sensor"):
            net._update_ipsundrum_sensor(I, rng=rng)  # type: ignore[attr-defined]
        else:
            net.set_sensor_value("Ns", I)

        net.step()

        # read visible nodes
        Ns = float(net.get("Ns").activation) if "Ns" in net.nodes else float("nan")
        Ne_node = float(net.get("Ne").activation) if "Ne" in net.nodes else float("nan")
        Nv = float(net.get("Nv").activation) if "Nv" in net.nodes else float("nan")
        Na = float(net.get("Na").activation) if "Na" in net.nodes else float("nan")

        P_state = net.get("P").script_state.name if "P" in net.nodes else net.get("R").script_state.name

        st = getattr(net, "_ipsundrum_state", {}) if hasattr(net, "_ipsundrum_state") else {}
        X = float(st.get("internal", float("nan")))
        M = float(st.get("motor", float("nan")))
        E = float(st.get("reafferent", float("nan")))
        g_base = float(st.get("g", float("nan")))
        g_eff = float(st.get("g_eff", g_base))
        precision = float(st.get("precision_eff", 1.0))
        alpha = float(st.get("alpha_eff", float("nan")))

        bb_true = float(st.get("bb_true", float("nan")))
        bb_model = float(st.get("bb_model", float("nan")))
        pe = float(st.get("pe", float("nan")))

        # store
        tr.t.append(t); tr.I.append(I)
        tr.Ns.append(Ns)
        tr.E_prev.append(E_last)
        tr.E.append(E)
        tr.X.append(X); tr.M.append(M); tr.Ne_node.append(Ne_node)

        tr.bb_true.append(bb_true); tr.bb_model.append(bb_model); tr.pe.append(pe)
        tr.Nv.append(Nv); tr.Na.append(Na)

        tr.g_base.append(g_base); tr.g_eff.append(g_eff); tr.precision.append(precision)
        tr.alpha_eff.append(alpha)

        tr.P_state.append(P_state)

        E_last = E

    return tr
```

### Cell 8
Plot trace diagnostics for sensory/reafference, control state, affect, and alpha_eff.
```python
import numpy as np
import matplotlib.pyplot as plt
from core.recon_core import ScriptState

def plot_mechanisms(tr: TraceFull, title=""):
    # ---- Ipsundrum: sensory + reafference + top-down gating ----
    plt.figure()
    plt.plot(tr.I, label="I_ext")
    plt.plot(tr.Ns, label="Ns")
    plt.plot(tr.E_prev, label="E_{t-1} (used in Ns)")
    plt.plot(tr.E, label="E_t (reafferent)")
    plt.title(f"{title} Ipsundrum: sensory & reafference")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tr.X, label="X (thick-moment integrator)")
    plt.plot(tr.M, label="M (motor command)")
    plt.plot(tr.Ne_node, label="Ne node (efference copy)")
    plt.title(f"{title} Ipsundrum: thick moment -> motor -> efference")
    plt.legend()
    plt.show()

    # Script state as a step plot
    state_to_int = {s.name: i for i, s in enumerate(ScriptState)}
    p_int = [state_to_int.get(s, np.nan) for s in tr.P_state]
    plt.figure()
    plt.step(tr.t, p_int, where="post")
    plt.yticks(list(state_to_int.values()), list(state_to_int.keys()))
    plt.title(f"{title} ReCoN control: P state over time")
    plt.show()

    # ---- Affect (Barrett-style): interoception + affect ----
    plt.figure()
    plt.plot(tr.bb_true, label="bb_true")
    plt.plot(tr.bb_model, label="bb_model")
    plt.plot(tr.pe, label="prediction error (pe)")
    plt.title(f"{title} Affect: interoceptive prediction loop")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tr.Nv, label="Nv (valence)")
    plt.plot(tr.Na, label="Na (arousal)")
    plt.title(f"{title} Affect: valence/arousal readout")
    plt.legend()
    plt.show()

    # ---- Thermostat coupling into ipsundrum criticality ----
    plt.figure()
    plt.plot(tr.g_base, label="g_base")
    plt.plot(tr.g_eff, label="g_eff (after affect modulation)")
    plt.plot(tr.precision, label="precision (after affect modulation)")
    plt.title(f"{title} Coupling: affect -> gain/precision")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tr.alpha_eff, label="alpha_eff")
    plt.axhline(1.0, linestyle="--", label="critical boundary alpha=1")
    plt.title(f"{title} Ipsundrum regime indicator (alpha_eff)")
    plt.legend()
    plt.show()
```

### Cell 9
Configure affect + loop parameters for Stage D, run a pulse stimulus, and plot mechanisms.
```python
from core.ipsundrum_model import Builder, LoopParams, AffectParams
from core.recon_core import ScriptState

def pulse_pain(t: int) -> float:
    return 1.0 if t < 6 else 0.0

aff = AffectParams(
    enabled=True,
    valence_scale=3.0,  # <-- key change
    k_homeo=0.10,
    k_pe=0.50,
    demand_motor=0.20,
    demand_stim=0.30,

    modulate_g=True,
    k_g_arousal=0.8,
    k_g_unpleasant=0.8,

    modulate_precision=True,
    precision_base=1.0,
    k_precision_arousal=0.5,
)

loop = LoopParams(
    g=1.0,
    h=1.0,
    internal_decay=0.6,
    fatigue=0.01,
    nonlinearity="linear",
    saturation=True,
)

b = Builder(params=loop, affect=aff)
net, w = b.stage_D(efference_threshold=0.05)
net.start_root(True)

tr = run_episode_full(net, stimulus=pulse_pain, steps=200, seed=0, builder=b)
plot_mechanisms(tr, title="Stage D (Ipsundrum + Affect ON)")
```

### Cell 10
Decompose Ns drive into external input and precision-weighted reafference.
```python
import numpy as np
import matplotlib.pyplot as plt

drive = np.array(tr.I) + np.array(tr.precision)*np.array(tr.E_prev)
plt.figure()
plt.plot(tr.I, label="I_ext")
plt.plot(np.array(tr.precision)*np.array(tr.E_prev), label="precision * E_{t-1}")
plt.plot(drive, label="Total drive")
plt.title("Decomposition of Ns drive")
plt.legend()
plt.show()
```

### Cell 11
Run ablations (Ipsundrum only vs Ipsundrum + Affect) and compare Ns trajectories.
```python
from core.ipsundrum_model import Builder, LoopParams, AffectParams
from core.recon_core import ScriptState
import numpy as np
import matplotlib.pyplot as plt

def pulse_pain(t): return 1.0 if t < 6 else 0.0

base_loop = LoopParams(
    g=1.0, h=1.0, internal_decay=0.6, fatigue=0.01,
    nonlinearity="linear", saturation=True
)

def run_case(name, aff):
    b = Builder(params=base_loop, affect=aff)
    net, _ = b.stage_D(efference_threshold=0.05)
    net.start_root(True)
    tr = run_episode_full(net, stimulus=pulse_pain, steps=200, seed=0, builder=b)
    return name, tr

cases = []

# Ipsundrum only
cases.append(run_case(
    "Ipsundrum only",
    aff=AffectParams(enabled=False)
))

# Ipsundrum + Affect
cases.append(run_case(
    "Ipsundrum + Affect",
    aff=AffectParams(
        enabled=True,
        valence_scale=3.0,
        k_homeo=0.10, k_pe=0.50,
        demand_motor=0.20, demand_stim=0.30,
        modulate_g=True, k_g_arousal=0.8, k_g_unpleasant=0.8,
        modulate_precision=True, precision_base=1.0, k_precision_arousal=0.5
    )
))

plt.figure(figsize=(9,5))
for name, tr in cases:
    plt.plot(tr.Ns, label=name)
plt.title("Ablation: Ns(t) under identical stimulus")
plt.legend()
plt.show()
```

### Cell 12
Provide paper-style plotting helpers with stimulus end/crossing annotations and regime shading.
```python
import numpy as np
import matplotlib.pyplot as plt

def _get_arr(tr, name, default=None):
    if hasattr(tr, name):
        return np.asarray(getattr(tr, name), dtype=float)
    if default is None:
        raise AttributeError(f"TraceFull missing field: {name}")
    return np.asarray(default, dtype=float)

def find_stim_off(tr, eps=1e-12):
    I = _get_arr(tr, "I")
    nz = np.where(np.abs(I) > eps)[0]
    if len(nz) == 0:
        return None
    return int(nz.max() + 1)  # boundary right after last nonzero

def find_alpha_cross(tr, threshold=1.0):
    alpha = _get_arr(tr, "alpha_eff")
    for i in range(1, len(alpha)):
        if alpha[i-1] > threshold and alpha[i] <= threshold:
            return int(i)
    return None

def annotate_lines(ax, t_off=None, t_cross=None):
    if t_off is not None:
        ax.axvline(t_off, linestyle="--", label="stimulus ends (t_off)")
    if t_cross is not None:
        ax.axvline(t_cross, linestyle=":", label="alpha crosses 1 (t*)")

def shade_ipsundrum(ax, t, alpha, threshold=1.0):
    # shade the x-range where alpha > 1
    mask = alpha > threshold
    ax.fill_between(t, 0, 1, where=mask, alpha=0.12, transform=ax.get_xaxis_transform(), label="alpha > 1 (ipsundrum regime)")

def plot_paper_figures(tr, title=""):
    t = _get_arr(tr, "t", default=np.arange(len(tr.I)))
    I = _get_arr(tr, "I")
    Ns = _get_arr(tr, "Ns")
    E = _get_arr(tr, "E", default=_get_arr(tr, "reafferent", default=np.zeros_like(Ns)))
    precision = _get_arr(tr, "precision", default=np.ones_like(Ns))
    alpha = _get_arr(tr, "alpha_eff", default=np.full_like(Ns, np.nan))

    # E_prev: use stored if present, else shift E by 1
    if hasattr(tr, "E_prev"):
        E_prev = _get_arr(tr, "E_prev")
    else:
        E_prev = np.concatenate([[0.0], E[:-1]])

    # decomposition terms (match your earlier plot)
    term_reaff = precision * E_prev
    total_drive = I + term_reaff

    t_off = find_stim_off(tr)
    t_cross = find_alpha_cross(tr)

    # --- Figure 1: Ns + E with criticality shading ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, I, label="I_ext")
    ax.plot(t, Ns, label="Ns")
    ax.plot(t, E_prev, label="E_{t-1} (used)")
    ax.plot(t, E, label="E_t")
    if np.all(np.isfinite(alpha)):
        shade_ipsundrum(ax, t, alpha, threshold=1.0)
    annotate_lines(ax, t_off=t_off, t_cross=t_cross)
    ax.set_title(f"{title} Ipsundrum: Ns / reafference + regime")
    ax.legend()
    plt.show()

    # --- Figure 2: Drive decomposition + markers ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, I, label="I_ext")
    ax.plot(t, term_reaff, label="precision * E_{t-1}")
    ax.plot(t, total_drive, label="Total drive")
    annotate_lines(ax, t_off=t_off, t_cross=t_cross)
    ax.set_title(f"{title} Drive decomposition (Ipsundrum + Affect)")
    ax.legend()
    plt.show()

    # --- Figure 3: alpha_eff itself (regime proof) ---
    if np.all(np.isfinite(alpha)):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(t, alpha, label="alpha_eff")
        ax.axhline(1.0, linestyle="--", label="alpha = 1")
        annotate_lines(ax, t_off=t_off, t_cross=t_cross)
        ax.set_title(f"{title} Ipsundrum regime indicator (alpha_eff)")
        ax.legend()
        plt.show()

    print("t_off =", t_off, "| t* (alpha crosses 1) =", t_cross)

# Usage:
plot_paper_figures(tr, title="Stage D (Ipsundrum + Affect)")
```

### Cell 13
Generate ablation figures (Ipsundrum only vs +Affect) and print crossing summaries.
```python
import numpy as np
import matplotlib.pyplot as plt

from core.ipsundrum_model import Builder, LoopParams, AffectParams
from core.recon_core import ScriptState

# --- stimulus ---
def pulse_pain(t: int) -> float:
    return 1.0 if t < 6 else 0.0


# ---------- helpers (robust to slightly different TraceFull versions) ----------

def _arr(tr, name, fallback=None):
    if hasattr(tr, name):
        return np.asarray(getattr(tr, name), dtype=float)
    if fallback is not None:
        return np.asarray(fallback, dtype=float)
    return None

def _get_E(tr):
    # prefer E (new traces), else use reafferent
    E = _arr(tr, "E")
    if E is None:
        E = _arr(tr, "reafferent")
    return E

def _get_E_prev(tr):
    # prefer E_prev if present, else shift E by 1
    E_prev = _arr(tr, "E_prev")
    if E_prev is not None:
        return E_prev
    E = _get_E(tr)
    if E is None:
        return None
    return np.concatenate([[0.0], E[:-1]])

def find_stim_off(tr, eps=1e-12):
    I = _arr(tr, "I")
    nz = np.where(np.abs(I) > eps)[0]
    if len(nz) == 0:
        return None
    return int(nz.max() + 1)

def find_alpha_cross(tr, threshold=1.0):
    alpha = _arr(tr, "alpha_eff")
    if alpha is None:
        return None
    for i in range(1, len(alpha)):
        if alpha[i-1] > threshold and alpha[i] <= threshold:
            return int(i)
    return None

def run_case(name, loop_params, aff_params, steps=200, seed=0):
    b = Builder(params=loop_params, affect=aff_params)
    net, _ = b.stage_D(efference_threshold=0.05)
    net.start_root(True)
    tr = run_episode_full(net, stimulus=pulse_pain, steps=steps, seed=seed, builder=b)
    return name, tr


# ---------- main generator ----------

def make_ablation_figures(steps=200, seed=0, title_prefix=""):
    # Base ipsundrum params (same across all ablations)
    loop = LoopParams(
        g=1.0, h=1.0, internal_decay=0.6, fatigue=0.01,
        nonlinearity="linear", saturation=True
    )

    # Affect ON params (Barrett-style thermostat)
    aff_on = AffectParams(
        enabled=True,
        valence_scale=3.0,   # key: avoid pinning Nv at 0
        k_homeo=0.10,
        k_pe=0.50,
        demand_motor=0.20,
        demand_stim=0.30,

        modulate_g=True,
        k_g_arousal=0.8,
        k_g_unpleasant=0.8,

        modulate_precision=True,
        precision_base=1.0,
        k_precision_arousal=0.5,
    )
    aff_off = AffectParams(enabled=False)

    cases = [
        ("Ipsundrum only",            aff_off),
        ("Ipsundrum + Affect",        aff_on),
    ]

    results = [run_case(name, loop, a, steps=steps, seed=seed) for name, a in cases]

    # ---- Figure 1: Ns overlay ----
    plt.figure(figsize=(9, 4.5))
    for name, tr in results:
        Ns = _arr(tr, "Ns")
        plt.plot(Ns, label=name)
    plt.title(f"{title_prefix} Ablation: Ns(t) under identical stimulus")
    plt.xlabel("t")
    plt.ylabel("Ns")
    plt.legend()
    plt.show()

    # ---- Figure 2: Decomposition per case (1x2) ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=False)
    if len(results) == 1:
        axs = [axs]

    # shared line styles for paper readability
    styles = {
        "I_ext": dict(linestyle="-"),
        "prec_E": dict(linestyle="--"),
        "total": dict(linestyle="-."),  # total
    }

    for ax, (name, tr) in zip(axs, results):
        t = _arr(tr, "t")
        if t is None:
            t = np.arange(len(tr.I))
        I = _arr(tr, "I")
        precision = _arr(tr, "precision")
        E_prev = _get_E_prev(tr)
        term_reaff = precision * E_prev
        total_drive = I + term_reaff

        t_off = find_stim_off(tr)
        t_cross = find_alpha_cross(tr)

        ax.plot(I, label="I_ext", **styles["I_ext"])
        ax.plot(term_reaff, label="precision*E_{t-1}", **styles["prec_E"])
        ax.plot(total_drive, label="Total drive", **styles["total"])

        if t_off is not None:
            ax.axvline(t_off, linestyle="--")
        if t_cross is not None:
            ax.axvline(t_cross, linestyle=":")

        ax.set_title(name)

    # one shared legend
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=3)
    # fig.suptitle(f"{title_prefix} Drive decomposition (Ipsundrum + Affect)\n"
    #              f"vertical lines: stimulus ends (--) and alpha crosses 1 (:)", y=1.02)
    # plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.suptitle(f"{title_prefix} Drive decomposition (Ipsundrum + Affect)\n"
                f"vertical lines: stimulus ends (--) and alpha crosses 1 (:)", y=1.08)
    plt.tight_layout()
    plt.show()

    # ---- Figure 3: alpha_eff per case (1x2) ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    if len(results) == 1:
        axs = [axs]

    for ax, (name, tr) in zip(axs, results):
        alpha = _arr(tr, "alpha_eff")
        if alpha is None:
            ax.text(0.1, 0.5, "alpha_eff missing in TraceFull", transform=ax.transAxes)
            continue

        t_off = find_stim_off(tr)
        t_cross = find_alpha_cross(tr)

        ax.plot(alpha, label="alpha_eff")
        ax.axhline(1.0, linestyle="--", label="alpha=1")

        if t_off is not None:
            ax.axvline(t_off, linestyle="--")
        if t_cross is not None:
            ax.axvline(t_cross, linestyle=":")

        ax.set_title(name)
        ax.set_ylim(0.0, max(1.5, float(np.nanmax(alpha)) + 0.05))

    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2)
    # fig.suptitle(f"{title_prefix} Ipsundrum regime indicator alpha_eff\n"
    #              f"alpha>1 ~= attractor-capable; alpha<1 ~= damped thick-moment", y=1.02)
    # plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.suptitle(f"{title_prefix} Ipsundrum regime indicator alpha_eff\n"
                f"alpha>1 ~= attractor-capable; alpha<1 ~= damped thick-moment", y=1.08)
    plt.tight_layout()
    plt.show()

    # print summary table of crossings
    print("\nAblation summary:")
    for name, tr in results:
        t_off = find_stim_off(tr)
        t_cross = find_alpha_cross(tr)
        print(f"  {name:>26} | t_off={t_off} | t* (alpha-><=1)={t_cross}")

    return results


# Run it:
results = make_ablation_figures(title_prefix="Stage D")
```

### Cell 14
Raise matplotlib animation embed limit to avoid truncation in JS output.
```python
import matplotlib as mpl
mpl.rcParams["animation.embed_limit"] = 200  # MB; prevents truncation of JS animation
```

### Cell 15
GridWorld demo using the shared viz_utils animation (Recon, Ipsundrum [prev. humphrey], Ipsundrum+affect [prev. humphrey_barrett]).
```python
import matplotlib as mpl
mpl.rcParams["animation.html"] = "jshtml"

from experiments.viz_utils import gridworld_viz
gridworld_viz.run_animation(horizon=10)
```

### Cell 16
CorridorWorld demo using the shared viz_utils animation (Recon, Ipsundrum [prev. humphrey], Ipsundrum+affect [prev. humphrey_barrett]).
```python
import matplotlib as mpl
mpl.rcParams["animation.html"] = "jshtml"

from experiments.viz_utils import corridor_viz
corridor_viz.run_animation(horizon=6)
```
### Cell 17
Run the full experiment suite (familiarity, exploratory play, lesion, pain-tail) and save results.
```python
!./run_experiments.sh
```