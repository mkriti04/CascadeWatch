"""
CascadeWatch Dashboard  –  python dashboard.py  →  http://127.0.0.1:8050
"""
import os, pickle, numpy as np, pandas as pd, networkx as nx, joblib
from collections import Counter
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State

# ── paths ──────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH  = os.path.join(BASE, "cascadewatch_graph.pkl")
CSV_PATH    = os.path.join(BASE, "synthetic_cascade_dataset.csv")
MODEL_PATH  = os.path.join(BASE, "rf_cascade_model.joblib")
LE_PATH     = os.path.join(BASE, "label_encoder.joblib")

# ── load once ──────────────────────────────────────────────────────────────────
print("Loading …")
with open(GRAPH_PATH, "rb") as f:
    GRAPH = pickle.load(f)
for n, d in GRAPH.nodes(data=True):
    if "type" in d and "node_type" not in d:
        d["node_type"] = d["type"]

DF  = pd.read_csv(CSV_PATH)
CLF = joblib.load(MODEL_PATH)
LE  = joblib.load(LE_PATH)

FEATURES = [
    "node_type_encoded","elevation","flood_depth","fragility_score",
    "degree_centrality","betweenness_centrality","redistributed_load",
    "overload_ratio","failed_neighbors_prev","cross_layer_dependency_stress",
    "distance_from_epicenter","was_failed_prev",
    "betweenness_centrality_log","redistributed_load_log","overload_ratio_log",
]
TOTAL_BY_TYPE = Counter(d.get("node_type","unknown") for _,d in GRAPH.nodes(data=True))
GEO_POS = {n:(float(d["x"]),float(d["y"])) for n,d in GRAPH.nodes(data=True) if "x" in d and "y" in d}
print(f"Graph: {GRAPH.number_of_nodes()} nodes, {GRAPH.number_of_edges()} edges  –  Ready.\n")

# ── simulation ─────────────────────────────────────────────────────────────────
def fragility(nt, depth):
    if nt == "road":      return 1/(1+np.exp(-10*(depth-0.3)))
    if nt in ("substation","power"): return 1/(1+np.exp(-6*(depth-1.2)))
    if nt == "hospital":  return 1/(1+np.exp(-4*(depth-1.5)))
    return 0.0

def run_prediction(intensity, steps=10):
    baseline = DF[DF["timestep"]==0].drop_duplicates("node_id").set_index("node_id")
    all_e = [d.get("elevation_m",0.0) for _,d in GRAPH.nodes(data=True)]
    max_e = max(all_e) if max(all_e)>0 else 1.0
    base_depths = {}
    for n,d in GRAPH.nodes(data=True):
        bd = d.get("flood_depth_m",0.0)
        if bd==0.0:
            e = d.get("elevation_m",max_e)
            bd = max(0.05,(1.0-e/max_e)*1.5)
        base_depths[n] = bd
    failed, nb_map = set(), {n:list(GRAPH.neighbors(n)) for n in GRAPH.nodes()}
    final_df = None
    for t in range(steps):
        rows = []
        for n,d in GRAPH.nodes(data=True):
            nt = d.get("node_type","road")
            nd = base_depths[n]*intensity
            wf = int(n in failed)
            fn = sum(1 for nb in nb_map[n] if nb in failed)
            if n in baseline.index:
                r=baseline.loc[n]; el=r["elevation"]; dg=r["degree_centrality"]
                bw=r["betweenness_centrality"]; bl=r["betweenness_centrality_log"]
                de=r["distance_from_epicenter"]
                ov=min(r["overload_ratio"]*(1+0.3*t),5.0); ol=np.log1p(ov)
                ds=min(r["cross_layer_dependency_stress"]+0.1*fn,1.0)
            else:
                el=d.get("elevation_m",0.0); dg=bw=bl=de=ov=ol=ds=0.0
            try: et=LE.transform([nt])[0]
            except: et=LE.transform(["road"])[0]
            rows.append({"node_id":n,"node_type_encoded":et,"elevation":el,
                "flood_depth":nd,"fragility_score":fragility(nt,nd),
                "degree_centrality":dg,"betweenness_centrality":bw,
                "redistributed_load":0.0,"overload_ratio":ov,
                "failed_neighbors_prev":fn,"cross_layer_dependency_stress":ds,
                "distance_from_epicenter":de,"was_failed_prev":wf,
                "betweenness_centrality_log":bl,"redistributed_load_log":0.0,
                "overload_ratio_log":ol})
        pf = pd.DataFrame(rows)
        pr = CLF.predict_proba(pf[FEATURES].fillna(0))[:,1]
        pf["predicted_failure"]=(pr>=0.5).astype(int)
        pf["failure_prob"]=pr
        for _,r in pf.iterrows():
            if r["predicted_failure"]==1: failed.add(r["node_id"])
        final_df=pf.copy()
    final_df["predicted_failure"]=final_df["node_id"].apply(lambda n:int(n in failed))
    return final_df, list(failed)

# ── node visual specs ──────────────────────────────────────────────────────────
#  Each entry: (fill_color, size, symbol, glow_opacity, legend_name)
_SPEC = {
    ("road",     0): ("#22c55e",  7, "circle", False, "Road — Survived"),
    ("road",     1): ("#ef4444", 10, "circle", True,  "Road — Failed"),
    ("hospital", 0): ("#38bdf8", 13, "square", False, "Hospital — Survived"),
    ("hospital", 1): ("#f97316", 17, "square", True,  "Hospital — Failed"),
    ("power",    0): ("#a78bfa", 14, "star",   False, "Power — Survived"),
    ("power",    1): ("#fde047", 18, "star",   True,  "Power — Failed"),
}

# Compute Chennai centre once from actual node positions
_all_lons = [v[0] for v in GEO_POS.values()]
_all_lats = [v[1] for v in GEO_POS.values()]
_CTR_LON  = (min(_all_lons) + max(_all_lons)) / 2
_CTR_LAT  = (min(_all_lats) + max(_all_lats)) / 2

# ── figure builder – real Chennai map overlay ─────────────────────────────────
def build_figure(pred_df=None, failed_ids=None, intensity=None, dark=True, show_map=False):
    mapbox_style = "carto-darkmatter" if dark else "carto-positron"
    fc    = "#e2e8f0" if dark else "#1e293b"
    bg    = "#0b1120" if dark else "#f1f5f9"
    lbg   = "rgba(15,23,42,0.9)"    if dark else "rgba(255,255,255,0.95)"
    lbc   = "rgba(148,163,184,0.3)" if dark else "rgba(100,116,139,0.3)"
    e_col = "rgba(180,180,210,0.22)" if dark else "rgba(80,100,140,0.22)"

    # Resolve failure map from whichever source is provided
    if pred_df is not None:
        pred_map = pred_df.set_index("node_id")["predicted_failure"].to_dict()
    elif failed_ids is not None:
        pred_map = {int(fid): 1 for fid in failed_ids}
    else:
        pred_map = {}

    # ── Edges ──
    e_x, e_y = [], []
    for u, v in GRAPH.edges():
        if u in GEO_POS and v in GEO_POS:
            x0, y0 = GEO_POS[u]; x1, y1 = GEO_POS[v]
            e_x += [x0, x1, None]
            e_y += [y0, y1, None]

    if show_map:
        traces = [go.Scattermapbox(lat=e_x, lon=e_y, mode="lines",
            line=dict(width=0.8, color=e_col), hoverinfo="none", showlegend=False)]
    else:
        traces = [go.Scatter(x=e_x, y=e_y, mode="lines",
            line=dict(width=0.5, color=e_col), hoverinfo="none", showlegend=False)]

    # ── Nodes grouped by type + failure status ──
    buckets = {k: {"x": [], "y": [], "ids": []} for k in _SPEC}
    for n, d in GRAPH.nodes(data=True):
        if n not in GEO_POS: continue
        nt  = d.get("node_type", "road")
        fl  = int(pred_map.get(n, 0))
        key = (nt, fl)
        if key in buckets:
            x, y = GEO_POS[n]
            buckets[key]["x"].append(x)
            buckets[key]["y"].append(y)
            buckets[key]["ids"].append(str(n))

    # survived first (bottom), failed on top
    for layer in [0, 1]:
        for (nt, fl), pts in buckets.items():
            if fl != layer or not pts["x"]: continue
            col, sz, sym, glow, lname = _SPEC[(nt, fl)]
            status = "Failed" if fl else "Survived"

            if show_map:
                if glow:
                    traces.append(go.Scattermapbox(
                        lat=pts["y"], lon=pts["x"], mode="markers",
                        marker=go.scattermapbox.Marker(size=sz+12, color=col, opacity=0.18),
                        hoverinfo="none", showlegend=False))
                traces.append(go.Scattermapbox(
                    lat=pts["y"], lon=pts["x"], mode="markers",
                    marker=go.scattermapbox.Marker(size=sz, color=col, symbol="circle", opacity=0.92),
                    name=lname, text=pts["ids"],
                    hovertemplate=(f"<b>Node %{{text}}</b><br>Type: {nt.capitalize()}<br>Status: {status}<extra></extra>")
                ))
            else:
                if glow:
                    traces.append(go.Scatter(
                        x=pts["x"], y=pts["y"], mode="markers",
                        marker=dict(size=sz+12, color=col, symbol=sym, opacity=0.18, line=dict(width=0)),
                        hoverinfo="none", showlegend=False))
                traces.append(go.Scatter(
                    x=pts["x"], y=pts["y"], mode="markers",
                    marker=dict(size=sz, color=col, symbol=sym, opacity=0.92,
                                line=dict(width=1.5 if fl else 0.8, color="rgba(255,255,255,0.3)")),
                    name=lname, text=pts["ids"],
                    hovertemplate=(f"<b>Node %{{text}}</b><br>Type: {nt.capitalize()}<br>Status: {status}<extra></extra>")
                ))

    title = (
        f"Chennai Infrastructure — Cascade Failure at Intensity {intensity}×"
        if intensity else
        "Chennai Infrastructure Network  (run simulation to see predictions)"
    )
    fig = go.Figure(data=traces)
    
    layout_args = dict(
        title=dict(text=title, font=dict(size=14, color=fc), x=0.5),
        showlegend=False,
        paper_bgcolor=bg,
        font=dict(color=fc, family="Inter, Arial, sans-serif"),
        margin=dict(l=0, r=0, t=46, b=0),
        uirevision="constant"
    )

    if show_map:
        layout_args["mapbox"] = dict(style=mapbox_style, center=dict(lat=_CTR_LAT, lon=_CTR_LON), zoom=11.2)
    else:
        layout_args["plot_bgcolor"] = bg
        layout_args["xaxis"] = dict(showgrid=False, zeroline=False, showticklabels=False)
        layout_args["yaxis"] = dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
        layout_args["margin"] = dict(l=10, r=10, t=50, b=10)

    fig.update_layout(**layout_args)
    return fig

# ── stat card ──────────────────────────────────────────────────────────────────
def stat_card(label, failed, total, color, dark=True):
    pct   = (failed/total*100) if total else 0
    card_bg = "#1e293b" if dark else "#f8fafc"
    lbl_c   = "#94a3b8"  if dark else "#64748b"
    tot_c   = "#64748b"  if dark else "#94a3b8"
    bar_bg  = "#0f172a"  if dark else "#e2e8f0"
    return html.Div([
        html.Div(label,style={"font-size":"0.72rem","font-weight":"600",
            "letter-spacing":"0.07em","text-transform":"uppercase",
            "color":lbl_c,"margin-bottom":"5px"}),
        html.Div([
            html.Span(f"{failed}",style={"font-size":"1.8rem","font-weight":"700","color":color}),
            html.Span(f" / {total}",style={"font-size":"1rem","color":tot_c}),
        ]),
        html.Div(style={"height":"5px","border-radius":"3px","background":bar_bg,"margin-top":"7px"},children=[
            html.Div(style={"height":"100%","border-radius":"3px",
                "width":f"{min(pct,100):.1f}%","background":color,"transition":"width 0.5s ease"}),
        ]),
        html.Div(f"{pct:.1f}% failure rate",style={"font-size":"0.7rem","color":tot_c,"margin-top":"3px"}),
    ],style={"background":card_bg,"border-radius":"10px","padding":"13px 15px",
             "margin-bottom":"9px","border-left":f"3px solid {color}"})

# ── theme palettes ─────────────────────────────────────────────────────────────
def theme(dark):
    if dark:
        return dict(
            page="#020617",sidebar="#0f172a",header="linear-gradient(135deg,#0f172a,#1e3a5f)",
            hborder="rgba(56,189,248,0.25)",sborder="rgba(56,189,248,0.12)",
            card="#1e293b",cborder="rgba(56,189,248,0.12)",
            text="#e2e8f0",muted="#64748b",accent="#38bdf8",
            badge_bg="#1e293b",scroll_track="#0f172a",scroll_thumb="#334155",
        )
    else:
        return dict(
            page="#f1f5f9",sidebar="#ffffff",header="linear-gradient(135deg,#e0f2fe,#ddd6fe)",
            hborder="rgba(14,165,233,0.3)",sborder="rgba(14,165,233,0.15)",
            card="#f8fafc",cborder="rgba(14,165,233,0.18)",
            text="#0f172a",muted="#475569",accent="#0284c7",
            badge_bg="#e0f2fe",scroll_track="#e2e8f0",scroll_thumb="#94a3b8",
        )

# ── app ────────────────────────────────────────────────────────────────────────
SLIDER_MARKS={i:{"label":f"{i}×","style":{"color":"#94a3b8","font-size":"0.78rem"}} for i in range(1,11)}

app=dash.Dash(__name__,title="CascadeWatch",
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}])

app.index_string="""<!DOCTYPE html>
<html>
<head>
  {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',sans-serif}
    .run-btn:hover{opacity:.88!important;transform:translateY(-1px);transition:all .2s}
    .run-btn:active{transform:translateY(0)}
    .theme-toggle{cursor:pointer;background:none;border:1px solid rgba(148,163,184,0.35);
      border-radius:20px;padding:5px 14px;font-size:.78rem;font-weight:600;
      display:flex;align-items:center;gap:6px;transition:all .2s}
    .theme-toggle:hover{border-color:rgba(56,189,248,.7)}
  </style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>"""

app.layout=html.Div(id="page-root",children=[
    dcc.Store(id="theme-store",data="dark"),
    dcc.Store(id="pred-store",data=None),   # persists last simulation results

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div(id="header",style={"padding":"16px 28px","display":"flex",
        "align-items":"center","gap":"14px"},children=[
        html.Div([
            html.H1("CascadeWatch",id="hdr-title",style={"font-size":"1.7rem","font-weight":"700",
                "background":"linear-gradient(90deg,#38bdf8,#818cf8)",
                "-webkit-background-clip":"text","-webkit-text-fill-color":"transparent",
                "background-clip":"text"}),
            html.P("Chennai Flood Infrastructure Cascade Failure Predictor",
                id="hdr-sub",style={"font-size":"0.8rem","margin-top":"2px"}),
        ]),
        html.Div(style={"margin-left":"auto","display":"flex","gap":"10px","align-items":"center"},children=[
            html.Span("820 nodes",id="badge-nodes",style={"border-radius":"20px","padding":"4px 12px",
                "font-size":"0.77rem","font-weight":"600","background":"#1e293b","color":"#38bdf8"}),
            html.Span("3 layers",id="badge-layers",style={"border-radius":"20px","padding":"4px 12px",
                "font-size":"0.77rem","font-weight":"600","background":"#1e293b","color":"#818cf8"}),
            html.Span("RF model",id="badge-model",style={"border-radius":"20px","padding":"4px 12px",
                "font-size":"0.77rem","font-weight":"600","background":"#1e293b","color":"#4ade80"}),
            # Toggles
            html.Button(id="map-btn",n_clicks=0,className="theme-toggle",
                style={"color":"#94a3b8"},children=["🗺️ Map: OFF"]),
            html.Button(id="theme-btn",n_clicks=0,className="theme-toggle",
                style={"color":"#94a3b8"},children=["🌙 Dark"]),
        ]),
    ]),

    # ── Body ──────────────────────────────────────────────────────────────────
    html.Div(id="body-wrap",style={"display":"flex","height":"calc(100vh - 70px)"},children=[

        # Sidebar
        html.Div(id="sidebar",style={"width":"300px","min-width":"300px",
            "padding":"22px 18px","overflow-y":"auto"},children=[

            # Intensity card
            html.Div(id="ctrl-card",style={"border-radius":"12px","padding":"18px 16px",
                "margin-bottom":"18px"},children=[
                html.Div("FLOOD INTENSITY",style={"font-size":"0.67rem","font-weight":"700",
                    "letter-spacing":"0.12em","color":"#38bdf8","margin-bottom":"5px"}),
                html.P("Dimensionless multiplier applied to each node's base flood depth.",
                    style={"font-size":"0.77rem","line-height":"1.4","margin-bottom":"16px","color":"#64748b"}),
                dcc.Slider(id="intensity-slider",min=1,max=10,step=0.01,value=5,
                    marks=SLIDER_MARKS,tooltip={"placement":"top","always_visible":True},
                    updatemode="drag"),
                html.Br(),
                # Reference legend
                html.Div(style={"display":"flex","flex-direction":"column","gap":"4px","margin-bottom":"18px"},children=[
                    html.Div(style={"display":"flex","gap":"8px","align-items":"center","font-size":"0.73rem"},children=[
                        html.Span("1×",style={"color":"#4ade80","font-weight":"700","width":"22px"}),
                        html.Span("baseline conditions",style={"color":"#64748b"})]),
                    html.Div(style={"display":"flex","gap":"8px","align-items":"center","font-size":"0.73rem"},children=[
                        html.Span("5×",style={"color":"#fb923c","font-weight":"700","width":"22px"}),
                        html.Span("severe flood",style={"color":"#64748b"})]),
                    html.Div(style={"display":"flex","gap":"8px","align-items":"center","font-size":"0.73rem"},children=[
                        html.Span("10×",style={"color":"#ef4444","font-weight":"700","width":"22px"}),
                        html.Span("extreme event",style={"color":"#64748b"})]),
                ]),
                html.Button("▶  Run Simulation",id="run-btn",n_clicks=0,className="run-btn",style={
                    "width":"100%","padding":"13px 0",
                    "background":"linear-gradient(135deg,#0ea5e9,#6366f1)",
                    "color":"#fff","border":"none","border-radius":"10px",
                    "font-size":"0.93rem","font-weight":"700","cursor":"pointer",
                    "box-shadow":"0 4px 18px rgba(56,189,248,0.25)"}),
            ]),

            # Stats panel
            html.Div(id="stats-panel",children=[
                html.Div("Run the simulation to see damage statistics.",
                    id="stats-placeholder",style={"font-size":"0.82rem","text-align":"center",
                    "padding":"28px 10px","color":"#475569"}),
            ]),



        ]),

        # Graph
        html.Div(id="graph-wrap",style={"flex":"1","overflow":"hidden","position":"relative"},children=[
            html.Div(id="map-legend", children=[]),
            dcc.Loading(id="loading-graph",type="circle",color="#38bdf8",children=[
                dcc.Graph(id="network-graph",figure=build_figure(dark=True),
                    config={"displayModeBar":True,"scrollZoom":True,
                        "modeBarButtonsToRemove":["toImage","sendDataToCloud"],"displaylogo":False},
                    style={"height":"calc(100vh - 70px)"}),
            ]),
        ]),
    ]),
])

# ── callbacks ──────────────────────────────────────────────────────────────────

# 1. Theme and Map toggle — flip store + restyle all containers
@app.callback(
    Output("theme-store","data"),
    Output("theme-btn","children"),
    Output("theme-btn","style"),
    Output("map-btn","children"),
    Output("map-btn","style"),
    Output("page-root","style"),
    Output("header","style"),
    Output("body-wrap","style"),
    Output("sidebar","style"),
    Output("ctrl-card","style"),
    Output("graph-wrap","style"),
    Output("hdr-sub","style"),
    Output("badge-nodes","style"),
    Output("badge-layers","style"),
    Output("badge-model","style"),
    Output("map-legend","style"),
    Output("map-legend","children"),
    Input("theme-btn","n_clicks"),
    Input("map-btn","n_clicks"),
    State("theme-store","data"),
    prevent_initial_call=False,   # fire on load so initial styles are consistent
)
def toggle_theme(t_clicks, m_clicks, current):
    ctx = dash.callback_context
    dark = (current == "dark")
    if ctx.triggered and ctx.triggered[0]['prop_id'] == "theme-btn.n_clicks":
        dark = not dark

    t    = theme(dark)
    # label shows the CURRENT theme name
    btn_label   = ["🌙 Dark"] if dark else ["☀️ Light"]
    btn_style   = {"color": t["muted"]}
    
    m_clicks = m_clicks or 0
    show_map = (m_clicks % 2 != 0)
    map_label = ["🗺️ Map: ON"] if show_map else ["🗺️ Map: OFF"]
    map_style = {"color": t["accent"]} if show_map else {"color": t["muted"]}

    page_style  = {"background":t["page"],"min-height":"100vh","color":t["text"]}
    hdr_style   = {"background":t["header"],"border-bottom":f"1px solid {t['hborder']}",
                   "padding":"16px 28px","display":"flex","align-items":"center","gap":"14px"}
    body_style  = {"display":"flex","height":"calc(100vh - 70px)"}
    side_style  = {"width":"300px","min-width":"300px","background":t["sidebar"],
                   "border-right":f"1px solid {t['sborder']}",
                   "padding":"22px 18px","overflow-y":"auto"}
    ctrl_style  = {"background":t["card"],"border-radius":"12px","padding":"18px 16px",
                   "margin-bottom":"18px","border":f"1px solid {t['cborder']}"}
    graph_style = {"flex":"1","overflow":"hidden","background":t["page"]}
    sub_style   = {"font-size":"0.8rem","margin-top":"2px","color":t["muted"]}

    base_badge  = {"border-radius":"20px","padding":"4px 12px",
                   "font-size":"0.77rem","font-weight":"600"}
    if dark:
        badge_nodes  = {**base_badge,"background":"#1e293b","color":"#38bdf8"}
        badge_layers = {**base_badge,"background":"#1e293b","color":"#818cf8"}
        badge_model  = {**base_badge,"background":"#1e293b","color":"#4ade80"}
    else:
        badge_nodes  = {**base_badge,"background":"#e0f2fe","color":"#0284c7"}
        badge_layers = {**base_badge,"background":"#ede9fe","color":"#6d28d9"}
        badge_model  = {**base_badge,"background":"#dcfce7","color":"#15803d"}

    leg_style = {
        "position":"absolute","top":"130px","right":"15px","z-index":"1000",
        "border-radius":"12px","padding":"14px 16px",
        "background": "rgba(30, 41, 59, 0.85)" if dark else "rgba(241, 245, 249, 0.85)",
        "backdrop-filter": "blur(8px)",
        "border": f"1px solid {t['cborder']}",
        "color": t['text'],
    }

    h_sym = "●" if show_map else "■"
    p_sym = "●" if show_map else "★"
    h_sz = "1rem" if show_map else "1.2rem"
    p_sz = "1rem" if show_map else "1.3rem"

    leg_children = [
        html.Div("MAP LEGEND",style={"font-size":"0.67rem","font-weight":"700",
            "letter-spacing":"0.12em","color":"#38bdf8","margin-bottom":"10px"}),
        html.Div(style={"display":"flex","flex-direction":"column","gap":"6px"},children=[
            html.Div(style={"display":"flex","align-items":"center","gap":"8px"},children=[
                html.Div("●",style={"font-size":"1rem","color":"#22c55e","line-height":"10px"}),
                html.Span("Road — Survived",style={"font-size":"0.75rem"}),
            ]),
            html.Div(style={"display":"flex","align-items":"center","gap":"8px"},children=[
                html.Div(h_sym,style={"font-size":h_sz,"color":"#38bdf8","line-height":"10px"}),
                html.Span("Hospital — Survived",style={"font-size":"0.75rem"}),
            ]),
            html.Div(style={"display":"flex","align-items":"center","gap":"8px"},children=[
                html.Div(p_sym,style={"font-size":p_sz,"color":"#a78bfa","line-height":"10px"}),
                html.Span("Power — Survived",style={"font-size":"0.75rem"}),
            ]),
            html.Div(style={"height":"4px"}),
            html.Div(style={"display":"flex","align-items":"center","gap":"8px"},children=[
                html.Div("●",style={"font-size":"1rem","color":"#ef4444","line-height":"10px",
                    "text-shadow":"0 0 5px rgba(239,68,68,0.7)"}),
                html.Span("Road — Failed",style={"font-size":"0.75rem"}),
            ]),
            html.Div(style={"display":"flex","align-items":"center","gap":"8px"},children=[
                html.Div(h_sym,style={"font-size":h_sz,"color":"#f97316","line-height":"10px",
                    "text-shadow":"0 0 5px rgba(249,115,22,0.7)"}),
                html.Span("Hospital — Failed",style={"font-size":"0.75rem"}),
            ]),
            html.Div(style={"display":"flex","align-items":"center","gap":"8px"},children=[
                html.Div(p_sym,style={"font-size":p_sz,"color":"#fde047","line-height":"10px",
                    "text-shadow":"0 0 5px rgba(253,224,71,0.7)"}),
                html.Span("Power — Failed",style={"font-size":"0.75rem"}),
            ]),
        ]),
    ]

    return (("dark" if dark else "light"), btn_label, btn_style, map_label, map_style,
            page_style, hdr_style, body_style, side_style,
            ctrl_style, graph_style, sub_style,
            badge_nodes, badge_layers, badge_model, leg_style, leg_children)


# helper — build stats children from raw numbers
def build_stats(intensity, total_failed, total_nodes, rf, rt, hf, ht, pf, pt, dark):
    pct = (total_failed/total_nodes*100) if total_nodes else 0
    sev_color,sev_label = (
        ("#22c55e","LOW") if pct<10 else
        ("#fbbf24","MODERATE") if pct<30 else
        ("#fb923c","SEVERE") if pct<55 else
        ("#ef4444","CRITICAL")
    )
    t = theme(dark)
    return [
        html.Div(style={"margin-bottom":"12px"},children=[
            html.Div(style={"display":"flex","justify-content":"space-between","align-items":"center"},children=[
                html.Div("DAMAGE SUMMARY",style={"font-size":"0.67rem","font-weight":"700",
                    "letter-spacing":"0.12em","color":t["accent"]}),
                html.Span(sev_label,style={"color":sev_color,
                    "border":f"1px solid {sev_color}","border-radius":"4px",
                    "padding":"2px 8px","font-size":"0.63rem","font-weight":"700"}),
            ]),
            html.P(f"Intensity {intensity}×  •  10-step cascade",
                style={"color":t["muted"],"font-size":"0.71rem","margin-top":"3px"}),
        ]),
        stat_card("Total Infrastructure Failures",total_failed,total_nodes,sev_color,dark),
        html.Div("BY NODE TYPE",style={"font-size":"0.63rem","font-weight":"700",
            "letter-spacing":"0.1em","color":t["muted"],"margin":"12px 0 7px"}),
        stat_card("● Roads",       rf, rt, "#f87171", dark),
        stat_card("■ Hospitals",   hf, ht, "#38bdf8", dark),
        stat_card("★ Power Nodes", pf, pt, "#fde047", dark),
    ]


# 2. Run simulation → store results + update graph
@app.callback(
    Output("network-graph","figure"),
    Output("pred-store","data"),
    Input("run-btn","n_clicks"),
    State("intensity-slider","value"),
    State("theme-store","data"),
    State("map-btn","n_clicks"),
    prevent_initial_call=True,
)
def simulate(n_clicks, intensity, theme_val, map_clicks):
    dark = (theme_val == "dark")
    map_clicks = map_clicks or 0
    show_map = (map_clicks % 2 != 0)
    print(f"Simulating intensity={intensity}× …")
    pred_df, _failed_list = run_prediction(intensity)
    fig     = build_figure(pred_df=pred_df, intensity=intensity, dark=dark, show_map=show_map)

    total_nodes  = len(pred_df)
    total_failed = int(pred_df["predicted_failure"].sum())
    failed_df    = pred_df[pred_df["predicted_failure"]==1].copy()
    sbt={}
    if len(failed_df)>0:
        failed_df["nts"] = LE.inverse_transform(failed_df["node_type_encoded"].astype(int))
        sbt = failed_df["nts"].value_counts().to_dict()

    store = {
        "intensity": round(float(intensity),2),
        "total_failed": total_failed, "total_nodes": total_nodes,
        "rf": sbt.get("road",0),     "rt": TOTAL_BY_TYPE.get("road",0),
        "hf": sbt.get("hospital",0), "ht": TOTAL_BY_TYPE.get("hospital",0),
        "pf": sbt.get("power",0),    "pt": TOTAL_BY_TYPE.get("power",0),
        "failed_ids": _failed_list,
    }
    print(f"Done → {total_failed}/{total_nodes} failures.")
    return fig, store


# 3. Stats panel — updates on new simulation OR theme change
@app.callback(
    Output("stats-panel","children"),
    Input("pred-store","data"),
    Input("theme-store","data"),
)
def update_stats(store, theme_val):
    if not store:
        return html.Div("Run the simulation to see damage statistics.",
            style={"font-size":"0.82rem","text-align":"center",
                   "padding":"28px 10px","color":"#475569"})
    dark = (theme_val == "dark")
    return build_stats(store["intensity"], store["total_failed"], store["total_nodes"],
                       store["rf"], store["rt"], store["hf"], store["ht"],
                       store["pf"], store["pt"], dark)


# 4. Re-theme graph on toggle (without re-running simulation)
@app.callback(
    Output("network-graph","figure",allow_duplicate=True),
    Input("theme-store","data"),
    Input("map-btn","n_clicks"),
    State("pred-store","data"),
    prevent_initial_call=True,
)
def retheme_graph(theme_val, map_clicks, store):
    dark = (theme_val == "dark")
    map_clicks = map_clicks or 0
    show_map = (map_clicks % 2 != 0)
    if store and store.get("failed_ids"):
        return build_figure(
            failed_ids=store["failed_ids"],
            intensity=store["intensity"],
            dark=dark,
            show_map=show_map
        )
    return build_figure(dark=dark, show_map=show_map)


if __name__=="__main__":
    app.run(debug=False, port=8050)
