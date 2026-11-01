"""LaTeX tables for the paper, generated from the evaluation outputs (best bold, second-best underlined).

  python proj/scripts/paper_tables.py [--out paper/tables]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfeat import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "runs" / "eval"
AUC_TH = {"hpatches": [1, 3, 5], "megadepth": [5, 10, 20]}
TH = {("megadepth", "orb"): 1.0, ("megadepth", "brisk"): 1.0, ("hpatches", "orb"): 2.0, ("hpatches", "brisk"): 4.0}
K = {"megadepth": 2048, "hpatches": 1024}
ROWS = [("vanilla", "Vanilla OpenCV"), ("pool_resp", "Pool, top-$K$ response"), ("pool_grid", "Pool, grid"),
        ("sup", "Supervised"), ("rl", "\\ours (5 seeds)")]
SEEDS = range(5)
TIMING = json.loads((ROOT / "paper" / "figures" / "timing.json").read_text())


def aucs(df, v, ds, th):
    return 100 * np.array(metrics.error_auc(df[df.variant == v][f"err@{th}"].values, AUC_TH[ds]))


def seed_aucs(ds, m, th, s, v="rl"):
    """AUCs of one RL seed: seed 0 is in the main eval, 1-2 in _seeds, 3-4 in _seeds34."""
    if s == 0:
        return aucs(pd.read_csv(EVAL / f"{ds}_{m}_k{K[ds]}_pairs.csv"), v, ds, th)
    tag = "_seeds" if s <= 2 else "_seeds34"
    return aucs(pd.read_csv(EVAL / f"{ds}_{m}_k{K[ds]}{tag}_pairs.csv"), v.replace("rl", f"rl_s{s}"), ds, th)


def ranks(col):
    """Return {row: 'B'|'U'|''}: best, and second-best when it is not also the worst (ties share)."""
    vals = sorted(set(np.round(list(col.values()), 1)), reverse=True)
    out = {}
    for r, x in col.items():
        x = round(x, 1)
        out[r] = "B" if x == vals[0] else ("U" if len(vals) > 2 and x == vals[1] else "")
    return out


def fmt(x, mark, std=None):
    s = f"{x:.1f}"
    if std is not None:
        s += rf"\,$\pm$\,{std:.1f}"
    return {"B": f"\\B{{{s}}}", "U": f"\\U{{{s}}}", "": s}[mark]


def main_table():
    lines = []
    for ds in ("megadepth", "hpatches"):
        res, std = {}, {}
        for m in ("orb", "brisk"):
            df = pd.read_csv(EVAL / f"{ds}_{m}_k{K[ds]}_pairs.csv")
            th = TH[(ds, m)]
            for v, _ in ROWS:
                if v == "rl":
                    A = np.stack([seed_aucs(ds, m, th, s) for s in SEEDS])
                    a = A.mean(0)
                    std[m] = A.mean(1).std(ddof=1)
                else:
                    a = aucs(df, v, ds, th)
                res[(m, v)] = list(a) + [float(np.mean(a))]
        marks = {}
        for m in ("orb", "brisk"):
            for j in range(4):
                for v, mk in ranks({v: res[(m, v)][j] for v, _ in ROWS}).items():
                    marks[(m, v, j)] = mk
        unit = "$^\\circ$" if ds == "megadepth" else "\\,px"
        head = " & ".join(f"@{t}{unit}" for t in AUC_TH[ds]) + " & Mean"
        title = ("MegaDepth-1500, relative pose ($K{=}2048$)" if ds == "megadepth"
                 else "HPatches, homography ($K{=}1024$)")
        lines += [("\\midrule" if ds == "hpatches" else ""),
                  f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}} \\\\",
                  f"& {{{head.split(' & ')[0]}}} & {{{head.split(' & ')[1]}}} & {{{head.split(' & ')[2]}}} & {{Mean}} "
                  f"& {{{head.split(' & ')[0]}}} & {{{head.split(' & ')[1]}}} & {{{head.split(' & ')[2]}}} & {{Mean}} \\\\",
                  "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}"]
        for v, name in ROWS:
            cells = []
            for m in ("orb", "brisk"):
                for j in range(4):
                    cells.append(fmt(res[(m, v)][j], marks[(m, v, j)], std[m] if (v == "rl" and j == 3) else None))
            row = f"{name} & " + " & ".join(cells) + " \\\\"
            lines.append(row)
    return "\n".join(l for l in lines if l)


def matched_table():
    A = {m: json.loads((ROOT / "runs" / "report" / f"paper_analysis_{m}.json").read_text())["summary"]
         for m in ("orb", "brisk")}
    rows_v = [("vanilla", "Vanilla"), ("pool_grid", "Pool, grid"), ("rl", "\\ours")]
    lines = []
    for m in ("orb", "brisk"):
        df = pd.read_csv(EVAL / f"megadepth_{m}_k2048_pairs.csv")
        t = TIMING["megadepth"][m]
        rt = {"vanilla": t["vanilla"], "pool_grid": t["pool"], "rl": t["rl_total"]}
        vals = {}
        for v, _ in rows_v:
            sub = df[df.variant == v]
            vals[v] = [100 * sub["mma@1"].mean(), 100 * sub["mma@3"].mean(), 100 * sub["mma@5"].mean(),
                       100 * (sub["inl@1.0"] / sub.nm.clip(lower=1)).mean(), 100 * A[m][v]["inl_cov"], rt[v]]
        marks = {}
        for j in range(6):
            col = {v: (-vals[v][j] if j == 5 else vals[v][j]) for v, _ in rows_v}
            for v, mk in ranks(col).items():
                marks[(v, j)] = mk
        for i, (v, name) in enumerate(rows_v):
            cells = [fmt(vals[v][j], marks[(v, j)]) for j in range(5)]
            r = f"{vals[v][5]:.0f}"
            cells.append({"B": f"\\B{{{r}}}", "U": f"\\U{{{r}}}", "": r}[marks[(v, 5)]])
            lead = f"\\multirow{{3}}{{*}}{{{m.upper()}}}" if i == 0 else ""
            lines.append(f"{lead} & {name} & " + " & ".join(cells) + " \\\\")
        if m == "orb":
            lines.append("\\midrule")
    return "\n".join(lines)


def seeds_table():
    lines = []
    for m in ("orb", "brisk"):
        for i, s in enumerate(SEEDS):
            cells = []
            for ds in ("megadepth", "hpatches"):
                th = TH[(ds, m)]
                a = seed_aucs(ds, m, th, s)
                cells += [f"{x:.1f}" for x in a]
            lead = f"\\multirow{{{len(SEEDS)}}}{{*}}{{{m.upper()}}}" if i == 0 else ""
            lines.append(f"{lead} & seed {s} & " + " & ".join(cells) + f" & {{VAL{m}{s}}} \\\\")
        if m == "orb":
            lines.append("\\midrule")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "tables"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "main_rows.tex").write_text(main_table() + "\n", encoding="utf-8")
    (out / "matched_rows.tex").write_text(matched_table() + "\n", encoding="utf-8")
    val = {("orb", 0): 35.1, ("orb", 1): 34.3, ("orb", 2): 34.5, ("brisk", 0): 59.5, ("brisk", 1): 60.0, ("brisk", 2): 58.3,
           ("orb", 3): 36.3, ("orb", 4): 33.6, ("brisk", 3): 60.3, ("brisk", 4): 59.7}
    st = seeds_table()
    for (m, s), v in val.items():   # best validation mean AUC per seed, from runs/logs/train_<m>_rl_s<s>.log
        st = st.replace(f"{{VAL{m}{s}}}", f"{v:.1f}")
    (out / "seeds_rows.tex").write_text(st + "\n", encoding="utf-8")
    for f in sorted(out.glob("*.tex")):
        print(f"--- {f.name}\n{f.read_text()}")
