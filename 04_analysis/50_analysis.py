"""Node/street analysis for the TiGIS revision.

usage: python 50_analysis.py <ecw_v2.csv> <corridor_nodes.csv> <out_dir>
Produces: nodes_R1.csv, streets_R1.csv, tables (csv), values.tex (macros for the manuscript), text summary.
Key design points (reviewer 2): nodes are nested in streets -> mixed model with street random intercept, ICC,
street-level permutation test, Moran's I; no node-level Kruskal-Wallis as inferential evidence.
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
ecw = pd.read_csv(sys.argv[1]); cor = pd.read_csv(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(exist_ok=True, parents=True)
cor["node"] = np.where(cor.src == "SCS", "scs_" + cor.id.astype(str).str.replace(r"\.0$", "", regex=True), cor.id)
COVER_MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5   # a node-side needs >= 50 % of its 20 m window observed (optional 4th arg: sensitivity)


def street_class(r):
    if r.w_corridor - r.w_carriage < 1.5:
        return "Shared lane"
    if r.yoltip == "BULVAR":
        return "Boulevard"
    if r.yoltip == "CADDE":
        return "Collector avenue"
    return "Wide local street" if r.w_corridor >= 10 else "Narrow local street"


def moran_i(x, xy, k=8, nperm=999, seed=1):
    n = len(x)
    if n < 20:
        return np.nan, np.nan
    tree = cKDTree(xy); _, idx = tree.query(xy, k=min(k + 1, n)); idx = idx[:, 1:]
    z = x - x.mean()
    def mi(z):
        return (n / (idx.size)) * (z[:, None] * z[idx]).sum() / (z ** 2).sum()
    obs = mi(z); rng = np.random.default_rng(seed)
    perm = np.array([mi(rng.permutation(z)) for _ in range(nperm)])
    return obs, (1 + (perm >= obs).sum()) / (nperm + 1)


def main():
    e = ecw[(ecw.coverage >= COVER_MIN) & ecw.ecw_min.notna()].copy()
    # node level: best continuous path over the observed sides, width-weighted phantom capacity
    e["wnom"] = e.zone_w.clip(upper=3.0)                       # nominal width: pedestrian zone capped at a 3.0 m design walkway (Eq. 5)
    e["cap_lost"] = 1 - e.ecw_min.clip(upper=e.wnom) / e.wnom
    g = e.groupby("node")
    nodes = pd.DataFrame(dict(
        n_sides=g.size(), ecw_best=g.ecw_min.max(), ecw_mean_sides=g.ecw_min.mean(), ecw_avg=g.ecw_avg.mean(),
        ecw_p05=g.ecw_min_mc_p05.min(), ecw_p95=g.ecw_min_mc_p95.max(),
        zone_w=g.zone_w.sum(), pc=g.apply(lambda x: np.average(x.cap_lost, weights=x.wnom)),
        ecw_persons=g.ecw_min_with_persons.max(), person_area=g.a_person.sum(),
        veh_area=g[["a_car", "a_truck", "a_motorcycle"]].sum().sum(1), furn_area=g[["a_chair", "a_table", "a_potted_plant"]].sum().sum(1),
        infra_area=g[["a_trash_can", "a_dumpster", "a_utility_pole", "a_street_light", "a_utility_box", "a_electrical_cabinet"]].sum().sum(1),
        coverage=g.coverage.mean(), shared=g.shared.max(), src=g.src.first())).reset_index()
    nodes = nodes.merge(cor[["node", "street_id", "road_name", "yoltip", "lanes", "w_corridor", "w_carriage", "floors_left", "floors_right",
                             "x_node", "y_node", "hex_id"]], on="node", how="left")
    nodes["class"] = nodes.apply(street_class, axis=1)
    nodes["hw"] = ((nodes.floors_left.fillna(0) + nodes.floors_right.fillna(0)) / 2 * 3.0) / nodes.w_corridor
    nodes["tier"] = pd.cut(nodes.ecw_best, [-1e-9, 0.9, 1.5, 1e9], labels=["Impassable (<0.9 m)", "Restricted (0.9-1.5 m)", "Accessible (>=1.5 m)"], right=False)
    nodes["ecw_unc"] = (nodes.ecw_p95 - nodes.ecw_p05)
    nodes.to_csv(out / "nodes_R1.csv", index=False)

    V = {}
    V["nNodes"] = len(nodes); V["nStreets"] = nodes.street_id.nunique(); V["nSides"] = len(e)
    V["shareImpassable"] = (nodes.ecw_best < 0.9).mean() * 100
    V["shareRestricted"] = ((nodes.ecw_best >= 0.9) & (nodes.ecw_best < 1.5)).mean() * 100
    V["shareAccessible"] = (nodes.ecw_best >= 1.5).mean() * 100
    V["meanPC"] = nodes.pc.mean() * 100; V["medianECW"] = nodes.ecw_best.median()
    V["meanUncWidth"] = nodes.ecw_unc.mean()

    # street level
    s = nodes.groupby("street_id").agg(n=("node", "size"), ecw=("ecw_best", "mean"), pc=("pc", "mean"), cls=("class", lambda x: x.mode().iat[0]),
                                        w=("w_corridor", "mean")).reset_index()
    s.to_csv(out / "streets_R1.csv", index=False)
    ml = nodes.dropna(subset=["ecw_best", "street_id"])
    ml = ml[ml.groupby("street_id").node.transform("size") >= 2]
    try:
        m0 = smf.mixedlm("ecw_best ~ 1", ml, groups=ml.street_id).fit(reml=True)
        vs, ve = float(m0.cov_re.iloc[0, 0]), float(m0.scale)
        V["ICC"] = vs / (vs + ve); V["nMLstreets"] = ml.street_id.nunique(); V["nMLnodes"] = len(ml)
        m1 = smf.mixedlm("ecw_best ~ C(Q('class'))", ml, groups=ml.street_id).fit(reml=True)
        vs1, ve1 = float(m1.cov_re.iloc[0, 0]), float(m1.scale)
        fx = np.var(m1.model.exog @ m1.fe_params.values)
        V["Rmarg"] = fx / (fx + vs1 + ve1); V["Rcond"] = (fx + vs1) / (fx + vs1 + ve1)
        m1.summary().tables[1].to_csv(out / "mixed_model_class.csv")
    except Exception as ex:
        print("mixed model failed:", ex)
    # street-level permutation test (class labels permuted across streets)
    sl = s[s.n >= 2]
    if sl.cls.nunique() > 1:
        F0 = stats.f_oneway(*[g.ecw.values for _, g in sl.groupby("cls") if len(g) > 1])[0]
        rng = np.random.default_rng(7); cnt = 0
        for _ in range(4999):
            p = sl.assign(cls=rng.permutation(sl.cls.values))
            cnt += stats.f_oneway(*[g.ecw.values for _, g in p.groupby("cls") if len(g) > 1])[0] >= F0
        V["permP"] = (cnt + 1) / 5000; V["permF"] = F0
    kw = stats.kruskal(*[g.ecw_best.values for _, g in nodes.groupby("class") if len(g) > 1])
    V["kwNodeH"] = kw.statistic; V["kwNodeP"] = kw.pvalue
    mi, mp = moran_i(nodes.ecw_best.values, nodes[["x_node", "y_node"]].values)
    V["MoranI"] = mi; V["MoranP"] = mp
    # tier by class
    pd.crosstab(nodes["class"], nodes.tier, normalize="index").mul(100).round(1).to_csv(out / "tier_by_class.csv")
    nodes.groupby("class").agg(n=("node", "size"), ecw_mean=("ecw_best", "mean"), ecw_med=("ecw_best", "median"), pc=("pc", "mean"),
                                veh=("veh_area", "mean"), furn=("furn_area", "mean"), infra=("infra_area", "mean"), person=("person_area", "mean")).round(2).to_csv(out / "class_summary.csv")
    tot = nodes[["veh_area", "furn_area", "infra_area"]].sum(); V["shareVehArea"] = tot.veh_area / tot.sum() * 100
    with open(out / "values.tex", "w", encoding="utf8") as f:
        for k, v in V.items():
            d = 1 if (k.startswith("share") or k in ("meanPC", "permF", "kwNodeH")) else 3 if k == "MoranP" else 2   # decimals as printed in the text
            f.write("\\newcommand{\\%s}{%s}\n" % (k, (("%.*f" % (d, v)).replace("-", "$-$")) if isinstance(v, float) else v))
    print(pd.Series(V).round(3).to_string())


main()
