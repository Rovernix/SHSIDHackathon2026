"""Build publication charts and a self-contained, offline interactive results artifact."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aidc.audit import audit
from aidc.paths import RESULTS, QA, FIGURES, DEMO, CONFIG

ROOT=Path(__file__).resolve().parents[1]
OUT=DEMO;FIG=FIGURES;RES=RESULTS
COLORS=["#667889","#d58b32","#5686b2","#7e6dab","#087e83"]
LABELS=["FIFO", "FIFO + storage", "Compute only", "Joint cost", "Joint green"]
KEYS=["fifo","fifo_storage","compute_only","joint_cost","joint_green"]


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))


def savefig(fig,name):
    fig.savefig(FIG/(name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(FIG/(name+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def build():
    FIG.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    d=read(RES/"input.json");summary=read(RES/"summary.json")
    sol={k:read(RES/f"{k}.json") for k in KEYS}
    for k,s in sol.items():
        if not audit(d,s)["passed"]:raise ValueError(f"Refuse to plot unaudited result: {k}")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.grid":True,"grid.alpha":.18,
                         "figure.constrained_layout.use":True})
    h=np.arange(d["T"])*d["dt_hours"]
    fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True)
    axs[0].step(h,d["price_cny_per_kwh"],where="post",color=COLORS[1],label="Grid price (CNY/kWh)")
    ax2=axs[0].twinx();ax2.plot(h,d["carbon_kg_per_kwh"],color=COLORS[4]);ax2.set_ylabel("kg CO2/kWh");axs[0].set_ylabel("CNY/kWh")
    axs[1].fill_between(h,d["renewable_kw"],step="post",color=COLORS[4],alpha=.35,label="Actual renewable")
    axs[1].plot(h,d["forecast_renewable_kw"],ls="--",color=COLORS[4],label="Previous-day persistence");axs[1].set_ylabel("kW");axs[1].legend(loc="upper right")
    axs[2].step(h,d["inference_gpus"],where="post",color=COLORS[2],label="Recorded short-job GPU reservation")
    axs[2].step(h,d["forecast_inference_gpus"],where="post",ls="--",color=COLORS[0],label="Forecast");axs[2].set_ylabel("GPUs");axs[2].set_xlabel("Hour of day");axs[2].legend(loc="upper left")
    axs[0].set_title(f"Public input records   |   {d['scenario_date_utc']} UTC energy day   |   Helios GPU cohort",loc="left")
    savefig(fig,"inputs")
    fig,axs=plt.subplots(1,3,figsize=(11,3.4))
    for ax,key,title in zip(axs,["cost_cny","carbon_kg","peak_kw"],["Daily operating proxy (CNY)","Operational emissions (kg CO2)","Peak grid import (kW)"]):
        values=[sol[k][key] for k in KEYS]
        ax.barh(LABELS,values,color=COLORS)
        ax.set_title(title,fontweight="bold",fontsize=9)
        ax.invert_yaxis();ax.set_xlim(0,max(values)*1.25)
        for i,v in enumerate(values):ax.text(v+max(values)*.02,i,f"{v:,.0f}",va="center",fontsize=8)
    savefig(fig,"comparison")
    green=sol["joint_green"]
    fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
    axs[0].step(h,sol["fifo"]["grid_kw"],where="post",color=COLORS[0],alpha=.7,label="FIFO grid")
    axs[0].step(h,green["grid_kw"],where="post",color=COLORS[4],label="Joint green grid")
    axs[0].plot(h,green["facility_kw"],color=COLORS[1],lw=1,label="Joint green facility")
    axs[0].set_ylabel("kW");axs[0].legend(ncol=3,fontsize=8)
    for b,c in zip(green["batteries"],[COLORS[2],COLORS[3]]):
        axs[1].step(h,np.asarray(b["discharge_kw"])-b["charge_kw"],where="post",label=b["name"]+" net discharge",color=c)
        axs[2].plot(np.arange(d["T"]+1)*d["dt_hours"],b["energy_kwh"],label=b["name"]+" energy",color=c)
    axs[1].axhline(0,color="black",lw=.5);axs[1].set_ylabel("kW (+ discharge)");axs[1].legend(fontsize=8,ncol=2)
    axs[2].set_ylabel("kWh");axs[2].set_xlabel("Hour of day");axs[2].legend(fontsize=8,ncol=2)
    axs[0].set_title("Dispatch and cyclic battery energy   |   joint green policy",loc="left")
    savefig(fig,"dispatch")
    fig,ax=plt.subplots(figsize=(10,7.5))
    for i,j in enumerate(d["jobs"]):
        ax.broken_barh([(j["release"]*.25,(j["deadline"]-j["release"])*.25)],(i-.37,.74),facecolor="#edf0f2")
        ax.broken_barh([(sol["fifo"]["starts"][j["id"]]*.25,j["duration"]*.25)],(i-.32,.18),facecolor=COLORS[1])
        ax.broken_barh([(green["starts"][j["id"]]*.25,j["duration"]*.25)],(i-.03,.31),facecolor=COLORS[4])
    ax.set_yticks(range(len(d["jobs"])),[j["id"] for j in d["jobs"]]);ax.invert_yaxis();ax.set_xlim(0,24);ax.set_xlabel("Hour of day")
    ax.set_title("Observed GPU jobs   |   grey: assumed window   orange: FIFO   teal: optimized",loc="left",fontsize=10)
    savefig(fig,"gantt")
    pts=[r for r in summary.get("pareto",[]) if r.get("feasible")]
    if pts:
        fig,ax=plt.subplots(figsize=(7,4.2))
        pts=sorted(pts,key=lambda x:x["carbon_kg"])
        ax.plot([p["carbon_kg"] for p in pts],[p["cost_cny"] for p in pts],"o-",c=COLORS[4],label="Feasible epsilon-budget samples")
        ax.scatter([green["carbon_kg"]],[green["cost_cny"]],marker="*",s=170,c=COLORS[1],label="Recommended 3% carbon budget",zorder=4)
        ax.scatter([sol["fifo"]["carbon_kg"]],[sol["fifo"]["cost_cny"]],s=45,c=COLORS[0],label="FIFO")
        ax.set_xlabel("Operational emissions (kg CO2)");ax.set_ylabel("Daily operating proxy (CNY)")
        ax.set_title("Cost and carbon trade-off   |   finite-gap feasible solutions",loc="left")
        ax.legend(fontsize=8);savefig(fig,"pareto")
    fig,ax=plt.subplots(figsize=(9,4))
    rows=[r for r in summary["stress"] if "cost_saving_percent" in r]
    ax.barh([r["scenario"] for r in rows],[r["cost_saving_percent"] for r in rows],color=COLORS[4]);ax.invert_yaxis()
    ax.set_xlabel("Cost savings relative to feasible same-scenario FIFO (%)")
    ax.set_title("Sensitivity   |   infeasible baselines excluded from savings",loc="left")
    savefig(fig,"sensitivity")
    fig,ax=plt.subplots(figsize=(7.5,3.4))
    rows=[r for r in summary["days"] if "cost_saving_percent" in r]
    ax.bar([str(r["date"]) for r in rows],[r["cost_saving_percent"] for r in rows],color=COLORS[2])
    ax.set_ylabel("Cost savings (%)");ax.set_xlabel("Public source date");ax.set_title("Additional public days where the 3% carbon policy is feasible",loc="left")
    savefig(fig,"additional_days")
    make_html(d,sol,summary)
    write_results(d,sol,summary)
    print("Built charts, report explorer and readable results.")




def make_html(d,sol,summary):
    payload=dict(input=d,solutions=sol,summary=summary)
    if (QA/"verification.json").exists():payload["testing"]=read(QA/"verification.json")
    rolling=RES/"rolling.json"
    if rolling.exists():payload["solutions"]["rolling"]=read(rolling)
    js=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")
    html=HTML.replace("__DATA__",js).replace("__STUDY_DATE__",d["scenario_date_utc"]).replace("__GPU_CAPACITY__",str(d["gpu_capacity"][0]))
    html=html.replace("__DEFAULT_RESERVE__",str(read(CONFIG)["online_reserve_gpus"])).replace("__SELECTED_RESERVE__",str(read(RES/"online_study.json")["selected_margin"]))
    (OUT/"Results_Explorer.html").write_text(html,encoding="utf-8")


def write_results(d,sol,b):
    from tools.build_docs import results_markdown
    (ROOT/"docs/results.md").write_text(results_markdown(d,sol,b).strip()+"\n",encoding="utf-8")


HTML=(ROOT/'templates/results_explorer.html').read_text(encoding='utf-8')


if __name__=="__main__":build()
