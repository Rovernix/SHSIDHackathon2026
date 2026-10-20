"""Select a reserve on two public days, then evaluate distinct days and a constructed shock."""
from pathlib import Path
import copy
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aidc.data import canonical_hash,save_json
from aidc.datasets import build_scenario
from aidc.paths import RESULTS
from aidc.rolling import replay


def main():
    path=RESULTS/"online_study.json"
    records=json.loads(path.read_text(encoding="utf-8")).get("runs",[]) if path.exists() else []
    def run(day,margin,stage,shock=False):
        d=build_scenario(day)
        if shock:
            for t in range(40,56):d["renewable_kw"][t]*=.15
            for t in range(44,52):d["inference_gpus"][t]+=48
            d["constructed_stress_scenario"]="unannounced_solar_loss_and_48_gpu_burst"
        fingerprint=canonical_hash(d)
        for record in records:
            if (record["date"],record["margin"],record["stage"],record["shock"],record["input_hash"])==(day,margin,stage,shock,fingerprint):
                return record["result"]
        r=replay(d,margin=margin)
        records.append(dict(date=day,margin=margin,stage=stage,shock=shock,input_hash=fingerprint,input=d,result=r))
        print(stage,day,margin,r["feasible"],r.get("cost_cny"),r.get("failed_at"),flush=True)
        save_json(path,dict(runs=records))
        return r
    margins=[0,2,4,8]
    scores={}
    for margin in margins:
        rs=[run(day,margin,"selection") for day in ["2020-06-02","2020-06-03"]]
        scores[margin]=(sum(not r["feasible"] for r in rs),sum(r.get("cost_cny",1e9) for r in rs))
    selected=min(scores,key=scores.get)
    for day in ["2020-06-04","2020-06-05"]:
        for margin in sorted({4,selected}):run(day,margin,"new_day_evaluation")
    main=run("2020-06-01",selected,"main_input_evaluation")
    if main["feasible"]:save_json(RESULTS/"rolling_optimized.json",main)
    for margin in sorted({4,selected}):run("2020-06-01",margin,"forecast_shock",True)
    save_json(path,dict(selection_dates=["2020-06-02","2020-06-03"],new_evaluation_dates=["2020-06-04","2020-06-05"],
        candidates=margins,selected_margin=selected,selection_scores={str(k):list(v) for k,v in scores.items()},
        selection_rule="Minimize failures first, then summed cost on selection dates only",runs=records,
        limitations="Related public workload days from one virtual cluster; heuristic selection is not a generalization guarantee. The combined shock is constructed, not a historical incident."))


if __name__=="__main__":main()
