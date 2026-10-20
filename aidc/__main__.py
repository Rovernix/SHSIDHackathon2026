import argparse
import json
from pathlib import Path
from .data import generate, save_json
from .datasets import build_scenario
from .paths import RESULTS
from .model import solve
from .audit import audit
from .experiments import benchmark


def main():
    parser=argparse.ArgumentParser(description="HackSHSID AIDC reproducible co-scheduling")
    sub=parser.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate");g.add_argument("--date",default="2020-06-01");g.add_argument("--output",default="data/processed/main.json")
    g.add_argument("--synthetic-fixture",action="store_true",help="Artificial correctness/debugging example only")
    g.add_argument("--seed",type=int,default=2026,help="Used only with --synthetic-fixture")
    p=sub.add_parser("demo");p.add_argument("--input");p.add_argument("--date",default="2020-06-01")
    p.add_argument("--output",default=str(RESULTS));p.add_argument("--time-limit",type=float,default=20.)
    p.add_argument("--full",action="store_true");p.add_argument("--rolling",action="store_true")
    s=sub.add_parser("solve");s.add_argument("input");s.add_argument("--output",default="results/custom/solution.json")
    s.add_argument("--carbon-cap",type=float);s.add_argument("--time-limit",type=float,default=30.)
    a=sub.add_parser("audit");a.add_argument("input");a.add_argument("result")
    args=parser.parse_args()
    def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
    if args.command=="generate":save_json(args.output,generate(args.seed) if args.synthetic_fixture else build_scenario(args.date))
    elif args.command=="demo":
        benchmark(read(args.input) if args.input else build_scenario(args.date),args.output,args.time_limit,args.full,args.rolling)
    elif args.command=="solve":
        d=read(args.input);r=solve(d,carbon_cap=args.carbon_cap,time_limit=args.time_limit)
        r["audit"]=audit(d,r);save_json(args.output,r)
        print(json.dumps({k:v for k,v in r.items() if k in ["feasible","status","message","cost_cny","carbon_kg","mip_gap"]},indent=2))
        if not r["feasible"]:raise SystemExit(2)
    else:
        result=audit(read(args.input),read(args.result));print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result["passed"]:raise SystemExit(2)


if __name__=="__main__":main()
