"""Optional PDF structural QA; requires PyMuPDF in the QA environment."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import pymupdf
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aidc.paths import REPORTS, RESULTS, QA

ROOT=Path(__file__).resolve().parents[1]


def main():
    pdf=REPORTS/"AIDC_Modeling_Report.pdf"
    shutil.copy2(REPORTS/"latex"/pdf.name,pdf)
    d=pymupdf.open(pdf);pages=[]
    for i,page in enumerate(d):
        extracted=page.get_text();bad=[]
        for block in page.get_text("dict")["blocks"]:
            if block["type"]!=0:continue
            for line in block["lines"]:
                for span in line["spans"]:
                    x0,y0,x1,y1=span["bbox"]
                    if x0<20 or y0<10 or x1>page.rect.width-15 or y1>page.rect.height-15:
                        bad.append(span["text"])
        pages.append(dict(page=i+1,replacement_glyphs=extracted.count("\ufffd"),out_of_bounds=bad,
                          links=len(page.get_links()),text_characters=len(extracted)))
    log=(REPORTS/"latex"/"AIDC_Modeling_Report.log").read_text(encoding="utf-8",errors="replace")
    text="\n".join(p.get_text() for p in d)
    headline=json.loads((RESULTS/"joint_green.json").read_text(encoding="utf-8"))
    tests=json.loads((QA/"verification.json").read_text(encoding="utf-8"))["unit_tests"]["tests_run"]
    required=[f"{headline['cost_cny']:,.2f}",f"{headline['carbon_kg']:,.2f}",f"{tests} unit tests","Helios","OPSD"]
    qa=dict(page_count=len(d),native_latex=True,
            scalable_fonts_only=all(f[2]=="Type1" for page in d for f in page.get_fonts()),
            overfull_boxes=re.findall(r"Overfull[^\n]+",log),
            latex_errors=[x for x in log.splitlines() if x.startswith("!")],pages=pages,
            required_results_present=all(x in text for x in required),
            supplemental_citation_only="IMMC" not in text.split("References\n",1)[0],
            tex_sha256=hashlib.sha256((REPORTS/"latex/AIDC_Modeling_Report.tex").read_bytes()).hexdigest(),
            sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    qa["passed"]=len(d)==7 and qa["scalable_fonts_only"] and qa["required_results_present"] and qa["supplemental_citation_only"] and not qa["overfull_boxes"] and not qa["latex_errors"] and all(not p["replacement_glyphs"] and not p["out_of_bounds"] for p in pages)
    (QA/"pdf_structure_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(f"PDF QA: {qa['passed']}; pages: {len(d)}; links: {sum(p['links'] for p in pages)}")
    if not qa["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
