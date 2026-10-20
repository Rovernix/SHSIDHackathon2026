"""Render all report pages and contact sheets for visual QA (optional PyMuPDF/Pillow)."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pymupdf
from PIL import Image,ImageDraw
from aidc.paths import ROOT,REPORTS,QA


def main():
    output=ROOT/"tmp/pdf/public_report";output.mkdir(parents=True,exist_ok=True)
    path=REPORTS/"AIDC_Modeling_Report.pdf"
    with pymupdf.open(path) as document:
        count=len(document)
        for i,page in enumerate(document):
            page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3),alpha=False).save(output/f"page-{i+1:02}.png")
    for start in range(0,count,6):
        board=Image.new("RGB",(1000,2170),"#dbe3e9");draw=ImageDraw.Draw(board)
        for j in range(6):
            index=start+j
            if index>=count:break
            with Image.open(output/f"page-{index+1:02}.png") as source:
                thumbnail=source.convert("RGB");thumbnail.thumbnail((470,690))
            x,y=15+(j%2)*500,30+(j//2)*720
            board.paste(thumbnail,(x,y));draw.text((x,y-19),f"Page {index+1}",fill="black")
        board.save(output/f"contact-{start+1:02}.png")
    (QA/"screenshots").mkdir(exist_ok=True)
    with Image.open(output/"page-01.png") as first:first.save(QA/"screenshots/report-first-page.png")
    record=dict(pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rendered_pages=count,
                renderer="PyMuPDF, 1.3x; Pillow contact sheets",visual_review="Not recorded by the renderer; inspect PNGs separately")
    (QA/"render_qa.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(f"Rendered {count} pages; contact sheets: {output}")


if __name__=="__main__":main()
