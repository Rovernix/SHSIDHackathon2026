"""Native LaTeX backend for the English modeling report."""
from pathlib import Path
from html.parser import HTMLParser
import json
import shutil
import subprocess
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aidc.paths import RESULTS, REPORTS, FIGURES, QA

ROOT=Path(__file__).resolve().parents[1];OUT=REPORTS;FIG=FIGURES;RES=RESULTS
W=483.;story=[]


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def fmt(x,n=2):return f"{x:,.{n}f}"
def pct(s,b,field):return 100*(1-s[field]/b[field])


def escape(s):
    mapping={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$","#":r"\#","_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return "".join(mapping.get(c,c) for c in str(s))


class Markup(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=="br":self.parts.append(r"\\ ")
        elif tag=="link":self.parts.append(r"\href{"+dict(attrs)["href"]+"}{")
    def handle_endtag(self,tag):
        if tag=="link":self.parts.append("}")
    def handle_data(self,data):self.parts.append(escape(data))


def prose(s):
    parser=Markup();parser.feed(s);return "".join(parser.parts)


def text(s,style="body"):
    t=prose(s)
    if style=="title":t=r"{\raggedright\LARGE\bfseries "+t+r"\par}\vspace{1em}"
    elif style=="heading":t=r"\section*{"+t+"}"
    elif style=="sub":t=r"\subsection*{"+t+"}"
    elif style in ("small","caption"):t=r"{\small "+t+r"\par}"
    story.append(t+"\n\n")


def title(n,s):
    story.append("\\clearpage\n\\section*{"+str(n)+"  "+escape(s)+"}\n")


def sub(s):text(s,"sub")


def table(headers,rows,widths=None,size=8.5):
    n=len(headers);widths=widths or [W/n]*n
    scale=(W-8*(n-1))/sum(widths)
    columns="@{}"+"".join("p{"+f"{v*scale:.2f}"+"pt}" for v in widths)+"@{}"
    out=[r"\begin{center}\begingroup\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{1.2}",f"\\fontsize{{{size}}}{{{size*1.4:.2f}}}\\selectfont",
         r"\begin{tabular}{"+columns+"}",r"\toprule",
         " & ".join(r"\textbf{"+escape(x)+"}" for x in headers)+r" \\",r"\midrule"]
    for row in rows:out.append(" & ".join(escape(x) for x in row)+r" \\")
    out += [r"\bottomrule\end{tabular}\endgroup\end{center}"]
    story.append("\n".join(out)+"\n\n")


def picture(name,width=W,maxheight=None,caption=None):
    options=r"width=\linewidth"
    if maxheight:options+=",height="+str(maxheight)+"pt,keepaspectratio"
    story.append(r"\begin{center}\includegraphics["+options+"]{figures/"+name+r".png}\end{center}"+"\n")
    if caption:text(caption,"caption")


FORMULAS={
"starts":r"\mathcal F_j=\{s\in\mathbb Z:r_j\le s\le\min(d_j-L_j,r_j+W_j)\},\qquad \sum_{s\in\mathcal F_j}z_{js}=1.",
"activity":r"a_{jt}=\sum_{\substack{s\in\mathcal F_j\\s\le t<s+L_j}}z_{js},\qquad\sum_jg_ja_{jt}+n_t\le N_t.",
"dag":r"\sum_{s\in\mathcal F_k}s z_{ks}\ge\sum_{s\in\mathcal F_j}(s+L_j)z_{js},\qquad(j,k)\in\mathcal E_{\rm dep}.",
"facility":r"P_t^{\rm IT}=P_{\rm base}+q_{\rm inf}n_t+\sum_jp_ja_{jt},\qquad P_t^{\rm fac}=\kappa_tP_t^{\rm IT}.",
"balance":r"G_t+R_t+\sum_bu_{bt}=P_t^{\rm fac}+\sum_bc_{bt}.",
"limits":r"\begin{aligned}0\le G_t\le\bar G_t,\quad 0\le R_t\le\bar R_t,\quad P_t^{\rm fac}\le\bar P^{\rm fac},\\ |G_t-G_{t-1}|\le\Gamma\quad(t\ge1).\end{aligned}",
"soc":r"E_{b,t+1}=E_{bt}+\eta_b^c c_{bt}\Delta t-\frac{u_{bt}\Delta t}{\eta_b^d}.",
"battery_limits":r"E_b^{\min}+E_b^{\rm res}\le E_{bt}\le E_b^{\max},\qquad E_{b0}=E_{bT}=E_b^{\rm init}.",
"battery_modes":r"\begin{aligned}0&\le c_{bt}\le(\bar c_b-P_b^{\rm res})v_{bt},\\0&\le u_{bt}\le(\bar u_b-P_b^{\rm res})(1-v_{bt}),\qquad v_{bt}\in\{0,1\}.\end{aligned}",
"cost":r"C=\Delta t\sum_t(\pi_tG_t+\pi_RR_t)+\lambda Q+\Delta t\sum_{b,t}k_b(c_{bt}+u_{bt}),\quad Q\ge G_t.",
"carbon":r"M=\Delta t\sum_t\xi_tG_t,\qquad \min C\quad\text{subject to}\quad M\le0.97M_{\rm FIFO}.",
"gap":r"L\le C^*\le U,\qquad\operatorname{gap}=\frac{U-L}{|U|}."}


def eq(name):story.append("\\begin{equation}\n"+FORMULAS[name]+"\n\\end{equation}\n")


def finish_report(source_only=False):
    folder=OUT/"latex";folder.mkdir(parents=True,exist_ok=True)
    (folder/"figures").mkdir(exist_ok=True)
    used_figures=set(re.findall(r"\{figures/([^}]+\.png)\}", "".join(story)))
    for name in used_figures:
        shutil.copy2(FIG/name,folder/"figures"/name)
    preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[textwidth=483pt,top=43pt,bottom=46pt]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\hypersetup{pdftitle={AI Data Center Power and Compute Co Scheduling},pdfauthor={HackSHSID 2026 project}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\footskip}{20pt}
\setlength{\emergencystretch}{2em}
\renewcommand{\baselinestretch}{1.08}
\pagestyle{plain}
\begin{document}
'''
    tex=folder/"AIDC_Modeling_Report.tex"
    tex.write_text(preamble+"".join(story)+"\n\\end{document}\n",encoding="utf-8")
    if not source_only:
        engine=shutil.which("pdflatex")
        if not engine:raise RuntimeError("LaTeX source generated; install TeX Live or MiKTeX to compile")
        version=subprocess.run([engine,"--version"],capture_output=True).stdout
        engine_options=["--disable-installer"] if b"MiKTeX" in version else []
        for pass_number in range(2):
            result=subprocess.run([engine,*engine_options,"-interaction=nonstopmode","-halt-on-error","-file-line-error",tex.name],cwd=folder,capture_output=True)
            (folder/f"compile-pass-{pass_number+1}.txt").write_bytes(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(f"LaTeX failed; inspect {folder / ('compile-pass-'+str(pass_number+1)+'.txt')}")
        shutil.copy2(folder/"AIDC_Modeling_Report.pdf",OUT/"AIDC_Modeling_Report.pdf")
    print(tex)
