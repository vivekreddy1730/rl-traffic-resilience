# IEEE Access Project Report Build Instructions

## Deliverables Summary
1. **`report/RL_Traffic_Resilience_Report.pdf`**: The compiled 12-page IEEE Access project report PDF.
2. **`report/main.tex`**: The complete LaTeX document source with all 13 required sections, Algorithms 1 & 2, tables, and mathematics.
3. **`report/references.bib`**: BibTeX database with 21 verified scholarly citations.
4. **`report/report_overleaf.zip`**: Self-contained zip archive ready for direct upload to Overleaf.
5. **`report/figures_src/generate_figures.py`**: Python script generating all 8 vector PDF figures from simulation logs.
6. **`report/check_numbers.py`**: Automated audit suite asserting numerical accuracy, page count bounds, and governance rules.

---

## 1. Compiling on Overleaf
1. Log into [Overleaf](https://www.overleaf.com/).
2. Click **New Project** $\to$ **Upload Project**.
3. Upload `report/report_overleaf.zip`.
4. Ensure the compiler is set to **pdfLaTeX** (default in Overleaf).
5. Click **Recompile**. The complete 12-page report will compile with zero errors and zero missing references.

---

## 2. Compiling Locally (Windows / Linux / macOS)

### Prerequisites
- TeX distribution: MiKTeX or TeX Live with `pdflatex` and `bibtex`.
- Python 3.11 with `matplotlib`, `numpy`, `pandas`, and `pymupdf`.

### Compilation Steps
From the project root:
```powershell
# 1. (Optional) Re-generate all vector PDF figures
python report/figures_src/generate_figures.py

# 2. Add MiKTeX binaries to PATH (if using user-level MiKTeX on Windows)
$env:PATH = "C:\Users\palla\AppData\Local\Programs\MiKTeX\miktex\bin\x64;" + $env:PATH

# 3. Enter report directory and run LaTeX toolchain
cd report
pdflatex --disable-installer -interaction=nonstopmode main.tex
bibtex --disable-installer main
pdflatex --disable-installer -interaction=nonstopmode main.tex
pdflatex --disable-installer -interaction=nonstopmode main.tex

# 4. Copy to final PDF deliverable
Copy-Item main.pdf RL_Traffic_Resilience_Report.pdf -Force

# 5. Run automated compliance and numerical verification audit
cd ..
python report/check_numbers.py
```

---

## 3. Automated Verification Checks
Run the audit script to verify all constraints:
```powershell
python report/check_numbers.py
```
Checks performed:
- Strict page count bounds ($10 \le \text{pages} \le 14$; currently exactly 12 pages).
- Zero occurrences of forbidden terms (`bulletproof`, `100% solid`, `first`, `novel`, `cusum`, `statistically significant`, `team lead`, `team leader`).
- Exact empirical accuracy of all 15 key numbers (DQN, Max-Pressure variants, baselines, monitor thresholds, diagnosis accuracy, mitigation recoveries).
- Equal standing team role attribution (Ayyappa Reddy, Sriram, Vivek Reddy).
- AI assistant disclosure verification.
- Zero undefined references or missing bibliography citations.

