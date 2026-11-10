# IEEE Access LaTeX Template Notes & Engineering Constraints

## 1. Overview
The IEEE Access template uses the specialized document class `ieeeaccess.cls` which extends and customizes `IEEEtran.cls` for open-access journal publication. The document employs a two-column format with specific title banner, header/footer styling, and biography blocks.

## 2. Mandatory Structural Macros & Hooks
- `\documentclass{ieeeaccess}`
- `\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}`
- `\doi{10.1109/ACCESS.2026.xxxxxxx}`
- `\title{...}` (Uppercase and lowercase, avoid excessive formulas)
- `\author{\uppercase{First M. Author}\authorrefmark{1}, \uppercase{Second Author}\authorrefmark{2}, and \uppercase{Third Author}\authorrefmark{3}}`
- `\address[1]{Department of Artificial Intelligence, School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India (e-mail: ...)}`
- `\tfootnote{This work was completed as part of the B.Tech Artificial Intelligence curriculum at Amrita Vishwa Vidyapeetham, Coimbatore. Mentored by [GUIDE NAME].}`
- `\markboth{Reddy \headeretal: Resilient Reinforcement Learning for Traffic Signal Control}{Reddy \headeretal: Resilient Reinforcement Learning for Traffic Signal Control}`
- `\corresp{Corresponding author: Ayyappa Reddy (e-mail: ayyappa@example.com).}`
- `\begin{abstract} ... \end{abstract}` (150--250 words, single paragraph, self-contained, no citations, no math equations)
- `\begin{keywords} ... \end{keywords}` (Alphabetical, comma-separated keywords)
- `\titlepgskip=-15pt` and `\maketitle`
- `\EOD` (Placed immediately after the final biography and before `\end{document}` to balance columns and close Access layout)

## 3. Dimensions & Formatting Rules
- **Column widths**:
  - Single column: approximately 3.5 in (88.9 mm / 252 pt). All single-column figures and tables must fit within 3.5 in.
  - Double column span (`figure*`, `table*`): 7.16 in (181.8 mm / 516 pt).
- **Package Incompatibilities**:
  - `ieeeaccess.cls` defines its own caption and font sizing macros. Do NOT load `caption` or `subcaption` packages as they break Access table and figure caption alignment.
  - Do NOT load `algorithm2e` if using `algorithmic` or `algorithm`. Use `\usepackage{algorithm}` and `\usepackage{algorithmic}` or `algorithmicx` / `algpseudocode` carefully.
  - Standard packages tested: `cite`, `amsmath,amssymb,amsfonts`, `algorithmic`, `algorithm`, `graphicx`, `textcomp`, `booktabs`, `multirow`, `url`, `xcolor`.

## 4. Figures and Tables
- Use vector PDF format for all generated plots and schematics for crisp rendering at arbitrary zoom.
- Store image assets in `report/` directly or referenced cleanly.
- Ensure all table text and headers use standard font sizes (`\footnotesize` or `\small`) to prevent overflowing single or double columns.

## 5. Author Order and Biographies
- **Author order**:
  1. Ayyappa Reddy (AM.SC.U4AIE23046) - Part 1: Environment, DQN, baselines, Max-Pressure tuning.
  2. Sriram (AM.SC.U4AIE23014) - Part 2: Distribution shift taxonomy, 3-sigma z-score monitor, diagnosis classifier.
  3. Vivek Reddy (AM.SC.U4AIE23050) - Part 3: Mitigation controller, fallback governor, closed-loop resilience.
- Equal standing: No team lead / head / leader terminology.
- Author biographies formatted with `\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{a1.png}}]{Author Name}` using template avatar placeholders `a1.png`, `a2.png`, `a3.png`.

