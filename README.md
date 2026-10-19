<div style="padding-top:1em; padding-bottom: 0.5em;">
<img src="logo.png" width =200 align="right" />
</div>

# Statistical Learning with Python

## Online usage (recommended)

Course materials for the psy300 (ds300) seminar of the Data Science and Machine Learning Master's course at the University of Oldenburg. The content should primarily be accessed from the online book: [![Jupyter Book Badge](https://raw.githubusercontent.com/jupyter-book/jupyter-book/next/docs/media/images/badge.svg)](https://mibur1.github.io/psy300/)

## Local usage

Install the dependencies once:

```bash
cd <path/to/psy300>
pip install -r requirements.txt
```

Then start a live preview server, which prints a link (usually <http://localhost:3000>):

```bash
jupyter book start --execute
```

The *.ipynb* notebooks for the exercises are located in the `book/` folder and can either be opened locally or through Google Colab.

## Repository layout

| Path | Purpose |
|---|---|
| `myst.yml` | Book configuration and table of contents |
| `book/` | All chapters, exercises, solutions and figures |
| `book/*/quiz/` | JSON question banks for `jupyterquiz` / `jupytercards` |
| `_static/custom.css` | Theme overrides (dark-mode fixes for the quiz widgets) |
| `.github/workflows/deploy.yml` | Builds and publishes the book on every push to `main` |
