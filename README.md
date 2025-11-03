<div style="padding-top:1em; padding-bottom: 0.5em;">
<img src="logo.png" width =200 align="right" />
</div>

# Statistical Learning with Python

## Online usage (recommended)

Course materials for the psy300 (ds300) seminar of the Data Science and Machine Learning Master's course at the University of Oldenburg. The content should primarily be accessed from the online book: [![Jupyter Book Badge](https://raw.githubusercontent.com/jupyter-book/jupyter-book/next/docs/media/images/badge.svg)](https://mibur1.github.io/psy300/)

## Local usage

Running the book locally requires to first build the book from source:

```python
cd <path/to/book/>
pip install -r requirements.txt
jb build .
```

This will create the html files in the `_build/` folder. The book can then be used by opening the `_build/html/index.html` file in a browser. The *.ipynb* notebooks for the exercises are located in the `book/` folder and can can either be opened locally or through Google Colab.
