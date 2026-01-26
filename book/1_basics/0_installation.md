---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# <i class="fa-brands fa-python"></i> Setup/Installation

All tutorials in this seminar can be accessed via **Google Colab** by clicking the rocket icon on the respective exercise pages (a Google account is required). Alternatively, you may choose to run the code locally on your own computer. The following sections provide a brief introduction to installing Python and using the course materials locally.

Python is an open-source programming language that can be freely downloaded and installed from its official website. In this guide, however, we will use Python through an environment manager called Conda. This allows you to install Python versions and manage project dependencies in isolated environments.


## Installing Conda

You can download and install Conda trough Miniforge as described in the the [documentation](https://conda-forge.org/download/).

**Linux/MacOS**

Open a terminal in the folder where your `.sh` script is located and run the script (*hint: after typing `bash Miniforge3` you should be able to hit `Tab` to autocomplete the command*):

```
bash Miniforge3-$(uname)-$(uname -m).sh
```

After a succesful installation, simply open a new terminal. You should now see a `(base)` in the beginning of each line, which tells you that the installation was succesful.

**Windows**

On Windows, simply run the `.exe` file and install everything. We also recommend you to check the **"Add Miniforge3 to my PATH environment variable"** checkbox during installation. This will enable you to use `conda` from any terminal, and not just `Miniforge Prompt`, which will be installed in any case. Once the installation is finished, depending on your choice during the installation, you can either open your standard terminal or the `Miniforge Prompt`. In any case, you should then be greeted with a `(base)` in the beginning of the line, which means the installation was succesful and you are now in the base environment.


Once the installation is complete, open a new terminal and create a new environment for the seminar:

```bash
conda create -n psy300 python
```

The terminal prompt should now display `(psy300)`, showing that the new environment is active. Inside this environment, you can install additional packages as needed, for example:

```bash
pip install numpy pandas matplotlib
```

To list all installed packages:

```bash
pip list
```

## Visual Studio Code

Next, install [Visual Studio Code](https://code.visualstudio.com/) (Microsoft) or the fully open source version [VSCodium](https://vscodium.com/). During installation, you can enable the *“Open with Code”* options for easier usability.

Inside VS Code:

1. Install the **Python** extension.
2. Create a new file (e.g. `example.py`) and type:

   ```python
   print("Welcome to psy300!")
   ```
3. Press the run button ▶️ at the top right.
   When prompted, select the **psy300** environment as your Python interpreter.


## Jupyter Notebooks

For interactive code and text (used in the exercises), install the **Jupyter** extension in VS Code.
You can then create `.ipynb` notebooks and select your `psy300` environment as the **kernel**.

You can then create code cells:

```{code-cell}
a = 1
b = 2
a + b
```

Or Markdown cells (for text/documentation):

```
## This is a title
You can write text, lists, or include equations like $a^2 + b^2 = c^2$.
```


## Working with the Book (Optional)

Clone/download the course book from [GitHub](https://github.com/mibur1/psy300) and install the required packages:

```bash
cd path/to/psy300
pip install -r requirements.txt
```

To build the book locally:

```bash
jb build .
```

After building, open `_build/html/index.html` or follow the terminal link to view it in your browser.

---

```{admonition} Summary
:class: tip

You can use Python in different ways:

- Through scripts (`.py` files)  
- Through interactive Jupyter notebooks (`.ipynb` files)

Environment Managers (such as Miniforge/Conda) allow you to:
- Create environments: `conda create -n <env_name> python=3.12`  
- Switch between them: `conda activate <env_name>`  
- Install packages: `pip install <package>` or `conda install <package>`
```
