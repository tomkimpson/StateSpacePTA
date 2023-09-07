# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'StateSpacePTA'
copyright = '2023, Tom Kimpson'
author = 'Tom Kimpson'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
#html_theme = 'furo'
html_static_path = ['_static']

#Include doctests 
import pathlib
import sys
#sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.append("../py_src") # Means that I dont have to make src/ a proper python package



extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]
