import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GrandFEP'
copyright = '2025, Chenggong Hui'
author = 'Chenggong Hui'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx_mdinclude'
]
typehints_fully_qualified = True       # <─ add this line
# Napoleon settings


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'page_width'       : '90%',
    'body_max_width'   : 'auto',
    'fixed_sidebar'    : True,
    'github_user'      : 'huichenggong',
    'github_repo'      : 'GrandFEP',
    'github_banner'    : True

}

autodoc_member_order = 'groupwise'

