# Configuration file for the Sphinx documentation builder.
#
### 3. Add the src Directory to sys.path so that Sphinx can import your modules. This is necessary for autodoc to work:
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NFL Game Predictions'
copyright = '2023, Ryan Kunkel'
author = 'Ryan Kunkel'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

# Use the sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Enable the dark mode (this is the key part)
html_theme_options = {
    "style_nav_header_background": "#343131",  # This is a dark color
    "dark_mode": True  # This enables the dark mode
}

html_static_path = ['_static']
