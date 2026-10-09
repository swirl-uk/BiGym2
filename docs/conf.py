"""Sphinx configuration for the BiGym 2.0 documentation.

sphinx-book-theme with repository buttons, copybutton, sphinx-design and
sphinxcontrib-bibtex.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "BiGym 2.0"
author = "BiGym 2.0 contributors"
copyright = "2026, BiGym 2.0 contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 2

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Strip prompts when copying shell/python blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Task images, linked from the pages; not Sphinx sources.
    "images",
]

html_theme = "sphinx_book_theme"
html_title = "BiGym 2.0"
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
        "alt_text": "BiGym 2.0",
    },
    "repository_url": "https://github.com/swirl-uk/BiGym2",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "collapse_navigation": True,
}
