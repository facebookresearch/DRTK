# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# -- Path setup --------------------------------------------------------------

import builtins
import os
import sys

builtins.__sphinx_build__ = True

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
print(target_dir)
from drtk import __version__

# -- Project information -----------------------------------------------------

project = "DRTK"
copyright = "2024 Meta Platforms, Inc"
author = "Meta"
version = __version__

# -- General configuration ---------------------------------------------------

html_theme = "pydata_sphinx_theme"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_markdown_builder",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.video",
]
autosummary_generate = True
katex_prerender = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

master_doc = "index"
autodoc_typehints = "none"


html_theme_options = {
    "navbar_align": "content",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_center": ["legal"],
    "collapse_navigation": True,
    "secondary_sidebar_items": ["page-toc"],
    "show_prev_next": False,
    "back_to_top_button": False,
    "pygments_light_style": "a11y-light",
    "pygments_dark_style": "a11y-dark",
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/facebookresearch/DRTK",
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github fa-2xl",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
            "attributes": {},
        }
    ],
    "logo": {
        "text": "DRTK",
    },
}

html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
