# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

print(sys.executable)


# -- Project information -----------------------------------------------------

project = 'Lightweaver'
copyright = '2022, C. Osborne'
author = 'C. Osborne'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
]

sphinx_gallery_conf = {'examples_dirs': '../examples', 'gallery_dirs': 'auto_examples'}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


def write_renames_table():
    """
    Generate the table of renamed names for the migration guide from
    lightweaver/_renames.py, so the docs can't drift from the code.
    """
    import os

    from lightweaver._renames import INTERNAL_NO_ALIAS, RENAMES

    rows = sorted(RENAMES.items(), key=lambda kv: kv[0].lower())
    lines = [
        '.. list-table::',
        '   :header-rows: 1',
        '',
        '   * - Old name',
        '     - New name',
    ]
    for old, new in rows:
        note = ' (no deprecated alias)' if old in INTERNAL_NO_ALIAS else ''
        lines += [f'   * - ``{old}``', f'     - ``{new}``{note}']
    path = os.path.join(os.path.dirname(__file__), '_generated_renames.rst')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def skip_deprecated_aliases(app, what, name, obj, skip, options):
    from lightweaver.deprecation import deprecated_alias

    if isinstance(obj, deprecated_alias):
        return True
    return None


def setup(app):
    write_renames_table()
    app.connect('autodoc-skip-member', skip_deprecated_aliases)
