#!/usr/bin/env python
from distutils.core import setup, Extension
import numpy as np

example_module = Extension('_pyabcore',
    sources=['pyabcore.cpp', 'pyabcore_wrap.cxx','bigraph.cpp', 'kcore.cpp'],
)
setup (
    name = 'pyabcore',
    version = '0.1',
    author = "yujianke",
    description = """alpha beta core for python""",
    ext_modules = [example_module],
    py_modules = ["pyabcore"],
    include_dirs=[np.get_include()]
)