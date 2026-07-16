from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "avcheck_native",
        ["native/bindings.cpp", "native/tone_curve.cpp"],
        include_dirs=["native"],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
