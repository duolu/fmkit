
from distutils.core import setup, Extension

module = Extension('fmkit_cutils', ['fmkit_cutils.c'])

setup(name='fmkit_cutils', version='1.0', ext_modules=[module])
