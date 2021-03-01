
from distutils.core import setup, Extension

module = Extension('fmkit_utilities', ['fmkit_utilities.c'])

setup(name='fmkit_utilities', version='1.0', ext_modules=[module])
