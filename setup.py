#! /usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="mk_motormap",
    version='v0.1',
    author="Erin Kado-Fong",
    author_email="kadofong@princeton.edu",
    packages=["modelcobras"],
    url="https://github.com/ekadofong/mk_motormap",
    license="MIT",
    description="Generate controlled step motor maps",
    install_requires=['scipy','george','pandas',
                      'matplotlib','numpy']    
)
