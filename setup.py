#!/usr/bin/env python
import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="bttr",
    version="0.0.1",
    description="Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer",
    author="Wenqi Zhao",
    author_email="1027572886a@gmail.com",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/Green-Wood/BTTR",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    packages=find_packages(),
)
