import io
from creadtonlp import __version__
from setuptools import find_packages, setup

setup(
    name="creadtonlp",
    version=__version__,
    author="Simon Anderson",
    url="www.creadto.com",
    description="Creadto Crawling System for Private Product",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
    ],
)