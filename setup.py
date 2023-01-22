from setuptools import setup
from codecs import open
from os import path
import re
package_name = "gdtuo_tf"
root_dir = path.abspath(path.dirname(__file__))


def _requirements():
    return [name.rstrip() for name in open(path.join(root_dir, "requirements.txt")).readlines()]


with open(path.join(root_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(
        r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name=package_name,
    version=version,
    description="Implementation of GDTUO for TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mya-Mya/gdtuo_tf",
    author="Mya-Mya",
    author_email="",
    license="MIT",
    keywords="Machine Learning, Gradient Descent, Artificial Intelligence, Optimization",
    packages=[package_name],
    install_requires=_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)