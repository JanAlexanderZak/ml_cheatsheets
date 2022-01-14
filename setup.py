"""
Execute:
python setup.py sdist bdist_wheel

Install locally in root dir
# pip install -e .
"""

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    description='environment_template',
    version="0.0.1",
    author='Jan Alexander Zak',
    author_email='@',
    url='.git',
    name='environment_template',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),  # this line resolves tests problems!
    python_requires=">=3.7",
    #extras_required=dict(test=['pytest-mypy-pylint']),
    install_requires=['pytest', 'mypy', 'pylint',],
)
