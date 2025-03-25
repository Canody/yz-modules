from setuptools import setup, find_packages

setup(
    name='yzplotlib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
    ],
    description='My custom plotting tools',
    author='Yao Zhang',
)
