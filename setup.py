from setuptools import setup, find_packages

setup(
    name='yz-tools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
    ],
    description='Collection of custom tools by Yao Zhang',
    author='Yao Zhang',
    package_data={},
)
