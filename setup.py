from setuptools import setup, find_packages

setup(
    name='tools',
    version='0.1',
    packages=find_packages(where="", exclude=("test", )),
    package_dir={"": "./"},
)