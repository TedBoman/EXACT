from setuptools import setup, find_packages

setup(
    name="ML_models",
    version="1.0.0",
    packages=find_packages(exclude=["model_interface", "setup", "__init__*"]),
)