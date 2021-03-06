from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="exptune",
    version="0.1",
    description="Experiment tuning interfaces implemented on top of Ray Tune",
    author="Shyam A. Tailor",
    author_email="sat62@cam.ac.uk",
    license="MIT",
    packages=find_packages(),
    install_requires=required,
)
