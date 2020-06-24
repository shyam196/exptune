from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="exptune",
    version="0.1",
    description="Experiment tuning interfaces implemented on top of Ray Tune",
    author="Shyam A. Tailor",
    author_email="shyam.tailor@cs.ox.ac.uk",
    license="MIT",
    packages=["exptune"],
    install_requires=required,
)
