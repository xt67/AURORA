from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="aurora",
    version="0.2.0",
    description="AURORA / MediSyn: reproducible TB-manifestation image augmentation research",
    author="Rayan",
    packages=find_packages(),
    install_requires=[line.strip() for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")],
    python_requires=">=3.10,<3.13",
)
