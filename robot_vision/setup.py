# setup.py
import os
from setuptools import setup, find_packages

# Optionally read your requirements.txt for dependencies:
def parse_requirements(file_path):
    """
    Parse a requirements file into a list of strings.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

requirements_file = "requirements.txt"
install_requires = parse_requirements(requirements_file) if os.path.exists(requirements_file) else []

# Optionally read your README.md for a detailed package description:
long_description = ""
readme_file = "README.md"
if os.path.exists(readme_file):
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="robot_vision",
    version="0.1.0",
    description="Robot Vision Infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Xiangyu Han",
    author_email="xiangyu.han@quantgroup.com",
    packages=find_packages(),        
    include_package_data=True,        
    install_requires=install_requires,
    python_requires=">=3.10",          
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
