"""
Setup script for IDS CNN Model Package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        if line.strip() and not line.startswith("#")
        else None
        for line in fh
    ]
    requirements = [r for r in requirements if r]

setup(
    name="ids-cnn-model",
    version="0.1.1",
    author="Dogancan Karakoc",
    author_email="dogncankrkc@gmail.com",
    description="Lightweight CNN model for network intrusion detection (binary + multiclass)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dogncankrkc/ids-1",   
    packages=find_packages(where="src"),          
    package_dir={"": "src"},                       
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking :: Monitoring :: IDS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.970",
        ],
    },
)
