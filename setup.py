from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pi-topology",
    version="0.1.0",
    author="Π-Topology Research Group",
    author_email="research@pi-topology.org",
    description="Parallel State-Space Mathematics Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pi-topology",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["torch>=2.0.0", "torchvision"],
        "visualization": ["matplotlib>=3.5.0", "plotly>=5.10.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "jupyter>=1.0.0",
        ],
        "examples": ["numpy>=1.21.0", "pandas>=1.4.0", "scikit-learn>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "pi-benchmark=pi_topology.benchmarks:run_benchmarks",
            "pi-visualize=pi_topology.visualizer:cli",
        ],
    },
    include_package_data=True,
    keywords="parallel-computing, optimization, gpu, mathematics, topology",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pi-topology/issues",
        "Source": "https://github.com/yourusername/pi-topology",
        "Documentation": "https://pi-topology.readthedocs.io",
    },
)