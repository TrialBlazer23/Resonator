from setuptools import setup, find_packages

setup(
    name="resonator-ai",
    version="0.1.0",
    author="Evan / Necessity Labs",
    description="Unsupervised thermodynamic cognitive safety monitoring for autoregressive LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": ["pytest", "jupyter", "pandas"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
