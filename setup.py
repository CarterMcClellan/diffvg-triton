from setuptools import setup, find_packages

setup(
    name="diffvg_triton",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton",
        "numpy",
    ],
    python_requires=">=3.8",
)
