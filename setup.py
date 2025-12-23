from setuptools import setup, find_packages

setup(
    name = "dgp_set",
    version = "0.1.0",
    description = "",
    author = "Yun Li",
    author_email = "yunli.tudelft@gmail.com",
    python_requires = ">= 3.8",

    package_dir = {"": "src"},
    packages=find_packages(where="src"),

    install_requires = [
        "numpy ~= 1.24",
        "matplotlib ~= 3.7",
        "scipy ~= 1.10",
        "scikit-learn ~= 1.3",
        "alphashape ~=1.3.1",
        "shapely ~= 2.0.1",
    ]
)
