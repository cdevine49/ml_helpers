import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_helpers",
    version="0.0.22",
    author="Conor Devine",
    author_email="conorjdevine@gmail.com",
    description="Functions to help build machine learning tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cdevine49/ml_helpers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
