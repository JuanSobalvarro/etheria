from setuptools import setup, find_packages

setup(
    name="etheria",
    version="0.0.1-alpha",
    description="Etheria MLP library for Edge AI applications",
    author="Juan Sobalvarro",
    author_email="sobalvarrog.juans@gmail.com",
    url="https://github.com/JuanSobalvarro/etheria",
    packages=find_packages(),  # finds 'etheria' and 'etheria._etheria'
    package_data={
        "etheria._etheria": ["_etheria.pyd"],  # include compiled module
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
)
