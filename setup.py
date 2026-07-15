from setuptools import find_packages, setup

setup(
    name="demand-forecasting",
    version="0.1.0",
    description="Modular ecommerce demand forecasting project",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "numpy",
    ],
)
