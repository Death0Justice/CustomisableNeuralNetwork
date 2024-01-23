from setuptools import setup, find_packages

setup(
    name="Customisable-numpy-NN",
    version="0.1.0",
    author="Death Justice",
    author_email="death0justice@gmail.com",
    description="""
        A layer-customisable Neural Network that only depends on NumPy.
        Custom functions usage is applicable by setting the self.net variables.
    """,
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
)