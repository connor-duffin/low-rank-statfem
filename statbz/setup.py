from setuptools import setup, find_packages

setup(name="statbz",
      version="0.0.1",
      author="Connor Duffin",
      author_email="connor.p.duffin@gmail.com",
      description="Statistical finite elements for reaction-diffusions.",
      license="MIT",
      packages=find_packages(),
      install_requires=[
          "numpy", "scipy", "pandas", "fenics", "h5py", "petsc4py", "pytest"
      ],
      classifiers=(
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ))
