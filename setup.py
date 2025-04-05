from setuptools import setup, find_packages
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='grandfep',
    version=get_version("grandfep/__init__.py"),
    author='Chenggong Hui',
    author_email='chenggong.hui@mpinat.mpg.de',
    description='GCMC/MD with free energy perturbation',
    packages=find_packages(),
    install_requires=["pymbar>=3,<5",
                      "numpy",
                      "pandas",
                      "scipy>=1.7.0",
                      "openmm", "openmmtools",
                      "parmed",
                      "mpi4py",
                      "mdtraj", "mdanalysis",
                      ],
    python_requires='>=3.11',
      entry_points={
          'console_scripts': [
          'grand_RE_MPI=grandfep.grand_RE_MPI:main',
          ],
      },
    classifiers=['Programming Language :: Python :: 3',],
)