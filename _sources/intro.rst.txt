Introduction
====================================================

GrandFEP : GCMC/FEP - Alchemical free energy calculations in Grand Canonical ensemble

1. Background
---------------------

This Python module is designed to perform grand canonical Monte Carlo (GCMC) and free energy
calculations using OpenMM

2. Installation
--------------------

2.1. Download the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash
    
    git clone https://github.com/huichenggong/GrandFEP.git
    cd GrandFEP

2.2. Prepare the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the repository directory, you can find the env.yml file. You can use it to create a conda environment

.. code:: bash
    
    mamba env create -f env.yml
    # edit cuda and MPI version according to your cluster
    mamba activate grandfep_env
    pip install .

Or:

.. code:: bash

    mamba create -n grandfep_env python=3.12 numpy scipy pandas openmm openmmtools pymbar-core openmpi=4.1.5 mpi4py parmed cudatoolkit=11.8

Be aware that the CUDA version should be compatible with your GPU drivers, and the OpenMPI version
should be compatible with your cluster.

2.3. Later on the cluster 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    source /home/NAME/SOFTWARE/miniforge3/bin/activate grandfep_env


The GrandFEP module is released under the MIT licence. I hope my publication will come soon.

3. Author(s) and Contact
-----------------------------

**Chenggong Hui**

- **Email:** `chenggong.hui@mpinat.mpg.de <mailto:chenggong.hui@mpinat.mpg.de>`_
- **ORCID:** `0000-0003-2875-4739 <https://orcid.org/0000-0003-2875-4739>`_

For any questions or issues regarding this module, please contact Chenggong Hui.

