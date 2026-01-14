Introduction
====================================================

GrandFEP : GCMC/FEP - Alchemical free energy calculations in Grand Canonical ensemble

.. contents:: On this page
    :local:
    :depth: 2


1. Background
---------------------

This Python module is designed to perform grand canonical Monte Carlo (GCMC) and free energy
calculations using OpenMM

2. Installation
--------------------

2.1. Download the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash
    
    git clone https://github.com/huichenggong/GrandFEP.git
    cd GrandFEP

2.2. Prepare the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the repository root directory, a ``env.yml`` file is provided. You can use it to create a conda environment.
Please edit the cuda and MPI version before you run the following command. Use ``nvidia-smi``
to check the hightest cuda version the driver supports. MPI version should follow your cluster
configuration.

.. code-block:: bash

    # edit cuda and MPI version according to your cluster
    mamba env create -f env.yml
    mamba activate grandfep_env
    pip install .

Or:

.. code-block:: bash

    mamba create -n grandfep_env python=3.12 numpy scipy pandas openmm openmmtools pymbar-core openmpi=4.1.5 mpi4py parmed cudatoolkit=11.8

Check cuda and MPI as what we previously mentioned.

2.3. Later on the cluster 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    source /home/NAME/SOFTWARE/miniforge3/bin/activate grandfep_env
    module load openmpi4/gcc/4.1.5


3. Citation
-----------------------------

The GrandFEP module is released under the MIT licence. I hope my publication will come soon.

4. Author(s) and Contact
-----------------------------

**Chenggong Hui**

- **Email:** `chenggong.hui@mpinat.mpg.de <mailto:chenggong.hui@mpinat.mpg.de>`_
- **ORCID:** `0000-0003-2875-4739 <https://orcid.org/0000-0003-2875-4739>`_
- **GitHub** `GrandFEP <https://github.com/huichenggong/GrandFEP>`_

For any questions or issues regarding this module, please contact Chenggong Hui (惠成功).


5. Developmental Log
-----------------------------
.. toctree::
    :maxdepth: 2

    devlog