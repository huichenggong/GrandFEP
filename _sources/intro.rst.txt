Introduction
====================================================

GrandFEP : GCMC/FEP - Alchemical free energy calculations in GC

Background
----------

This Python module is designed to perform grand canonical Monte Carlo (GCMC) and free energy
calculations using OpenMM

Installation
--------------------

The mamba/conda environment can be prepared using the following command:

.. code:: bash

    mamba env create -f env.yml
    # edit cuda and MPI version according to your cluster

Or:

.. code:: bash

    mamba create -n grandfep_env python=3.12 numpy scipy pandas openmm openmmtools pymbar-core openmpi=4.1.5 mpi4py parmed cudatoolkit=11.8

Be aware that the CUDA version should be compatible with your GPU drivers, and the OpenMPI version
should be compatible with your cluster.

Later on the cluster, 

.. code:: bash

    source /home/NAME/SOFTWARE/miniforge3/bin/activate grandfep_env


The GrandFEP module is released under the MIT licence. I hope my publication will come soon.

Author(s)
------------

- Chenggong Hui `<chenggong.hui@mpinat.mpg.de>` ORCID:0000-0003-2875-4739

Contact
-------

If you have any problems or questions regarding this module, please contact Chenggong

