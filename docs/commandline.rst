GrandFEP CLI
============

.. contents:: On this page
    :local:
    :depth: 1

run_GC_RE.py
-----------------
Run GrandFEP REST2+GCMC simulations

**Example**

.. code-block:: bash

    for win in {0..15}
    do
        mkdir $win/0 $win/1
    done
    mpirun -np 16 $script_dir/run_GC_RE.py \
        -pdb    system.pdb \
        -system system.xml.gz \
        -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
        -yml gc_eq.yml \
        -maxh 18 \
        -ncycle 576 \
        -start_rst7   gc_start.rst7 \
        -start_jsonl  gc_start.jsonl \
        -deffnm       0/gc

    mpirun -np 16 $script_dir/run_GC_RE.py \
        -pdb    system.pdb \
        -system system.xml.gz \
        -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
        -yml gc.yml \
        -maxh 9 \
        -ncycle 1200 \
        -start_rst7  0/gc.rst7 \
        -start_jsonl 0/gc.jsonl \
        -deffnm      1/gc

**Description**

This script combines MD / GC / RE. It requires pdb file as topology, system xml file as system definition,
yaml file as simulation parameters, and multiple directories for different lambdas.


run_NPT_RE.py
-----------------
Run NPT REST2+GCMC simulations

**Example**

.. code-block:: bash

    for win in {0..15}
    do
        mkdir $win/0 $win/1
    done
    mpirun -np 16 $script_dir/run_NPT_RE.py \
        -pdb    system.pdb \
        -system system.xml.gz \
        -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  \
        -yml npt.yml \
        -maxh 10 \
        -ncycle 1000 \
        -start_rst7  npt_eq.rst7 \
        -deffnm      0/npt
    mpirun -np 16 $script_dir/run_NPT_RE.py \
        -pdb    system.pdb \
        -system system.xml.gz \
        -multidir 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  \
        -yml npt.yml \
        -maxh 10 \
        -ncycle 1000 \
        -start_rst7  0/npt.rst7 \
        -deffnm      1/npt