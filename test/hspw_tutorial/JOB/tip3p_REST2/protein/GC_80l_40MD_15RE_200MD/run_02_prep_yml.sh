#!/usr/bin/env bash
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name center; do
    mkdir -p $base/$edge_name/$ff/$leg/$prot/rep_999
    cd       $base/$edge_name/$ff/$leg/$prot/rep_999
    pwd
    # sed keyword GEN_VEL and RESTRAINT
    sed -e 's/GEN_VEL/true/' \
        -e "s/RESTRAINT/false/" \
        ../../../../../JOB/$ff/$leg/$prot/rep_999/gc_tmp.yml    >  ./gc_eq.yml
    cat ../../../../../JOB/$ff/$leg/$prot/prot_eq.yml           >> ./gc_eq.yml
    cat ../../../OPT_lam.yml                                    >> ./gc_eq.yml
    sed -e "s/CENTERATOM/$center/" \
        ../../../../../JOB/$ff/$leg/$prot/rep_999/GC_center.yml >> ./gc_eq.yml
    
    sed -e 's/GEN_VEL/false/' \
        -e "s/RESTRAINT/false/" \
        ../../../../../JOB/$ff/$leg/$prot/rep_999/gc_tmp.yml    >  ./gc.yml
    cat ../../../../../JOB/$ff/$leg/$prot/prot.yml              >> ./gc.yml
    cat ../../../OPT_lam.yml                                    >> ./gc.yml
    sed -e "s/CENTERATOM/$center/" \
        ../../../../../JOB/$ff/$leg/$prot/rep_999/GC_center.yml >> ./gc.yml

    for win in {0..15}
    do
        mkdir $win -p
        sed "s/INIT_LAMBDA_STATE/$win/" gc_eq.yml > $win/gc_eq.yml
        sed "s/INIT_LAMBDA_STATE/$win/" gc.yml    > $win/gc.yml
    done

    # if rep_0, rep_1, rep_2 exists, update the yml files
    for rep in {0..2}
    do
        if [ -d $base/$edge_name/$ff/$leg/$prot/rep_$rep ]; then
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
            echo "Updating $PWD"
            for win in {0..15}
            do
                cd $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win
                cp ../../rep_999/$win/gc*.yml .
            done
        fi
    done

    
done < <($script_dir/csv_emit.py center.csv edge_name center)
