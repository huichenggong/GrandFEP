#!/usr/bin/env bash
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

while IFS=$'\x1f' read -r edge_name; do
    if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_999 ]; then
        continue
    fi
    cd       $base/$edge_name/$ff/$leg/$prot/
    pwd
    for rep in {0..2}
    do
        cd $base/$edge_name/$ff/$leg/$prot/
        if [ ! -d rep_$rep ]; then
            cp rep_999 rep_$rep -r
        fi
        cd rep_$rep
        for win in {0..15}
        do
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win
            # if link npt_eq.rst7 not exists, create link
            if [ ! -f npt_eq.rst7 ] || [ ! -L npt_eq.rst7 ]; then
                ln -s ../../../NPT_prod/rep_$rep/$win/npt_eq.rst7
            fi
        done
    done
    cd $base/$edge_name/$ff/$leg/$prot/
    $script_dir/run_GC_prep_box.py \
        -pdb ../system.pdb \
        -system ../system.xml.gz \
        -multidir rep_?/?/ rep_?/??/ \
        -yml gc_eq.yml \
        -start_rst7 npt_eq.rst7 \
        -odeffnm gc_start 

    
done < <($script_dir/csv_emit.py edge.csv edge_name)
