#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

# make sure meta.sh and memory.sh exist
for filename in meta.sh memory.sh; do
    if [[ ! -f $filename ]]; then
        echo "$filename not found!"
        exit 1
    fi
done

source meta.sh
PR_cycle=${meta_data[PR_cycle]}


sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

tempdir=$(mktemp -d /dev/shm/mbar_${sys_name}_${ff}_${leg}_${prot}_XXXXXX)

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"
echo "PR cycle   : $PR_cycle"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    for rep in 0 1 2 10
    do
        if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep ]; then
            continue
        fi
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd
        for partN in 2 4 5 8 10 12 15 16 24 30 32
        do
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
            if [ -f mbar_${partN}ns/mbar.log ] &&  grep -q "Total " mbar_${partN}ns/mbar.log  ; then
                echo "rep_$rep part $partN done."
                continue
            fi

            if [ -f 15/$partN/gc.log ] && grep -q "RE Step $((PR_cycle * 5 -1))" 15/$partN/gc.log ; then
                echo "rep_$rep part $partN MBAR ..."
                mkdir -p mbar_${partN}ns
                cd       mbar_${partN}ns
                for i in $(seq 1 $partN)
                do
                    mkdir -p $tempdir/0/$i
                    cp ../0/$i/gc.log $tempdir/0/$i/
                done
                for win in {0..15}
                do
                    grep "T   =" $tempdir/0/1/gc.log | tail -n 1  >  $tempdir/$win.dat
                    for i in $(seq 1 $partN)
                    do
                        grep " - INFO: $win:" $tempdir/0/$i/gc.log  >> $tempdir/$win.dat
                    done
                    sed -i "s/- INFO: $win:/Reduced_E:/" $tempdir/$win.dat
                    awk 'NR % 3 == 1' $tempdir/$win.dat > ./$win.dat
                done
                rm -r $tempdir/*
                $script_dir/analysis/MBAR.py -log ?.dat ??.dat -kw "Reduced_E:" -m MBAR -csv mbar_dG.csv > mbar.log 2> mbar.err
                rm ?.dat ??.dat
            else
                echo "rep_$rep part $partN not finished yet."
            fi

        done
    done
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
rm -r $tempdir
