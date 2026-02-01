#!/usr/bin/env bash
script_dir="/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/Script/"
source /home/chui/Software-2023-04-11/miniforge3/bin/activate grandfep_dev

# make sure meta.sh exist
if [[ ! -f meta.sh ]]; then
    echo "meta.sh not found!"
    exit 1
fi

source meta.sh

sys_name=$(basename $(dirname $(dirname $(dirname $(dirname $PWD)))))
ff=$(basename $(dirname $(dirname $PWD)))
leg=$(basename $(dirname $PWD))
prot=$(basename $PWD)

PR_cycle=${meta_data[PR_cycle]}
maxh=${meta_data["PR_maxh_$sys_name"]}

echo "System     : $sys_name"
echo "water ff   : $ff"
echo "Alchem leg : $leg"
echo "Protocol   : $prot"
echo "PR cycles  : $PR_cycle"

cd ../../../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    for rep in 0 1 2
    do
        if [ ! -d $base/$edge_name/$ff/$leg/$prot/rep_$rep/ ]; then
            echo "$edge_name rep_$rep not found!"
            continue
        fi
        cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
        pwd

        partN=5
        for partN in {5..30..5}
        do
            cd $base/$edge_name/$ff/$leg/$prot/rep_$rep
            dcd_all=""
            for part in $(seq $((partN -4)) $partN)
            do
                dcd_all+="$part/npt.dcd "
            done

            if [ -f 15/$partN/npt.log ] && grep -q "RE Step $((PR_cycle * 5 - 1))" 15/$partN/npt.log ; then
                printf "rep_%d clean %2d - %2d ...\n" "$rep" "$((partN -5))" "$partN"
                for win in {0..15}
                do
                    cd $base/$edge_name/$ff/$leg/$prot/rep_$rep/$win
                    # all exists?
                    all_exist=true
                    for part in $(seq $((partN -4)) $partN)
                    do
                        if [ ! -f $part/npt.dcd ] ; then
                            all_exist=false
                            break
                        fi
                    done
                    if [ "$all_exist" = false ] ; then
                        continue
                    fi
                    
                    xtc_out=$(printf "npt_%02d_%02d.xtc" "$((partN -4))" "$partN")
                    $script_dir/dcd_2_xtc.py -p ../../../system.pdb \
                        -dcd   $dcd_all \
                        -xtc   $xtc_out > clean_$xtc_out.log 2>&1
                    
                    # check xtc output and clean dcd
                    if [ -f $xtc_out ]; then
                        trj_frame=$(~/Software-2023-04-11/bin/xtc_stat $xtc_out 2>&1 | awk '/nr of frames/{print $NF}')
                        if [ "$trj_frame" -eq 25 ] ; then
                            rm $dcd_all
                        else
                            echo "  WARNING: $PWD/$xtc_out has $trj_frame frames, expected 25."
                        fi
                    else
                        echo "  WARNING: $PWD/$xtc_out not found!"
                    fi
                done
            fi
        done
    done
done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)
