#!/usr/bin/env bash
ff=$(basename $PWD)
amber_name="07_tip3p"

echo "water ff   : $ff"

cd ../../
base=$PWD

while IFS=$'\x1f' read -r edge_name lig1 lig2; do
    echo "$edge_name $lig1 $lig2"
    mkdir -p "$base/$edge_name/$ff/water"
    cd       "$base/$edge_name/$ff/water"
    pwd
    cp ../../map.yml                    ../hybrid_map.yml
    cat ../../../JOB/basic.yml       >> ../hybrid_map.yml

    if [ ! -f system.xml.gz ]; then
        $script_dir/hybrid.py \
            -prmtopA "../../../lig_prep/$lig1/$amber_name.prmtop" \
            -prmtopB "../../../lig_prep/$lig2/$amber_name.prmtop" \
            -inpcrdA "../../../lig_prep/$lig1/$amber_name.inpcrd" \
            -inpcrdB "../../../lig_prep/$lig2/$amber_name.inpcrd" \
            -yml ../hybrid_map.yml \
            -pdb system.pdb \
            -system system.xml.gz \
            -REST2 \
            -dum_dihe_scale 0.0  1.0  0.0  0.0  0.0 > hybrid.log 2> hybrid.err &
    fi

    mkdir -p "$base/$edge_name/$ff/protein"
    cd       "$base/$edge_name/$ff/protein"
    pwd
    if [ ! -f system.xml.gz ]; then
        $script_dir/hybrid.py \
            -prmtopA "../../../pro_prep/$lig1/$amber_name.prmtop" \
            -prmtopB "../../../pro_prep/$lig2/$amber_name.prmtop" \
            -inpcrdA "../../../pro_prep/$lig1/$amber_name.inpcrd" \
            -inpcrdB "../../../pro_prep/$lig2/$amber_name.inpcrd" \
            -yml ../hybrid_map.yml \
            -pdb system.pdb \
            -system system.xml.gz \
            -REST2 \
            -dum_dihe_scale 0.0  1.0  0.0  0.0  0.0 > hybrid.log 2> hybrid.err &
    fi
    wait
    
    cd       "$base/$edge_name/$ff/"
    ls -lh */system.*



done < <($script_dir/csv_emit.py edge.csv edge_name Lig_1 Lig_2)



