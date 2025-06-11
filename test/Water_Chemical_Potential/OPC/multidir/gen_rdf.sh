
base=$PWD
for i in 2 3 4 5
do
    cd $base/$i
    python ../0/dcd_2_xtc.py
    gmx rdf -s ../../water_opc.pdb -f opc_npt_output.xtc -n ../index.ndx -ref center -sel O -bin 0.01
done

