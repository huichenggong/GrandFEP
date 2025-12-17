base=/home/chui/E29Project-2023-04-11/136-grandFEP/benchmark/08-Water-Set/Benchmark_waterSet_2025_10_01/sim/hspw
for edge in edge_1_to_4 edge_4_to_3 edge_1_to_3 edge_1_to_2 edge_3_to_2 edge_2_to_4
do
    mkdir -p ./$edge/tip3p_REST2/
    cp $base/$edge/04_tip3p_REST2/OPT_lam.yml \
    ./$edge/tip3p_REST2/
done