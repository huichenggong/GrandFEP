for i in {0..5}
do
    mkdir -p $i
    cp md.yml $i
    sed -i "s/INIT_LAMBDA/$i/g" $i/md.yml
done
