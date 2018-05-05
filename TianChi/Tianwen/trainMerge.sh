#!/bin/bash


# For training data
path=$1

cd $path
#cd ../data/first_train_index_20180131/

while read line
do
    cat $line >> subtrain_4w_times.txt
    echo '\n' >> subtrain_4w_times.txt
done < ../subtrain_label_4w_times.csv

























