#!/bin/bash

# For rank data
path=$1
cd $path
#cd ../data/first_rank_index_20180307/

#first
ls | head -n 10000 > sub1_label.txt
files=`ls | head -n 10000`
for i in $files
do
	cat $i >> sub1.txt
	echo '\n' >> sub1.txt
done


from=10000 #needn't change
to=20000
ls | head -n $to | tail -n $from > sub2_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub2.txt
	echo '\n' >> sub2.txt
done

to=30000
ls | head -n $to | tail -n $from > sub3_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub3.txt
	echo '\n' >> sub3.txt
done

to=40000
ls | head -n $to | tail -n $from > sub4_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub4.txt
	echo '\n' >> sub4.txt
done

to=50000
ls | head -n $to | tail -n $from > sub5_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub5.txt
	echo '\n' >> sub5.txt
done

to=60000
ls | head -n $to | tail -n $from > sub6_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub6.txt
	echo '\n' >> sub6.txt
done

to=70000
ls | head -n $to | tail -n $from > sub7_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub7.txt
	echo '\n' >> sub7.txt
done

to=80000
ls | head -n $to | tail -n $from > sub8_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub8.txt
	echo '\n' >> sub8.txt
done

to=90000
ls | head -n $to | tail -n $from > sub9_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub9.txt
	echo '\n' >> sub9.txt
done

to=100000
ls | head -n $to | tail -n $from > sub10_label.txt
files=`ls | head -n $to | tail -n $from`
for i in $files
do
	cat $i >> sub{10}.txt
	echo '\n' >> sub{10}.txt
done







