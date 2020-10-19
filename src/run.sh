#!/bin/bash

for k in {2..14}
do
        echo $k
        for i in {1..40}
        do
                python exp.py k >> random_20201019_$k.log
        done 
done
                



