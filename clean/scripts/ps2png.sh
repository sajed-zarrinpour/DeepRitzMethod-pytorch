#! /bin/bash
#
in=${1?Error: no input file name given}
out=${2?Error: no output file name given}
n=${3?Error: no input given}
PREFIX="n_${n}_results"
convert $in "${PREFIX}/${out}"
#rm *.ps
echo "->  Done!"
