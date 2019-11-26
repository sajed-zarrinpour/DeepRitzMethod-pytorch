#!/bin/bash
n=${1?Error: no input given}
export BASE_DIRECTORY="n_${n}_results"
mkdir "${BASE_DIRECTORY}" >> Output
echo "Solving The PDE with FreeFem++"
FreeFem++ poisson.edp -n $n >> Output
mv "${BASE_DIRECTORY}/data.csv" "${BASE_DIRECTORY}/train.csv"
FreeFem++ poisson.edp -n $n + 10 >> Output
mv "${BASE_DIRECTORY}/data.csv" "${BASE_DIRECTORY}/test.csv"
#echo "Exporting FreeFem++ Generated Files To PNG Formatt..."
#sh ./ps2png.sh mesh.eps mesh.png $n
#sh ./ps2png.sh solution.eps solution.png $n
#echo "Creating Train And Test DataSets..."
#/home/samim/anaconda2/bin/python ./DataSetOperations.py --filename $BASE_DIRECTORY 
echo "Learning..."
#echo "Exporting The Model Output...."
#python2 ./DRM.py --test "${BASE_DIRECTORY}/test.csv" --train "${BASE_DIRECTORY}/train.csv" --PREFIX "${BASE_DIRECTORY}"
/home/samim/anaconda2/bin/python ./drm-torch.py --test "${BASE_DIRECTORY}/test.csv" --train "${BASE_DIRECTORY}/train.csv" --PREFIX "${BASE_DIRECTORY}"
gnuplot -c plot "${BASE_DIRECTORY}/predictions.csv" "${BASE_DIRECTORY}/DRM_SOLUTION.png"

echo "cleaning up..."
rm mesh.eps
rm solution.eps
