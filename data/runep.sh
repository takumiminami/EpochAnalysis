#!/bin/bash

PLTPY=pltsnglni-v171.py
version="_v41716"

if [ $# -eq 1 ]; then
    case $1 in
        "clean")
            rm *.log
            rm deck.*
            rm epoch2d.dat 
            rm *.sdf
            /bin/rm -r figures
            /bin/rm -r fn
            /bin/rm -r field
            /bin/rm -r phase 
            ;;
        *)
            echo "Undefined option : "$1
            exit 1 
            ;;
    esac
fi

# ------- functions ------- #
setStarttime() {
    start_time=`date +%s`
}
function getEndtime() {
    end_time=`date +%s`

    SS=`expr ${end_time} - ${start_time}`

    HH=`expr ${SS} / 3600`
    SS=`expr ${SS} % 3600`
    MM=`expr ${SS} / 60`
    SS=`expr ${SS} % 60`

    echo "Elapse     : ${HH}:${MM}:${SS}"
    echo ""
}
# ------------------------- #

echo "--Config---------------------"
read -p "epoch dimension       :" dim
read -p "use qed on server(y/n):" qswich
#read -p "output file header    :" aho
read -p "mpirun -np            :" thr
echo ""
echo "--Execute--------------------"

if [ "${qswich}" = "y" ]; then
	qed="qed"
else
	qed=""
fi

index=1
flag=1
while [ $flag == 1 ]; do
	fname=out$(printf "%04d" $index).log
	if [ -e $fname ]; then
		index=$(( $index + 1))
	else
		name=$fname
		flag=0
	fi
done

echo epoch${dim:='2'}d_v${version}${qed} 
#echo output file Name = ${name} 
read -p "enter to start..." ch

echo ${PWD} > deck.file
EXELOG="execute.log"
{
    echo "----------------------------"
	echo "EPOCH${dim}D ${version}"
    echo "start time : "`LANG=US.UTF-8 date`
    echo "Output     : "${fname}
    echo "Processes  : "${thr:=1}
    echo ""
} >> ${EXELOG}
PLTLOG=plot$(printf "%04d" $index).log

setStarttime
if [ -z "${thr}" ]; then
	nohup epoch${dim}d${version} < deck.file > ${name} && getEndtime >> ${EXELOG} && \
    setStarttime && ./${PLTPY} &> ${PLTLOG} && getEndtime >> ${PLTLOG} &
else
	nohup mpirun -np ${thr:=1} epoch${dim}d${version}${qed} < deck.file > ${name} && getEndtime >> ${EXELOG} && \
    setStarttime && ./${PLTPY} &> ${PLTLOG} && getEndtime >> ${PLTLOG} &
fi



