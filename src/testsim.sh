#!/bin/bash
export DNNCDEBUG=0xff

#---------working dirs-----------#
exedir="/home/yujc/caffe-dev/build/tools/"
#transdir="simpletest/"
transdir="floatfix/"
deploydir="fix_results/"
outdir="out_results/"

#----------file names------------#
#originfile="model_simple"
originfile="float_model"
deployfile="deploy"
fixtrainfile="fix_train_test"

#--------other parameters--------#
netname="vggface"
dpu="4096FA"
cpu="arm32"

if [ "$1" == "dece" ]
then
	echo "# quantize: "
	#${exedir}deephi_fix fix 						\
	decent fix 						\
			-model ${transdir}${originfile}.prototxt		\
			-weights ${transdir}${originfile}.caffemodel		\
		       	-calib_iter 10						\
		       	-gpu 1
		       	#-ignore_layers_file ignore_layers.txt			\

elif [ "$1" == "trans" ]
then
	${exedir}decent fixed2deploy -model fixedresults/fixed_model.prototxt	\
			-weights fixedresults/fixed_model.caffemodel 		\
			-gpu 1
	

elif [ "$1" == "defi" ]
then
	echo "# decent finetune"
	# ${exedir}deephi_fix finetune						\
	decent finetune						\
			-model ${deploydir}fix_train_test.prototxt		\
			-weights ${deploydir}fix_train_test.caffemodel		\
			-solver ${deploydir}fix_finetune_solver.prototxt	\
			-gpu 2

elif [ "$1" == "test" ]
then
	echo "# decent test"
	decent test -model ${deploydir}${fixtrainfile}.prototxt		\
			-weights ${deploydir}${fixtrainfile}.caffemodel		\
			-gpu 2							\
			-test_iter 1						\
			1> ${deploydir}fix_test.log 2>&1

elif [ "$1" == "dnnc" ]
then
	echo "# compile: "
	dnnc --prototxt=${deploydir}${deployfile}.prototxt 		\
		         --caffemodel=${deploydir}${deployfile}.caffemodel	\
			 --output_dir=${outdir}					\
			 --dpu=${dpu}						\
			 --cpu_arch=${cpu}					\
			 --net_name=${netname}					\
			 --mode=debug
#	sudo ./dnnc --prototxt=${deploydir}${deployfile}prototxt --caffemodel=${deploydir}${deployfile}caffemodel --output_dir=${outdir} --dpu=${dpu} --cpu_arch=${cpu} --net_name=${netname} --mode debug

else
	echo "*  error"
	echo "*  input dece for quantize"
	echo "*  input dnnc for compile"
fi

