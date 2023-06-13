
#####################################################
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_DEBUG=VERSION
export NCCL_DEBUG=INFO
NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1 ##1(LOC) prevents hang on 16111MB and GPU-volatile 100%
export NCCL_P2P_LEVEL=nvl
#####################################################
subject="313"

declare -A pid_dict
pid_dict["313"]=0
pid_dict+=(["315"]=1 ["377"]=2 ["386"]=3 )
pid=${pid_dict[$subject]}

ARGS_MAIN_PY="sub_$subject/NeuralShader/painterDensity/main_neuralShader.py"
ARGS_TRAIN_PRINT_FILE="sub_$subject/NeuralShader/painterDensity/ProgramOutput/ModelTrain_$subject.txt"
#########  T R A I N   All Frames    ######### 
python $ARGS_MAIN_PY >> $ARGS_TRAIN_PRINT_FILE  \
--pid $pid --bs=2  --run_type=training 


ARGS_TRAIN_PRINT_FILE="sub_$subject/NeuralShader/painterDensity/ProgramOutput/ModelTest_$subject.txt"
#########  T E S T    ######### 
python $ARGS_MAIN_PY >> $ARGS_TRAIN_PRINT_FILE  \
--pid $pid --bs=1  --run_type=testing





