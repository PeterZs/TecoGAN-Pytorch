#!/bin/bash

# create sbatch_run.sh 


if [ "$#" -ne 5 ]; then
    echo "########################################################" 
    echo " Usage: ./g_sbatch [jobname] [gpu_num] [node_name] [degradation] [model]."
    echo "########################################################" 
    exit 1 
fi 

jobname=$1       #your job script name 
# script_name=$5
path=$PWD 
if [ ! -d "${path}/logs" ]; then 
  mkdir -p ${path}/logs 
fi 

if [ ! -d "${path}/logs/${jobname}" ]; then 
  mkdir -p ${path}/logs/${jobname} 
fi

sbatch_file=sbatch_run.sh
touch ${sbatch_file}
cat > ${sbatch_file} <<'endmsg'
#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p pixel
endmsg



# basic settings
degradation=$4
model=$5
exp_id=001
gpu_id=7


# retain training or train from scratch
start_iter=0
if [[ ${start_iter} > 0 ]]; then
	suffix=_iter${start_iter}
else
	suffix=''
fi


exp_dir=./experiments_${degradation}/${model}/${exp_id}
# check
#if [ -d "$exp_dir/train" ]; then
#  	echo "Train folder already exists: $exp_dir/train"
#    echo "Please delete the train folder, or setup a new experiment"
#    echo "Exiting ..."
#  	exit 1
#fi


# backup codes
mkdir -p ${exp_dir}/train
cp -r ./codes ${exp_dir}/train/codes_backup${suffix}

echo "#SBATCH --gres=gpu:$2" >> ${sbatch_file}
echo "#SBATCH --nodelist=$3" >> ${sbatch_file}
echo "#SBATCH --job-name=${jobname}" >> ${sbatch_file} 
echo "#SBATCH -o ${path}/logs/${jobname}/%j.txt" >> ${sbatch_file} 
echo "srun --mpi=pmi2  python  codes/main.py --exp_dir ${exp_dir}  --mode train --model ${model} --opt train${suffix}.yml --gpu_id ${gpu_id} > ${exp_dir}/train/train${suffix}.log  2>&1 & " >> ${sbatch_file}
echo "echo \"Submit the ${jobname} job by run \'sbatch\'\" " >>  ${sbatch_file}

cat sbatch_run.sh
echo "Submit the ${jobname} job by run 'sbatch ${sbatch_file}'"
sbatch sbatch_run.sh
