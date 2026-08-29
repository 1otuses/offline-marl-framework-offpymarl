# bash test/offline_train/train.sh 0 bc/qmix_cql/matd3_bc/... 3m/2s3z/... expert/medium/medium-replay/random 0/1/...
# export CUDA_VISIBLE_DEVICES=$1
# bash test/offline_train/train.sh qmix_cql 2s1z_vs_3z expert 1 2
python src/main.py \
    --offline \
    --config=$1 \
    --env-config=sc2_offline \
    --map_name=$2  \
    --offline_data_quality=$3 \
    --seed=$4 \
    --g=$5 \
    --t_max=40000 \
    --test_interval=250 \
    --log_interval=250 \
    --runner_log_interval=250 \
    --learner_log_interval=250 \
    --save_model_interval=10000
