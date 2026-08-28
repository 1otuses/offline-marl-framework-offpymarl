## bash xx/medium.sh 0 qmix 2s1z_vs_3z medium 0.6
# export CUDA_VISIBLE_DEVICES=$1 # 这里显示规定训练时使用哪个GPU
python src/main.py \
    --collect \
    --config=$1 \
    --env-config=sc2_collect \
    --map_name=$2 \
    --offline_data_quality=$3 \
    --save_replay_buffer=False \
    --num_episodes_collected=4000 \
    --stop_winrate=$4 \
    --seed=1 \
    --g=$5
    # --config=$1:选择scr.algs/<algs_name>.yaml
    # 几个用于online的baseline算法:VDN,QMIX,QTRAN,COMA,MADDPG,MATD3
    # --env-config=$2:选择scr.envs/<env_name>.yaml
    # <map_name>
    # --offline_data_quality=数据质量{expert, medium, random, full}
    # random: 直接跳过online训练,采用random_mac收集数据
    # medium: 会使用stop_winrate/stop_return作为提前训练结束标志
    # expert: 不会参考stop_winrate/stop_return,会一直训练到num_episodes_collected
    # full: 保存完整的训练过程,并为最终策略增加final标签识别
    # --save_replay_buffer=False: 是否保存replay_buffer
    # --stop_winrate=$4: 停止训练的胜率
    