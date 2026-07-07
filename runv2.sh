# road_boundary 数据集的配置文件
# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4  ./configs/topoboundary.sh >> A_S2_topodoundary.log 2>&1

# crowdai 建筑物数据集的配置文件
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4  ./configs/crowdai.sh > A_S5_corwdai.log 2>&1  

# SN3 道路数据集的配置文件
# CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4  ./configs/SN3/deformable_dert_sn3_twostage_multscale.sh > A_S2_sn3_buildvfinal2_dataaugv6_losschange.log 2>&1