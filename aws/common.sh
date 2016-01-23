cp -r /data/deep_q_rl /root
cd /root/deep_q_rl
git pull
cd deep_q_rl

SAVE_PATH=/data/logs
mkdir -p $SAVE_PATH

COMMON_PARAMS="--save-path=$SAVE_PATH"
