# On Alpha:
export NCCL_SOCKET_IFNAME=enp3s0f0 
accelerate launch --config_file ./default_config.yaml train.py  --root /datasets/gtzan
# On Beta: 
# export NCCL_SOCKET_IFNAME=enp3s0f0np0
# accelerate launch --config_file ./default_config.yaml train.py  --root /lan/datasetsAlpha4090/gtzan