# On Alpha:
export NCCL_SOCKET_IFNAME=enp3s0f0 
NCCL_DEBUG=INFO accelerate launch --config_file ./default_config.yaml test.py  --root /datasets/gtzan
# On Beta: 
# export NCCL_SOCKET_IFNAME=enp3s0f0np0
# NCCL_DEBUG=INFO accelerate launch --config_file ./default_config.yaml test.py  --root /lan/datasetsAlpha4090/gtzan