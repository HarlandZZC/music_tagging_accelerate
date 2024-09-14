# Music Tagging Distributed Training with Accelerate

This repository contains an example of distributed training for music tagging using the Accelerate framework. It demonstrates how to set up multi-machine, multi-GPU distributed training.

## Setup Instructions

Please follow the steps below on **both machines** to set up and run the training process:

1. Clone the repository:

    ```bash
    git clone git@github.com:HarlandZZC/music_tagging_accelerate.git
    cd music_tagging_accelerate
    ```

2. Set up the environment:

    ```bash
    conda env create -f music_tagging_env.yaml
    ```

3. Start training by running:

   ```bash
   bash train.sh
   ```

4. To evaluate the training results, run:

   ```bash
   python test.py
   ```

Please note that `test.py` only runs on a single GPU on a single machine. For more information on how to use the Accelerate framework, please refer to [https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index).
