# GFDN

[KDD 2023] Group-based Fraud Detection Network on e-Commerce Platforms

[https://doi.org/10.1145/3580305.3599836](https://doi.org/10.1145/3580305.3599836)

## QueryOPT
The project that helps us implement (alpha, beta)-core is [QueryOPT](https://github.com/boge-liu/alpha-beta-core).

## Quick Start
0. Prepare the environment:
     ```
     conda create -n gfdn python=3.8
     conda activate gfdn
     conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch
     pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu101.html && pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu101.html && pip install torch-geometric==2.0.0
     ```
1. download the dataset from the [competition](https://tianchi.aliyun.com/dataset/dataDetail?dataId=123862). Please download the final round dataset.
2. Pre-process: ``python dataset/get_data.py``
3. Install [swig](https://github.com/swig/)
4. Build pyabcore: ``cd ./queryopt && ./build.sh && cd ..``
5. Start:``python main.py``

## Note
The last four entries of customer vertex in the dataset are noise and have been ignored. Experimental results may vary slightly due to different hardware configurations.
