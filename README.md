# GFDN

Submission of SIGMOD 2023

## QueryOPT
The project that help us implement (alpha, beta)-core is [QueryOPT](https://github.com/boge-liu/alpha-beta-core).

## Quick Start

1. download dataset from the [competition](https://tianchi.aliyun.com/dataset/dataDetail?dataId=123862)
2. Pre-process: ``python dataset/get_data.py``
3. Install [swig](https://github.com/swig/)
4. Build pyabcore: ``cd ./queryopt && ./build.sh && cd ..``
5. Start:``python main.py``

## Note
The last four entrys of customer vertex in the dataset are noise and have been ignored.
