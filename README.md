This is the source code of the paper: Hongxu Chen, Hongzhi Yin, Xiangguo Sun, Tong Chen, Bogdan Gabrys and Katarzyna Musial. Multi-level Graph Convolutional Networks for Cross-platform Anchor Link Prediction. 26th ACM SIGKDD Conference On Knowledge Discovery and Data Mining (KDD'20), San Diego, USA. August, 2020


To run the source code:

- Step1: use 1.preprocess.py to get graph partition, and other necessary variables.
- Step2: use 2.model.py to learn node embeddings of each partition.
- Step3: use 3.1 and 3.2 to match embeddings between partitions and networks
- Step4: use 4.anchor_predict.py to predict anchor links

Prerequisite 

```
pytorch
python-louvain
networkx
```



Please cite our paper if you use this code

citation:

```bibtex
@inproceedings{chen2020multi,
  title={Multi-level Graph Convolutional Networks for Cross-platform Anchor Link Prediction},
  author={Chen, Hongxu and Yin, Hongzhi and Sun, Xiangguo and Chen, Tong and Gabrys, Bogdan and Musial, Katarzyna},
  booktitle={26th ACM SIGKDD Conference On Knowledge Discovery and Data Mining (KDD'20)},
  year={2020}
}

```
