# DGRec

A PyTorch and DGL implementation for the WSDM 2023 paper below:  
[DGRec: Graph Neural Network for Recommendation with Diversified Embedding Generation](https://arxiv.org/pdf/2211.10486.pdf)

Environment
DGL version 1.0.1
Pytorch version 1.12.1

## Running
``python main.py``  
Then you can get similar result on TaoBao dataset as illustrated in the paper.  

You can check different hyper-parameters in `utils/parser.py`

## Citation
If you use our code, please cite the paper below:
```bibtex
@inproceedings{yang2023dgrec,
  title={DGRec: Graph Neural Network for Recommendation with Diversified Embedding Generation},
  author={Yang, Liangwei and Wang, Shengjie and Tao, Yunzhe and Sun, Jiankai and Liu, Xiaolong and Yu, Philip S and Wang, Taiqing},
  booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  pages={661--669},
  year={2023}
}
```
