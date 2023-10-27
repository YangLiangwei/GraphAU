# Official Implementation of GraphAU.  

An official Pytorch and DGL implementation for the CIKM 2023 paper (Best Short Paper Honorable Mention) below:
[Graph-based Alignment and Uniformity for Recommendation](https://arxiv.org/abs/2308.09292)
  

* How to use:  
    python main.py --dataset amazon-office --model graphau --lr 0.1 --weight_decay 1e-6 --layers 2 --gamma_au 0.4 --alpha_n 0.1
  
    python main.py --dataset amazon-toys --model graphau --lr 0.1 --weight_decay 0.0 --layers 3 --gamma_au 0.4 --alpha_n 0.1
  
    python main.py --dataset amazon-beauty --model graphau --lr 0.1 --weight_decay 0.0 --layers 2 --gamma_au 0.4 --alpha_n 0.1
  
    python main.py --dataset gowalla --model graphau --lr 0.1 --weight_decay 0.0 --layers 2 --gamma_au 1.7 --alpha_n 0.1
    
    dataset currently support amazon-office amazon-beauty amazon-toys gowalla

* If you use this code, please add the following citation:

``````bibtex
@inproceedings{yang2023graph,
  title={Graph-based Alignment and Uniformity for Recommendation},
  author={Yang, Liangwei and Liu, Zhiwei and Wang, Chen and Yang, Mingdai and Liu, Xiaolong and Ma, Jing and Yu, Philip S},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={4395--4399},
  year={2023}
}

``````

