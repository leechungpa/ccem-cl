<h1 align="center"> <p>Analysis of Using Sigmoid Loss for Contrastive Learning</p></h1>
<h4 align="center">
    <p>Chungpa Lee, Joonhwan Chang, Jy-yong Sohn</p>
</h4>
<p align="center">
    <a href="https://proceedings.mlr.press/v238/lee24a.html">
        <img alt="paper" src="https://img.shields.io/badge/Paper-blue.svg">
    </a>
    <a href="https://arxiv.org/abs/2402.12613">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-b31b1b.svg">
    </a>
</p>


Contrastive learning has emerged as a prominent branch of self-supervised learning for several years. Especially, CLIP, which applies contrastive learning to large sets of captioned images, has garnered significant attention. Recently, SigLIP, a variant of CLIP, has been proposed, which uses the sigmoid loss instead of the standard InfoNCE loss. SigLIP achieves the performance comparable to CLIP in a more efficient manner by eliminating the need for a global view. However, theoretical understanding of using the sigmoid loss in contrastive learning is underexplored. In this paper, we provide a theoretical analysis of using the sigmoid loss in contrastive learning, in the perspective of the geometric structure of learned embeddings. First, we propose **the double-Constant Embedding Model (CCEM)**, a framework for parameterizing various well-known embedding structures by a single variable. Interestingly, the proposed CCEM is proven to contain the optimal embedding with respect to the sigmoid loss. Second, we mathematically analyze the optimal embedding minimizing the sigmoid loss for contrastive learning. The optimal embedding ranges from simplex equiangular-tight-frame to antipodal structure, depending on the temperature parameter used in the sigmoid loss. Third, our experimental results on synthetic datasets coincide with the theoretical results on the optimal embedding structures.

## Experimental results

<p align="center">
  <img src="https://raw.githubusercontent.com/leechungpa/ccem-cl/main/pic.png" width="500">
</p>

The normalized similarity $s$ of positive pairs measured for the embeddings trained by sigmoid loss $\mathcal{L}^{\text{sig}}$, for various $N$ and $t$ when $d=N$. We train a encoder (two-layer fully-connected ReLU network) which outputs embeddings, rather than directly optimizing the embedding vectors.

The code example is available in [the Jupyter Notebook](https://github.com/leechungpa/ccem-cl/blob/main/example.ipynb).

## Citation
```tex
@InProceedings{lee2024analysis,
  title = {Analysis of Using Sigmoid Loss for Contrastive Learning},
  author = {Lee, Chungpa and Chang, Joonhwan and Sohn, Jy-yong},
  booktitle = {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  year = {2024},
  url = {https://proceedings.mlr.press/v238/lee24a.html}
}
```
