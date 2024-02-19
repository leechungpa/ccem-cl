# Analysis of Using Sigmoid Loss for Contrastive Learning

Chungpa Lee, Joonhwan Chang, [Jy-yong Sohn](https://scholar.google.co.kr/citations?hl=en&user=Cs75s1MAAAAJ&view_op=list_works&sortby=pubdate)

[Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS) 2024, Valencia, Spain](https://aistats.org/aistats2024/). PMLR: Volume 238.

## Abstract

> Contrastive learning has emerged as a prominent branch of self-supervised learning for several years. Especially, CLIP, which applies contrastive learning to large sets of captioned images, has garnered significant attention. Recently, SigLIP, a variant of CLIP, has been proposed, which uses the sigmoid loss instead of the standard InfoNCE loss. SigLIP achieves the performance comparable to CLIP in a more efficient manner by eliminating the need for a global view. However, theoretical understanding of using the sigmoid loss in contrastive learning is underexplored. In this paper, we provide a theoretical analysis of using the sigmoid loss in contrastive learning, in the perspective of the geometric structure of learned embeddings. First, we propose **the double-Constant Embedding Model (CCEM)**, a framework for parameterizing various well-known embedding structures by a single variable. Interestingly, the proposed CCEM is proven to contain the optimal embedding with respect to the sigmoid loss. Second, we mathematically analyze the optimal embedding minimizing the sigmoid loss for contrastive learning. The optimal embedding ranges from simplex equiangular-tight-frame to antipodal structure, depending on the temperature parameter used in the sigmoid loss. Third, our experimental results on synthetic datasets coincide with the theoretical results on the optimal embedding structures.

## Experimental results

<p align="center">
  <img src="https://raw.githubusercontent.com/leechungpa/ccem-cl/main/pic.png" width="500">
</p>

The normalized similarity $s = \frac{1}{2} (1 + \frac{1}{N} \sum_{i=1}^{N} \mathbf{u}_i^\top\mathbf{v}_i)$ of positive pairs measured for the embeddings trained by sigmoid loss $\mathcal{L}^{\text{sig}}$, for various $N$ and $t$ when $d=N$. We train a encoder (two-layer fully-connected ReLU network) which outputs embeddings, rather than directly optimizing the embedding vectors.

The code example is available in [the Jupyter Notebook](https://github.com/leechungpa/ccem-cl/blob/main/example.ipynb).
