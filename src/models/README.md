# 1. SSAN-Adapt

### Abstract

Entities, as the essential elements in relation extraction tasks, exhibit certain structure. In this work, we formulate such structure as
distinctive dependencies between mention pairs. We then propose SSAN, which incorporates these structural dependencies within the standard
self-attention mechanism and throughout the overall encoding stage. Specifically, we design two alternative transformation modules inside
each self-attention building block to produce attentive biases so as to adaptively regularize its attention flow. Our experiments
demonstrate the usefulness of the proposed entity structure and the effectiveness of **SSAN**. It significantly outperforms competitive
baselines, achieving new state-of-the-art results on three popular document-level relation extraction datasets. We further provide ablation
and visualization to show how the entity structure guides the model for better relation extraction. Our code is publicly available.

For the more details please refer to:
* [SSAN-Adapt on arXiv](https://arxiv.org/abs/2102.10249v1)
* [SSAN-Adapt on GitHub](https://github.com/BenfengXu/SSAN)
* [SSAN-Adapt on GitHub (PaddlePaddle)](https://github.com/PaddlePaddle/Research)
* [SSAN-Adapt on paperwithcode.com](https://paperswithcode.com/paper/entity-structure-within-and-throughout)

### Global rank on April 2023

| Dataset | Metric Name | Metric Value | Global Rank |
|---------|-------------|--------------|-------------|
| DocRED  | F1          | 65.92        | 3           |
| CDR     | F1          | 68.7         | 7           |
| GDA     | F1          | 83.9         | 3           |

Results are obtained from [paperwithcode.com](https://paperswithcode.com/paper/entity-structure-within-and-throughout)

---

# 2. DocUNet

### Abstract

Document-level relation extraction aims to extract relations among multiple entity pairs from a document. Previously proposed graph-based
or transformer-based models utilize the entities independently, regardless of global information among relational triples. This paper 
approaches the problem by predicting an entity-level relation matrix to capture local and global information, parallel to the semantic
segmentation task in computer vision. Herein, we propose a Document U-shaped Network for document-level relation extraction. Specifically,
we leverage an encoder module to capture the context information of entities and a U-shaped segmentation module over the image-style
feature map to capture global interdependency among triples. Experimental results show that our approach can obtain state-of-the-art
performance on three benchmark datasets DocRED, CDR, and GDA.

For the more details, please refer to:
* [DocUNet on arXiv](https://arxiv.org/abs/2106.03618v2)
* [DocUNet on GitHub](https://github.com/zjunlp/DocuNet)
* [DocUNet on paperswithcode.com](https://paperswithcode.com/paper/document-level-relation-extraction-as)


### Global rank on April 2023

| Dataset | Metric Name | Metric Value | Global Rank |
|---------|-------------|--------------|-------------|
| DocRED  | F1          | 64.55        | 6           |
| CDR     | F1          | 76.3         | 4           |
| GDA     | F1          | 85.3         | 4           |

Results are obtained from [paperwithcode.com](https://paperswithcode.com/paper/document-level-relation-extraction-as)
