# SSAN-Adapt

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
