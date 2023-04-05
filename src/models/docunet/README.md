# DocUNet

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