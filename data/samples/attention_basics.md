# Attention 与 Transformer 基础

Self-attention computes a representation of each token by comparing its query with the keys of all tokens and using the resulting weights to combine value vectors. Scaled dot-product attention divides the query-key dot product by the square root of the key dimension before applying softmax. The scaling prevents dot products from growing too large and pushing softmax into regions with very small gradients.

Multi-head attention projects queries, keys, and values into several learned subspaces. Different heads can capture different relationships, and their outputs are concatenated and projected again. Positional information must be added because attention alone does not encode token order.

Transformer blocks normally combine attention, a position-wise feed-forward network, residual connections, and layer normalization. The original architecture used an encoder-decoder layout; many later language models use decoder-only blocks with causal masking.

