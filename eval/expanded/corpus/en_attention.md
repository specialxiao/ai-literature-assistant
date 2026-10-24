# Scaled Dot-Product Attention

Scaled dot-product attention transforms a query matrix Q, a key matrix K, and a value matrix V into context-dependent outputs. Its core expression is softmax(QK^T / sqrt(d_k))V, where d_k is the width of each key vector. The softmax operates across the candidate key positions for each query. The resulting weights form a weighted combination of value vectors, not a selection of the key vectors themselves.

The square-root divisor controls the magnitude of attention logits. If independent query and key components have roughly unit variance, the variance of their dot product grows with d_k. Dividing the dot product by sqrt(d_k) keeps its variance roughly constant under those assumptions. Without scaling, large logits can push softmax toward saturated distributions and make optimization less stable. This argument describes a useful initialization heuristic, not a guarantee that learned activations always have unit variance.

A causal mask prevents a token from attending to future positions during autoregressive training. Disallowed logits receive negative infinity before softmax, making their attention weights zero. A padding mask instead suppresses artificial padding positions; it does not impose chronological order. These masks solve different problems and can be combined.

Multi-head attention applies separate learned projections before computing several attention operations. Their outputs are concatenated and projected back into the model representation. Multiple heads provide distinct learned representation subspaces; they do not automatically guarantee distinct linguistic roles.

Ordinary dense self-attention materializes interactions among all token pairs. Its attention-score matrix therefore contains a quadratic number of entries in sequence length. Memory-efficient attention kernels can avoid storing that entire matrix without changing the underlying dense attention operation. They should not be confused with sparse attention, which deliberately omits some token interactions.
