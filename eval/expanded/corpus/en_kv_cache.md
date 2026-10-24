# Key-Value Caching in Autoregressive Decoding

An autoregressive transformer generates a continuation one token at a time. During ordinary causal self-attention inference, previously processed tokens do not gain access to future tokens. Their per-layer key and value vectors can therefore be reused when a new token arrives. A key-value cache stores those vectors rather than recomputing them for every decoding step.

Prefill processes the prompt and initializes the cache. Decode steps then compute representations for new tokens and attend to cached positions. Caching avoids repeatedly recomputing old key and value projections, but each new query still attends over the applicable cached context. A cache does not make the cost of reading a growing context disappear.

A basic storage estimate is two times batch_size times layer_count times sequence_length times kv_head_count times head_dimension times bytes_per_element. The factor two accounts for keys and values. This estimate excludes allocator overhead, padding, temporary tensors, and implementation-specific metadata. The number of query heads should not replace kv_head_count in models that share keys and values across query heads.

Multi-query attention shares one key head and one value head across all query heads. Grouped-query attention uses an intermediate number of key-value heads. Both reduce cache size relative to otherwise comparable multi-head attention, which assigns separate key-value heads to query heads. They are architectural choices and are not identical to compressing an already constructed cache.

Paged cache management allocates and maps cache blocks to reduce waste from variable sequence lengths and to support serving workloads. Paging changes storage management, not the semantic contents of the attention vectors by itself. Cache quantization separately reduces storage precision and may affect numerical behavior. Prefix sharing can reuse a common cached prefix across requests only when model, tokenization, positions, and execution semantics are compatible; matching visible text alone is not a sufficient cache key.
