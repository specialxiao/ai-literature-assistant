# Position Information and Rotary Embeddings

Self-attention alone has no built-in notion of token order when inputs are permuted consistently and no positional restrictions are applied. A language model therefore needs positional information or an ordering mechanism. Absolute positional embeddings add a position-dependent vector to each token representation. Learned absolute embeddings usually associate a learned table entry with a position index. Sinusoidal absolute encodings instead construct entries from fixed sine and cosine functions.

Rotary position embedding, abbreviated RoPE, takes a different approach. RoPE rotates pairs of query and key coordinates by angles determined by token position. Values are not rotated by the basic RoPE operation. Each coordinate pair uses a frequency, so different pairs encode position at different angular scales. The rotation preserves the Euclidean norm of the rotated vector.

The important relationship appears inside a query-key dot product. With matching rotation frequencies, the dot product between a query rotated at position m and a key rotated at position n depends on the relative offset n minus m. This creates a relative-position dependence without storing a separate learned embedding for every relative offset. It does not mean that the entire network becomes independent of absolute position: masks, boundaries, data, and other components still matter.

RoPE is not the same as adding a relative attention bias. An additive bias modifies an attention logit directly, whereas RoPE changes query and key vectors before their dot product. Likewise, a causal mask controls which positions are legal, rather than encoding distance among legal positions.

Longer inference contexts require care. A rotary formula can be evaluated at positions beyond the training range, but this alone does not guarantee reliable extrapolation. Changing rotary frequencies or position scaling changes how distances are represented and should be evaluated on long-context tasks. A model accepting a longer sequence is not evidence that it retrieves or reasons over that sequence accurately.
