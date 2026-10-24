# Reciprocal Rank Fusion for Retrieval

Reciprocal rank fusion, or RRF, combines ranked result lists without requiring their underlying scores to share a scale. This is useful when one retriever emits lexical relevance scores and another emits cosine similarities. Raw addition would mix quantities whose numerical ranges and calibration may differ across queries and collections.

For a document d, basic RRF sums 1 divided by (c plus rank_i(d)) over the input lists containing d. Ranks start at one in this convention, and c is a positive smoothing constant. A document absent from an input list contributes zero for that list. The constant c is not the number of results requested; it controls the relative influence of very high ranks compared with deeper ranks. Increasing c makes differences between nearby ranks less pronounced.

Before fusion, the application needs a consistent document identity. Two entries referring to the same passage should accumulate contributions under the same identifier. Two different passages from the same paper should remain distinct if the retrieval unit is a passage. Deduplicating merely by paper title could discard useful supporting evidence from another page.

RRF rewards agreement, but it does not assess factual correctness or reading comprehension. A passage ranked reasonably well by two complementary retrievers can outrank a passage appearing only in one list. However, if both retrievers share the same blind spot, fusion cannot invent an omitted relevant passage. Candidate-list depth therefore affects which passages are eligible for the final ranking.

Fusion and reranking are different stages. A learned reranker can read a query together with each candidate passage and assign a new relevance score. RRF only uses rank positions from its input lists. An application may fuse first and rerank later, but a fused score is neither a calibrated probability of relevance nor a measurement of whether a generated answer is supported.
