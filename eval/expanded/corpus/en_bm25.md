# BM25 Term Saturation and Length Normalization

BM25 is a lexical ranking family that combines term rarity with a saturating contribution from term frequency. It matches terms represented by an analyzer or tokenizer. It does not independently learn that two unrelated surface forms are synonyms. The analyzer is therefore part of the retrieval design, especially for Chinese text, identifiers, and mixed-language documents.

A common term-frequency factor is tf times (k1 plus 1), divided by tf plus k1 times (1 minus b plus b times document_length divided by average_document_length). Here tf counts occurrences of a query term in the document. The exact inverse-document-frequency formula and handling of query term frequency vary across implementations, so numerical scores should be interpreted within a particular implementation and corpus.

The parameter b controls document-length normalization. Setting b to zero removes the document-length adjustment from the term-frequency factor. Setting b to one applies the full document-length ratio in that adjustment. For a fixed positive term frequency and b greater than zero, a document longer than the corpus average receives a smaller term contribution than an otherwise equivalent average-length document. This is not a blanket rule that short documents always outrank long ones, because other matched terms and their frequencies still contribute.

The parameter k1 controls how slowly additional occurrences saturate. With a positive k1, repeated occurrences increase the term contribution with diminishing returns. Raising k1 generally makes the score more sensitive to repetition; it does not introduce semantic matching. Document length and average document length must use the same tokenization convention.

Chunking changes BM25 statistics. Splitting a paper into short passages changes length distributions and document frequencies, so scores from a paper-level index are not directly comparable to scores from a passage-level index. A cutoff calibrated on one collection should be reevaluated after substantial changes to the corpus or chunking policy.
