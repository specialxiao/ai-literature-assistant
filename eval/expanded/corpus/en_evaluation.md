# Leakage-Safe Retrieval Evaluation

Retrieval evaluation starts with an information need, a fixed corpus, and relevance judgments for retrieval units such as pages or passages. A test question must have its supporting evidence inside the indexed corpus if it is labeled answerable. If annotations name pages but retrieval returns chunks, the scorer must explicitly map chunks to pages and deduplicate repeated hits to avoid counting the same evidence several times.

Development data is used to choose thresholds, routing rules, and other settings. The test split is held out until those choices are fixed. Repeatedly changing a similarity cutoff after inspecting test failures makes the resulting test score optimistic. Reporting a held-out label is insufficient if the answers were used to tune the system indirectly.

Near-duplicate questions should be grouped before splitting. Otherwise, a paraphrase of a test question can appear in development data and make the split misleadingly easy. When the intended claim concerns unseen documents, splitting only questions from the same documents is not enough; document-level grouping is needed. A same-corpus question split can still be useful, but its generalization claim is narrower.

Recall at k measures how much annotated relevant evidence appears among the first k results. Reciprocal rank for a question is one divided by the rank of its first relevant result, or zero if none appears. Mean reciprocal rank averages that value across questions. Neither metric checks whether a generated answer faithfully uses the retrieved evidence. Answer faithfulness requires a separate judgment or evaluation procedure.

Unanswerable questions probe refusal and false-positive retrieval behavior. Their no-hit rate measures how often the retriever returns no evidence under its acceptance rule, not whether the language model refuses correctly. Synthetic questions can verify expected behavior and expose regressions, but a high score on authored notes should not be described as a real-world paper benchmark. Report corpus provenance, sample counts, split policy, and metric definitions alongside the results.
