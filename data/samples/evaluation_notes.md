# RAG 评测笔记

检索阶段可以使用 Recall@k 和 Mean Reciprocal Rank（MRR）。Recall@k 检查前 k 个结果是否包含至少一个标注相关证据；MRR 关注第一个相关结果的排名，越靠前得分越高。延迟也应记录平均值和 P95，避免只看质量而忽略交互体验。

生成阶段不能只看语言是否流畅。人工评测应检查答案正确性、关键信息完整性、每个引用是否真实支持相邻结论，以及资料无答案时是否正确拒答。LLM-as-judge 可以辅助扩大规模，但不应该成为作品集中的唯一评价依据。

一个实用小型评测集可由三到五篇公开论文和二十到三十个问题组成，其中至少五个问题应当无法从资料回答。A small practical evaluation set should contain 20 to 30 questions drawn from three to five public papers. 固定数据版本、检索参数和模型名，才能让不同实验可复现。

