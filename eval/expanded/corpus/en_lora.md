# Low-Rank Adaptation of Model Weights

Low-rank adaptation, commonly called LoRA, adapts a pretrained model through a constrained update to selected weight matrices. For a frozen matrix W with shape output_dimension by input_dimension, the update can be represented as a product BA. A has shape rank by input_dimension, and B has shape output_dimension by rank. The trainable parameter count for this update is rank times the sum of input_dimension and output_dimension.

The forward operation uses the original matrix plus a scaled low-rank update. A common convention scales BA by alpha divided by rank. The base matrix remains frozen while the adapter matrices receive gradients. Keeping a weight frozen prevents updates to that weight, but training can still require activations and gradient propagation through surrounding operations. Freezing weights does not make training memory-free.

The rank is a capacity choice, not the number of model layers or training examples. A higher rank can express a broader class of updates but also increases trainable parameters and associated optimizer state. Which layers receive adapters is another independent choice. Applying adapters to attention projections is common, but LoRA is not restricted to the query and value projections in principle.

For compatible unquantized linear layers, the learned low-rank update can be multiplied out and merged into the base weight for inference. Merging must use the same scaling convention as training. Keeping adapters separate instead allows multiple task-specific adapters to share one base model. An adapter by itself generally does not contain all the weights needed to run the base model.

LoRA and quantization address different aspects of resource use. LoRA reduces the number of trained update parameters; quantization reduces numerical precision of stored values. QLoRA combines low-rank adaptation with a quantized frozen base model and additional implementation techniques. It should not be described as ordinary LoRA automatically converting every activation or optimizer state to low precision.
