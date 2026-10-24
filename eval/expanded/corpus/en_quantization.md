# Quantization Scales and Numerical Error

Quantization represents numerical values using a smaller set of discrete levels. In affine integer quantization, a real value x is mapped approximately by q equals round(x divided by scale) plus zero_point, followed by clipping to the supported integer range. Dequantization reconstructs x_hat as scale times (q minus zero_point). The scale is positive, and the zero point shifts the representable interval relative to real zero.

Quantization error comes from rounding and clipping. Rounding introduces a difference between a value and its nearest representable level. Clipping occurs when a value falls outside the chosen range and is replaced by an endpoint. A range wide enough to include extreme outliers can produce coarse spacing for the many smaller values. Choosing a narrower range can improve resolution near common values but clip those outliers.

Per-tensor quantization shares one set of quantization parameters across a tensor. Per-channel quantization gives selected channels their own scales, which can better accommodate channels with different value ranges. The channel axis must match the operator and implementation; it is not automatically the sequence dimension of a language model.

Post-training quantization operates on a pretrained model without ordinary full retraining. Some methods collect activation statistics from calibration examples, while certain weight-only methods do not need representative activation samples. Quantization-aware training simulates quantization effects during training so parameters can adapt. These labels describe workflows, not a universal promise about accuracy preservation.

Weight-only quantization reduces storage precision for model weights, but it does not automatically quantize the key-value cache or all activations. Lower bit width also does not guarantee proportional end-to-end speedup. Runtime kernels, dequantization overhead, memory bandwidth, batch size, and hardware support determine actual latency. Evaluation should separately measure model-quality changes, peak memory, and application latency rather than inferring all three from the nominal weight bit width.
