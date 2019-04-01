# Cut samples into pieces of size receptive_field +
# sample_size with receptive_field overlap
"""
dilations: A list with the dilation factor for each layer.

filter_width: The samples that are included in each convolution,
    after dilating.

residual_channels: How many filters to learn for the residual.

dilation_channels: How many filters to learn for the dilated
    convolution.

skip_channels: How many filters to learn that contribute to the
    quantized softmax output.

quantization_channels: How many amplitude values to use for audio
    quantization and the corresponding one-hot encoding.
    Default: 256 (8-bit quantization).
"""

tensorshape: batch x height x width x color

receptive_field = (filter_width - 1) * sum(dilations) + 1

