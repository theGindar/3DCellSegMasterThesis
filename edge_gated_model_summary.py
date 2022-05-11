from func.network import CellSegNet_basic_edge_gated, CellSegNet_basic_lite, CellSegNet_basic_edge_gated_II
from torchinfo import summary

model=CellSegNet_basic_lite(input_channel=1, n_classes=3, output_func = "softmax")
print(summary(model, (1, 1, 64, 64, 64)))

model=CellSegNet_basic_edge_gated(input_channel=1, n_classes=3, output_func = "softmax")
print(summary(model, (1, 1, 64, 64, 64)))

model=CellSegNet_basic_edge_gated_II(input_channel=1, n_classes=3, output_func = "softmax")
print(summary(model, (1, 1, 64, 64, 64)))

