import torch
import torchvision.models as models
import torch.nn as nn


CLASSES = [
    "AMARELA",
    "NO_SIGN",
    "R-1",
    "R-12",
    "R-15",
    "R-19",
    "R-2",
    "R-24a",
    "R-24b",
    "R-25a",
    "R-25b",
    "R-25c",
    "R-25d",
    "R-26",
    "R-27",
    "R-28",
    "R-29",
    "R-3",
    "R-33",
    "R-34",
    "R-4a",
    "R-4b",
    "R-5",
    "R-6a",
    "R-6b",
    "R-6c",
    "R-7",
    "R-9",
    "SETAS",
    "VERDE",
    "VERMELHA",
]

print(f"numero de classes: {len(CLASSES)}")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES)) 

checkpoint = torch.load("classifier.pth", map_location='cuda')
model.load_state_dict(checkpoint)
model.cuda()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).cuda()

torch.onnx.export(
    model, 
    dummy_input, 
    "classifier.onnx",
    verbose=False,
    opset_version=11,
    input_names=['input_0'],
    output_names=['output_0'],
    dynamic_axes={'input_0': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}} 
)
