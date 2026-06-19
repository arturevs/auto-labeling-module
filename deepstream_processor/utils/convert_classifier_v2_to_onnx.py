"""
Converte best_model.pt (SmallTrafficSignCNN) para classifier.onnx
compatível com o nvinfer do DeepStream (config_infer.txt do traffic_sign_classifier_v2).
"""
import torch
import torch.nn as nn


class SmallTrafficSignCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()

        def make_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            make_block(3, 32),
            make_block(32, 64),
            make_block(64, 128),
            make_block(128, 256),
            make_block(256, 384),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    CHECKPOINT = "detection_models/traffic_sign_classifier_v2/best_model.pt"
    OUTPUT_ONNX = "detection_models/traffic_sign_classifier_v2/classifier.onnx"
    INPUT_SIZE = 224  # deve bater com infer-dims no config_infer.txt

    print(f"Carregando checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")

    num_classes = len(ckpt["class_to_idx"])
    print(f"Classes detectadas no checkpoint: {num_classes}")
    print(f"Mapeamento: {ckpt['class_to_idx']}")

    model = SmallTrafficSignCNN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Verificação rápida: forward pass deve funcionar sem erros
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Forward pass OK — output shape: {out.shape}")

    print(f"Exportando ONNX para: {OUTPUT_ONNX}")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX,
        verbose=False,
        opset_version=11,
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={
            "input_0": {0: "batch_size"},
            "output_0": {0: "batch_size"},
        },
    )
    print("Exportação concluída com sucesso.")


if __name__ == "__main__":
    main()
