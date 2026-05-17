import torch
from torchvision import models
import onnx
import onnxruntime as ort
import numpy as np

# 1. 加载模型
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# 2. 创建示例输入
# 创建一个假的输入，形状为 (batch=1, 通道=3, 高=224, 宽=224)，里面的值是随机数。
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print("✅ 导出完成：resnet50.onnx")

# 4. 验证 ONNX 结构
onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")

# 5. 用 ONNX Runtime 推理并对比结果
# 创建一个 ONNX Runtime 的推理会话，可以理解为“用 ONNX Runtime 加载模型”。
ort_session = ort.InferenceSession("resnet50.onnx")

# 使用相同的 dummy_input 做推理
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)[0]

# PyTorch 推理
with torch.no_grad():
    torch_output = model(dummy_input).numpy()

# 对比差异
max_diff = np.abs(torch_output - ort_outputs).max()
print(f"✅ PyTorch 与 ONNX Runtime 输出最大差异：{max_diff:.6e}")

if max_diff < 1e-5:
    print("🎉 导出成功，推理结果完全一致！")
else:
    print("⚠️ 差异较大，请检查导出设置")