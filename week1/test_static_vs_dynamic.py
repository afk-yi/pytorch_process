import torch
from torchvision import models
import onnxruntime as ort
import numpy as np

# 加载一个简单模型（这里用 ResNet50，但为了快速演示，可以换成更小的模型）
model = models.resnet50(weights=None)  # 不加载权重，只取结构
model.eval()

# 创建一个 dummy 输入，batch=1
dummy_input = torch.randn(1, 3, 224, 224)

# ===== 1. 导出静态轴模型（batch 固定 = 1）=====
torch.onnx.export(
    model,
    dummy_input,
    "static_batch.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,           # 全部静态
    opset_version=11
)
print("✅ 静态模型导出: static_batch.onnx")

# ===== 2. 导出动态轴模型（batch 可变）=====
torch.onnx.export(
    model,
    dummy_input,
    "dynamic_batch.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print("✅ 动态模型导出: dynamic_batch.onnx")

# ===== 3. 测试静态模型：输入 batch=1（应该成功）=====
sess_static = ort.InferenceSession("static_batch.onnx")
input_batch1 = np.random.randn(1, 3, 224, 224).astype(np.float32)
output_static_1 = sess_static.run(None, {"input": input_batch1})
print("✅ 静态模型，batch=1：推理成功")

# ===== 4. 测试静态模型：输入 batch=2（会报错）=====
input_batch2 = np.random.randn(2, 3, 224, 224).astype(np.float32)
try:
    output_static_2 = sess_static.run(None, {"input": input_batch2})
    print("❌ 静态模型，batch=2：居然没有报错？（不应该）")
except Exception as e:
    print("❌ 静态模型，batch=2：报错如下（预期）")
    print(f"   错误信息: {e}")

# ===== 5. 测试动态模型：输入 batch=2（应该成功）=====
sess_dynamic = ort.InferenceSession("dynamic_batch.onnx")
output_dynamic_2 = sess_dynamic.run(None, {"input": input_batch2})
print("✅ 动态模型，batch=2：推理成功")