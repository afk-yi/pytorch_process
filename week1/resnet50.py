import torch
from torchvision import models, transforms
from PIL import Image
import requests

# --- 在这里选择权重类型 ---
# 选择 V2 版本 (推荐，性能更好)
weights = models.ResNet50_Weights.IMAGENET1K_V2
# 或者你也可以显式地指定 V1 版本
# weights = models.ResNet50_Weights.IMAGENET1K_V1

# 1. 加载模型
# 使用 weight 对象加载模型
model = models.resnet50(weights=weights)
model.eval()  # 🎯 设置为评估模式


# 2. 图片预处理
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# 直接从 weight 对象获取所需的预处理流程，这是最关键的一步！
preprocess = weights.transforms()

# 从 URL 读取一张猫图（也可替换成本地路径）
img_path = "./hajimi.png"
img = Image.open(img_path).convert('RGB')
img_tensor = preprocess(img).unsqueeze(0)  # 增加batch维度


# 3. 推理
with torch.no_grad():  # 🎯 禁用梯度计算
    output = model(img_tensor)

# 4. 处理结果
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_idx = torch.topk(probabilities, 5)

# 加载 ImageNet 类别标签
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(labels_url).text.splitlines()

print("Top 5 预测类别:")
for i in range(5):
    print(f"{labels[top5_idx[i]]}: {top5_prob[i].item():.2%}")

