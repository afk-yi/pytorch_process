# 自动求导
import torch

# 创建一个需要求导的张量
x = torch.tensor(2.0, requires_grad=True)

# 定义一个函数 y = x^2
y = x ** 2
print('\ny:', y)

# 自动计算导数 dy/dx
y.backward()

# 打印梯度
print('\nx.grad:', x.grad)   # 输出: tensor(4.0)  因为 2*x = 4


# 向量求导
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()   # y = x1^2 + x2^2 + x3^2
y.backward()
print(x.grad)        # 输出: tensor([2., 4., 6.])


# 理解计算图
a = torch.tensor(3.0, requires_grad=True)
b = a * 2
c = b ** 3
c.backward()
print(a.grad)   # dc/da = 3*(2a)^2 * 2 = 24*a^2 = 24*9 = 216


# 阻止梯度追踪 （评估模型时常用）
x = torch.tensor(5.0, requires_grad=True)
with torch.no_grad():
    y = x ** 2
print(y.requires_grad)   # False
