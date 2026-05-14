import torch

scalar = torch.tensor(3.14)
print("\nscalar:", scalar)

vec = torch.tensor([1,2,3])
print("\nvec:", vec)

mat = torch.tensor([[1,2],[3,4]])
print("\nmat:", mat)

tensor_3d = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print("\ntensor_3d:", tensor_3d)

#特殊值张量
zeros = torch.zeros(3,4)
ones = torch.ones(2,3)
eye = torch.eye(4)
full = torch.full((2,2),8)

print("\nzeros:", zeros)
print("\nones:", ones)
print("\neye:", eye)
print("\nfull:", full)

# 随机张量
rand_uniform = torch.rand(3, 3)      # 均匀分布 [0, 1)
print("\nrand_uniform:", rand_uniform)
rand_normal = torch.randn(3, 3)      # 标准正态分布
print("\nrand_normal:", rand_normal)
rand_int = torch.randint(0, 100, (3, 3))  # [0,100) 的随机整数
print("\nrand_int:", rand_int)

# 序列张量
arange = torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
print("\narange:", arange)
linspace = torch.linspace(0, 1, 5)   # [0.0, 0.25, 0.5, 0.75, 1.0]
print("\nlinspace:", linspace)

# "Like"类函数 复用形状
x = torch.rand(2, 3)
zeros_like = torch.zeros_like(x)   # 形状与 x 相同
print("\nzeros_like:", zeros_like)
ones_like = torch.ones_like(x)
print("\nones_like:", ones_like)

# 查看Tensor属性
x = torch.rand(2, 3)
print("\nx.shape:", x.shape)        # torch.Size([2, 3])
print("x.dtype:", x.dtype)        # torch.float32
print("x.device:", x.device)       # cpu
print("x.numel():", x.numel())      # 元素总数 = 6

# 基础运算
# 算术运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法
print("\na + b:", a + b)
print("torch.add(a, b):", torch.add(a, b))

# 减法
print("\nb - a:", b - a)

# 逐元素乘法
print("\na * b:", a * b)

# 矩阵乘法（dot product）
dot = torch.dot(a, b)   # 1*4 + 2*5 + 3*6 = 32
print("\ndot:", dot)

# 矩阵乘法（二维）
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
C = torch.matmul(A, B)   # 或 A @ B
print("\nC (matmul):", C)


# 原地运算
a = torch.tensor([1, 2, 3])
a.add_(5)        # a 变成 [6, 7, 8]，注意下划线
print("\na (after add_):", a)


# 形状变化
t = torch.arange(12)      # [0,1,...,11]
t_reshaped = t.reshape(3, 4)   # 变成 3x4 矩阵
print("\nt_reshaped:", t_reshaped)

# view 要求内存连续，通常先用 reshape 更安全
t_view = t.view(2, 6)
print("\nt_view:", t_view)

# 重排维度
x = torch.rand(2, 3, 4)     # 形状 (2,3,4)
x_perm = x.permute(2, 1, 0) # 变成 (4,3,2)
print("\nx_perm.shape:", x_perm.shape)

# 交换维度
x = torch.rand(2, 3)
x_t = x.transpose(0, 1)     # 变成 3x2
print("\nx_t:", x_t)

# 展平
x = torch.rand(2, 3, 4)
flat = x.flatten()           # 变成 1维，长度 24
print("\nflat:", flat)


# GPU操作，把张量移到GPU（如可用）
if torch.cuda.is_available():
    x_cpu = torch.rand(3, 3)
    x_gpu = x_cpu.cuda()          # 移到 GPU
    # 或 x_gpu = torch.rand(3, 3).cuda()
    print("\nx_gpu.device:", x_gpu.device)           # cuda:0
    # 移回 CPU
    x_back = x_gpu.cpu()
    print("\nx_back (cpu):", x_back)



# 1.创建一个 3×3 的全1矩阵（torch.float32）。
ones_3d = torch.ones(3, 3)
print("\nones_3d:", ones_3d)

# 2.创建一个 5×5 的随机整数矩阵，值的范围 0~100（使用 torch.randint）。
rand_5d = torch.randint(0, 100, (5,5))
print("\nrand_5d:", rand_5d)

# 3.将第1步的全1矩阵原地乘以 2（mul_），并打印结果。
ones_3d_mul = ones_3d.mul_(2)
print("\nones_3d_mul (after mul_):", ones_3d_mul)

# 4.计算修改后的矩阵所有元素的和（.sum()）。
mul_sums = ones_3d_mul.sum()
print("\nmul_sums:", mul_sums)

# 5.创建一个形状为 (2, 3, 4) 的随机张量 t，用 permute 将其变换为 (4, 3, 2)，并打印变换后的形状。
t = torch.rand(2, 3, 4)
t_perm = t.permute(2, 1, 0)
print("\nt.shape:", t.shape)
print("\nt_perm.shape:", t_perm.shape)


