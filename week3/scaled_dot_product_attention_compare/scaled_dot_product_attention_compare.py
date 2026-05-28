import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import time

# ---------- 手写实现（修正 softmax dim） ----------
def scaled_dot_product_attention_manual(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == True, -1e9)
    attn = torch.softmax(scores, dim=-1)   # 注意 dim=-1
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn, value)
    return output, attn

# ---------- 测试函数 ----------
def benchmark_attention(impl, q, k, v, mask=None, name="", num_warmup=10, num_iter=100):
    # 同步并清空缓存
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 预热
    for _ in range(num_warmup):
        out, _ = impl(q, k, v, mask)
    torch.cuda.synchronize()
    
    # 记录峰值显存（在运行正式迭代前重置）
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iter):
        out, _ = impl(q, k, v, mask)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event) / num_iter  # 平均每次耗时 (ms)
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    print(f"{name:20s} 平均时间: {elapsed_ms:.4f} ms, 峰值显存: {peak_memory:.2f} MB")
    return elapsed_ms, peak_memory

# ---------- 测试配置 ----------
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试参数 (可以修改)
    batch_size = 2
    num_heads = 8
    seq_len_q = 128
    seq_len_k = 128
    d_k = 64
    d_v = 64
    use_mask = True
    
    # 生成随机输入 (已投影并拆分多头的形状)
    q = torch.randn(batch_size, num_heads, seq_len_q, d_k, device=device)
    k = torch.randn(batch_size, num_heads, seq_len_k, d_k, device=device)
    v = torch.randn(batch_size, num_heads, seq_len_k, d_v, device=device)
    
    mask = None
    if use_mask and seq_len_q == seq_len_k:
        # 生成因果掩码 (下三角)
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=device), diagonal=1).bool()
        # 扩展维度到 (1, 1, seq_len_q, seq_len_k) 以便广播
        mask = mask.unsqueeze(0).unsqueeze(0)
    
    # 手动实现
    benchmark_attention(scaled_dot_product_attention_manual, q, k, v, mask, name="手动实现")
    
    # 官方实现 (注意：官方函数要求输入形状为 (batch, seq_len, embed_dim) 或 (batch, num_heads, seq_len, head_dim)?)
    # 官方 F.scaled_dot_product_attention 接受 3D 或 4D 输入。4D 时要求 (batch, num_heads, seq_len, head_dim)
    # 我们直接使用相同的输入形状，只需将 mask 的形状调整为官方期望的格式（官方 mask 可以是 2D 或 4D，支持广播）
    def official_impl(q, k, v, mask):
        # 官方函数如果传入 4D 张量，会自动将后两维视为 (seq_len, head_dim)
        # 因果掩码推荐使用 is_causal=True 参数，而不是手动传 mask
        if mask is not None and seq_len_q == seq_len_k:
            # 使用 is_causal 代替手动掩码，性能更好
            return F.scaled_dot_product_attention(q, k, v, is_causal=True), None
        else:
            return F.scaled_dot_product_attention(q, k, v), None
    
    benchmark_attention(official_impl, q, k, v, mask, name="官方实现 (is_causal)")