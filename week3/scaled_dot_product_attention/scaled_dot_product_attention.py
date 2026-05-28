import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        dropout: nn.Dropout = None
):
    """
    实现缩放点积注意力机制
    参数：
        query: 查询张量，形状 [批次大小, 多头数, 查询序列长度, 键维度]
        key: 键张量，形状 [批次大小, 多头数, 键序列长度, 键维度]
        value: 值张量，形状 [批次大小, 多头数, 值序列长度, 值维度]
        mask: 掩码张量，支持广播适配分数矩阵，True 代表需要屏蔽的位置
        dropout: 可选dropout层，用于注意力权重正则化
    返回：
        output: 注意力输出特征
        attention_weights: 注意力权重矩阵
    """
    # 获取Q/K/V 向量维度
    d_k = query.shape[-1]

    # 计算Q与K转置的点积 将key的最后两个维度转置
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # 缩放
    attention_scores = attention_scores / math.sqrt(d_k)

    # 应用掩码 
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == True, -1e9)

    print("attention_scores stats:", attention_scores.min(), attention_scores.max(), attention_scores.mean())

    # 归一化
    attention_weights = torch.softmax(attention_scores, dim = -1)

    # 可选dropout 正则化
    # 通常在训练时启用，推理时关闭
    # 即随机将一部分权重置为 0，并以 1/(1-p) 的因子缩放，以防止过拟合
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # 加权求和
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


if __name__ == "__main__":
    batch_size_test = 2
    num_heads_test = 8
    # q k 长度不一致：测试解码器第二个注意力子层的普遍情况
    seq_len_q_test = 5
    seq_len_k_test = 7
    d_k_test = 64
    d_v_test = 64

    dummy_q = torch.randn(batch_size_test, num_heads_test, seq_len_q_test, d_k_test)
    dummy_k = torch.randn(batch_size_test, num_heads_test, seq_len_k_test, d_k_test)
    dummy_v = torch.randn(batch_size_test, num_heads_test, seq_len_k_test, d_v_test)

    print("--- 无掩码测试 ---")
    output_no_mask, weights_no_mask = scaled_dot_product_attention(dummy_q, dummy_k, dummy_v)
    print(f"输出形状: {output_no_mask.shape}")
    print(f"权重形状: {weights_no_mask.shape}")
    sum_weights = weights_no_mask.sum(dim=-1)
    max_diff = (sum_weights - 1).abs().max()
    print(f"最大误差: {max_diff.item()}")
    assert torch.allclose(weights_no_mask.sum(dim=-1), torch.ones_like(weights_no_mask.sum(dim=-1)), atol=1e-6)
    print("无掩码测试通过")

    print("\n--- 前瞻掩码测试 ---")
    mask_size = seq_len_q_test
    look_ahead_mask = torch.triu(torch.ones(mask_size, mask_size), diagonal=1).bool()
    dummy_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)

    dummy_k_masked = torch.randn(batch_size_test, num_heads_test, seq_len_q_test, d_k_test)
    dummy_v_masked = torch.randn(batch_size_test, num_heads_test, seq_len_q_test, d_v_test)

    output_masked, weights_masked = scaled_dot_product_attention(dummy_q, dummy_k_masked, dummy_v_masked, mask=dummy_mask)
    print(f"掩码输出形状: {output_masked.shape}")
    assert torch.all(weights_masked.masked_select(dummy_mask) == 0)
    print("掩码测试通过")


