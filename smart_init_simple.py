"""
智能初始化 - 优化版

改动点：
1. resize_token_embeddings 传入 mean_resizing=False，避免无用的随机初始化
2. 正确处理权重共享（tie_word_embeddings）
3. 增加已存在 token 的统计日志
"""

import torch
from typing import Dict, Tuple


def add_discrete_tokens(tokenizer, model, local_rank=0):
    """
    添加离散化数值 Token（共 3000 个）并进行智能初始化
    """

    # ========== Step 1: 获取现有数字 token 的嵌入 ==========
    old_embeddings = model.get_input_embeddings().weight.data
    embedding_dim = old_embeddings.shape[1]
    old_vocab_size = old_embeddings.shape[0]

    digit_embeddings = {}

    for d in range(10):
        for token_str in [str(d), f" {d}"]:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) == 1:
                digit_embeddings[str(d)] = old_embeddings[token_ids[0]].clone()
                break

    for symbol in ['.', '-', '+']:
        for token_str in [symbol, f" {symbol}"]:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) == 1:
                digit_embeddings[symbol] = old_embeddings[token_ids[0]].clone()
                break

    if local_rank == 0:
        print(f"[Smart Init] Found digit embeddings: {list(digit_embeddings.keys())}")

    # ========== Step 2: 生成新 token 并添加到 tokenizer ==========
    int_tokens = [f"<s{s}i{i:03d}>" for s in [0, 1] for i in range(1000)]
    frac_tokens = [f"<d{i:03d}>" for i in range(1000)]
    new_tokens = int_tokens + frac_tokens

    num_added = tokenizer.add_tokens(new_tokens)
    num_already_exist = len(new_tokens) - num_added

    if local_rank == 0:
        print(f"[Smart Init] Added {num_added} new tokens, {num_already_exist} already existed")
        print(f"[Smart Init] All {len(new_tokens)} tokens will be smart-initialized regardless")
        print(f"[Smart Init] Vocab size: {old_vocab_size} -> {len(tokenizer)}")

    # ========== Step 3: 调整模型嵌入层大小 ==========
    # ★ 优化：mean_resizing=False，跳过无用的随机初始化（反正后面会覆盖）
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # ========== Step 4: 智能初始化新 token 嵌入 ==========
    new_embeddings = model.get_input_embeddings().weight.data

    old_std = old_embeddings.std(dim=0)

    # 初始化 INT tokens
    for token_str in int_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            continue

        sign = 1 if token_str[2] == '0' else -1
        i_value = int(token_str[4:7])
        int_part = i_value // 10
        frac_first = i_value % 10
        coarse_value = sign * (int_part + frac_first / 10.0)

        emb = torch.zeros(embedding_dim, device=new_embeddings.device, dtype=new_embeddings.dtype)
        weight_sum = 0.0

        if sign == -1 and '-' in digit_embeddings:
            emb += 0.5 * digit_embeddings['-'].to(emb.device).to(emb.dtype)
            weight_sum += 0.5

        int_digits = ['0'] if int_part == 0 else list(str(int_part))
        for i, d in enumerate(int_digits):
            w = 1.0 / (i + 1)
            if d in digit_embeddings:
                emb += w * digit_embeddings[d].to(emb.device).to(emb.dtype)
                weight_sum += w

        if '.' in digit_embeddings:
            emb += 0.3 * digit_embeddings['.'].to(emb.device).to(emb.dtype)
            weight_sum += 0.3

        if str(frac_first) in digit_embeddings:
            emb += 0.5 * digit_embeddings[str(frac_first)].to(emb.device).to(emb.dtype)
            weight_sum += 0.5

        if weight_sum > 0:
            emb /= weight_sum

        normalized_value = coarse_value / 100.0
        emb += 0.1 * normalized_value * old_std.to(emb.device).to(emb.dtype)

        new_embeddings[token_id] = emb

    # 初始化 FRAC tokens
    for token_str in frac_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id == tokenizer.unk_token_id:
            continue

        d_value = int(token_str[2:5])
        fine_value = d_value / 10000.0

        d_digits = [str((d_value // 100) % 10),
                    str((d_value // 10) % 10),
                    str(d_value % 10)]

        emb = torch.zeros(embedding_dim, device=new_embeddings.device, dtype=new_embeddings.dtype)
        weight_sum = 0.0

        if '.' in digit_embeddings:
            emb += 0.4 * digit_embeddings['.'].to(emb.device).to(emb.dtype)
            weight_sum += 0.4

        if '0' in digit_embeddings:
            emb += 0.2 * digit_embeddings['0'].to(emb.device).to(emb.dtype)
            weight_sum += 0.2

        for i, d in enumerate(d_digits):
            w = 0.4 / (i + 1)
            if d in digit_embeddings:
                emb += w * digit_embeddings[d].to(emb.device).to(emb.dtype)
                weight_sum += w

        if weight_sum > 0:
            emb /= weight_sum

        normalized_value = fine_value * 10
        emb += 0.05 * normalized_value * old_std.to(emb.device).to(emb.dtype)

        new_embeddings[token_id] = emb

    # ========== Step 5: 初始化 lm_head ==========
    # ★ 优化：正确处理权重共享
    is_tied = getattr(model.config, 'tie_word_embeddings', False)

    if hasattr(model, 'lm_head'):
        lm_head_weight = model.lm_head.weight.data

        if lm_head_weight.data_ptr() == new_embeddings.data_ptr():
            if local_rank == 0:
                print("[Smart Init] Weight tying active: embed_tokens and lm_head share memory")
        else:
            # 不共享时，手动复制
            for token_str in int_tokens + frac_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                if token_id != tokenizer.unk_token_id:
                    lm_head_weight[token_id] = new_embeddings[token_id].clone()

            if local_rank == 0:
                print("[Smart Init] Initialized lm_head separately (not tied)")

    # ========== Step 6: 验证初始化质量 ==========
    if local_rank == 0:
        print(f"\n[Smart Init] Verification:")

        test_tokens = ["<s0i000>", "<s0i123>", "<s1i050>", "<d000>", "<d456>"]
        for token_str in test_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            norm = new_embeddings[token_id].norm().item()
            print(f"  {token_str}: ||emb||={norm:.4f}")

        print(f"\n[Smart Init] Cosine similarity for nearby values:")
        pairs = [
            ("<s0i100>", "<s0i101>", "10.0 vs 10.1"),
            ("<s0i100>", "<s0i200>", "10.0 vs 20.0"),
            ("<s0i050>", "<s1i050>", "+5.0 vs -5.0"),
        ]
        for t1, t2, desc in pairs:
            id1 = tokenizer.convert_tokens_to_ids(t1)
            id2 = tokenizer.convert_tokens_to_ids(t2)
            cos_sim = torch.nn.functional.cosine_similarity(
                new_embeddings[id1].unsqueeze(0).float(),
                new_embeddings[id2].unsqueeze(0).float()
            ).item()
            print(f"  {desc}: cos_sim={cos_sim:.4f}")

        # ★ 新增：打印权重共享状态
        print(f"\n[Smart Init] Model config tie_word_embeddings={is_tied}")

    return tokenizer, model