# defenses: Byzantine-Robust FL Defense Analysis

## 概述

本模块基于每轮、每客户端的 **Euclidean distance** 和 **cosine similarity** 指标，对联邦学习中的 Byzantine 攻击（如 ALIE、Gaussian RMP）进行检测，并计算检测率。

## 重要说明

**标准 Krum**（Blanchard et al., NeurIPS 2017）需要**客户端更新之间的成对距离** `||Δ_i - Δ_j||`。本模块提供的是：

- 每个客户端到加权平均的 Euclidean distance：`||Δ_i - Δ_g||`
- 每个客户端到加权平均的 cosine similarity：`cos(Δ_i, Δ_g)`

因此无法直接计算 Krum 所需的成对距离。本模块提供的是：

1. **Krum-inspired 启发式**：基于距离与均值的相对关系作为代理
2. **ALIE 启发式**：攻击者通常具有**低距离**和**高 cosine**（统计上更“合理”）
3. **多种检测方法**：low_distance、high_cosine、combined、krum_proxy、MAD

## 使用方法

### 1. 直接运行（使用内置 ALIE 数据）

```bash
python defenses/krum_defense_analysis.py
```

### 2. 在代码中调用

```python
from defenses.krum_defense_analysis import (
    run_defense_analysis,
    print_analysis_report,
    parse_metric_table,
)

# 从文本或文件读取
euclidean_text = open("euclidean_distances.txt").read()
cosine_text = open("cosine_similarities.txt").read()

analysis = run_defense_analysis(
    euclidean_text,
    cosine_text,
    attacker_ids=[5, 6],  # 可选，若未提供则从表头解析
    methods=["combined", "krum_proxy", "low_distance", "high_cosine", "mad"]
)
print_analysis_report(analysis)
```

### 3. 数据格式

表格格式示例：

```
Round | Client0(B) | Client1(B) | Client2(B) | Client3(B) | Client4(B) | Client5(A) | Client6(A) | Mean | Std
--------------------------------------------------------------------------------
1      | 1.310655       | 1.275939       | 1.267773       | ...
```

- `(B)` 表示 benign 客户端
- `(A)` 表示 attacker 客户端
- 每行：Round 号 + 各客户端数值 + Mean + Std

## 检测方法说明

| 方法 | 说明 |
|------|------|
| `low_distance` | 标记距离低于 percentile 的客户端（ALIE 攻击者距离更小） |
| `high_cosine` | 标记 cosine 高于 percentile 的客户端（ALIE 攻击者 cosine 更高） |
| `combined` | 同时满足：低距离 + 高 cosine（ALIE 典型模式） |
| `krum_proxy` | 标记距离最小的 k 个客户端（k = 攻击者数量） |
| `mad` | 基于 Median Absolute Deviation 的异常值检测 |

## 输出指标

- **Attack Presence Detection Rate**：检测到至少一个攻击者的轮次占比
- **Attacker Identification Rate**：识别出的攻击者占比（TP / 总攻击者-轮次）
- **Precision / Recall / F1**：攻击者识别任务的精确率、召回率、F1

## 可调参数

```python
run_defense_analysis(
    euclidean_text, cosine_text,
    dist_percentile=25.0,   # low_distance 阈值
    cos_percentile=75.0,    # high_cosine 阈值
    num_byzantine=2,       # 攻击者数量（krum_proxy）
)
```
