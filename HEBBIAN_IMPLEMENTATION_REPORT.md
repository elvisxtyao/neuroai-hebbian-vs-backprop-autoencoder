# Hebbian 显式局部学习：实现与 seed-0 运行报告

日期：2026-07-20
公共标准：`phase0-v1`
正式运行：`results/20260720T065919Z_hebbian_seed0`
对照运行：`results/20260720T044850Z_bp_seed0`

## 1. 结论

本阶段要求已经完成：共享 3-layer convolutional autoencoder 上实现了显式
WTA/Oja 局部更新、卷积核 L2 归一化和 `enc1 -> enc2 -> enc3` greedy
逐层训练；随后冻结 Hebbian encoder，用与 BP baseline 相同的数据划分、decoder
结构和 frozen linear probe 协议完成重建与分类评估。

正式 seed-0 结果如下：

| 指标 | BP | Hebbian | Hebbian 相对 BP |
|---|---:|---:|---:|
| test reconstruction MSE | 0.003289 | 0.019896 | 约 6.05 倍，更差 |
| frozen-probe test accuracy | 91.81% | 89.00% | -2.81 percentage points |
| frozen-probe test macro-F1 | 91.67% | 88.82% | -2.85 percentage points |
| frozen-probe test CE | 0.2792 | 0.3662 | +0.0870 |

Hebbian 表示的分类性能显著高于 10% 随机猜测水平，证明 encoder 学到了有用表示；
但其重建和线性可分性均弱于 BP。逐层诊断同时发现明显的深度退化：enc3 的
active-neuron ratio 从第 1 epoch 的 84.38% 降到第 2 epoch 的 20.31%，之后保持
在该水平。这不是训练失败或 NaN，而是当前竞争规则下可复现的 winner concentration。

该结果仅为单 seed 初版本，不代表最终统计结论。decoder 和 probe 使用 BP，因此准确
表述应是 **Hebbian-trained encoder with a BP-trained decoder/linear probe**，不能称为
“纯 Hebbian 端到端 autoencoder”。

## 2. 公平比较设置

BP 与 Hebbian 运行共享以下内容：

- MNIST 固定 split：train 50,000 / validation 10,000 / test 10,000；
- split seed 与 model seed 均为 `0`；输入为 `[0,1]`，无 z-score；
- batch size `128`，`num_workers=0`；
- 同一 `ConvAutoencoder`：encoder `1 -> 16 -> 32 -> 64`，三层输出依次为
  `16x14x14`、`32x7x7`、`64x1x1`；
- latent dimension `64`；无 BatchNorm、Dropout、pooling 或 target clamping；
- 相同 decoder class、MSE 定义、linear probe、特征标准化、SGD 和评估指标；
- frozen linear probe 的输入均为展平后的 `z`，不会反向修改 encoder。

两次运行的初始完整模型 hash 均为
`d5c874a63a2a9144304fd979764ba9b32d9a68f0fc93f1a0c903178d69e19c89`，审计结果
为完全匹配。

两条路径的必要差异是 encoder 学习协议：BP baseline 对 autoencoder 联合训练 10
epochs；Hebbian encoder 每层训练 10 epochs，共 30 次 dataset passes，再冻结 encoder
并用 BP 单独训练 decoder 10 epochs。因此不能只按名义 epoch 比较两者的学习速度。

## 3. 显式 WTA/Oja 更新

实现位于 `learning_rules/hebbian.py`，没有使用 custom autograd，也没有把标签传入
encoder 学习路径。对某个卷积层，先计算：

```text
y = ReLU(Conv(x; W))
```

对每个 sample 和空间位置，在 output-channel 维选择 top-k winner。当前
`winner_fraction=0.20`，`k = ceil(0.20 * out_channels)`。用 mask `M` 得到：

```text
y_wta = y * M
```

将输入转换为与卷积感受野一致的 patch 后，每个输出 filter 的候选更新为：

```text
Delta W_o = mean(y_wta,o * x_patch)
            - mean(y_wta,o^2) * W_o
```

第一项是局部 Hebbian correlation，第二项是 Oja stabilization。候选更新由
`compute_local_update()` 在 `torch.no_grad()` 下返回，不修改 layer；随后
`apply_local_update()` 执行：

```text
W <- W + learning_rate * Delta W
W_o <- W_o / max(||W_o||_2, epsilon)
```

所以每次更新后每个 filter 的 L2 norm 保持为 1。正式配置使用逐层学习率：

| layer | learning rate | epochs | winner fraction |
|---|---:|---:|---:|
| enc1 | 5e-4 | 10 | 0.20 |
| enc2 | 1e-4 | 10 | 0.20 |
| enc3 | 5e-5 | 10 | 0.20 |

逐层训练时只更新 active layer：训练 enc2 时 enc1 已冻结，训练 enc3 时 enc1/enc2
均冻结。每层结束均保存独立 checkpoint。

## 4. 稳定性 pilot 与配置变更记录

最初 pilot 使用 `winner_fraction=0.10` 和三层统一 `lr=1e-3`。enc2 第 3 epoch
出现 `active_neuron_ratio=0.21875`、`winner_entropy=0.4003`，因此主动中止，未将
该 run 用于模型比较。中止记录位于
`results/20260720T065348Z_hebbian_seed0/PILOT_ABORTED.md`。

之后只根据训练集机制诊断将 winner fraction 提高至 0.20，并随深度降低学习率。
一轮逐层 pilot 的 active ratios 为 `1.0 / 1.0 / 0.9844`，随后才启动正式 10-epoch
逐层运行。正式运行显示该修改延缓了塌缩，但没有解决 enc3 在多 epoch 下的竞争集中。

## 5. Hebbian encoder 逐层输出

以下数值均来自正式运行的 `metrics.csv`。`update norm` 是应用学习率前的候选更新
范数；`sparsity` 是 ReLU 后非正激活比例；winner entropy 已按通道数归一化到 `[0,1]`。

| layer | epoch | update norm | activation mean | sparsity | active ratio | winner entropy |
|---|---:|---:|---:|---:|---:|---:|
| enc1 | 1 | 0.5475 | 0.0686 | 0.8517 | 1.0000 | 0.8247 |
| enc1 | 5 | 0.6092 | 0.0869 | 0.8420 | 1.0000 | 0.7815 |
| enc1 | 10 | 0.5630 | 0.1072 | 0.8359 | 1.0000 | 0.7466 |
| enc2 | 1 | 5.3227 | 0.1184 | 0.7195 | 1.0000 | 0.8468 |
| enc2 | 5 | 14.7782 | 0.3624 | 0.6944 | 0.9688 | 0.6681 |
| enc2 | 10 | 4.5396 | 0.5089 | 0.6920 | 0.4688 | 0.5711 |
| enc3 | 1 | 900.8724 | 11.1838 | 0.4406 | 0.8438 | 0.6277 |
| enc3 | 2 | 638.2307 | 11.7706 | 0.4393 | 0.2031 | 0.6167 |
| enc3 | 5 | 658.4578 | 11.7705 | 0.4393 | 0.2031 | 0.6167 |
| enc3 | 10 | 637.9235 | 11.7705 | 0.4393 | 0.2031 | 0.6167 |

三层的 `weight_norm_mean` 在所有记录点均为 `1.0`，证明逐 filter L2 normalization
实际生效。不同层 update norm 不应直接横向解释为“学习更快”，因为层尺寸、输入激活
尺度和学习率均不同；Q4 的正式比较仍需使用 norm ratio、cosine/alignment、bias 和 SNR。

## 6. Frozen decoder 重建结果

Hebbian encoder 完成后 checksum 为：

```text
d3a7462868374f96ce8b08ed289ed12cd6e3f05f2fe7ec4da1eb7169267957b7
```

decoder 训练前后 checksum 完全相同，代码也进行了强制断言。validation MSE 输出：

| decoder epoch | validation MSE |
|---:|---:|
| 1 | 0.040496 |
| 2 | 0.030249 |
| 3 | 0.027122 |
| 4 | 0.024615 |
| 5 | 0.023619 |
| 6 | 0.022672 |
| 7 | 0.022111 |
| 8 | 0.021318 |
| 9 | 0.020709 |
| 10 | 0.020278 |

loss 单调下降；最佳 validation checkpoint 为 epoch 10。最终 test pixel-mean MSE 为
`0.019896`。重建不是均值图或常数输出，多数数字仍可辨认，但比 BP 明显更模糊、局部
笔画更容易失真。

## 7. Frozen linear probe 结果

probe 训练 30 个固定 epochs，只根据 validation accuracy 选 checkpoint。最佳选择为
epoch 12：

| split | selected epoch | CE | accuracy | macro-F1 |
|---|---:|---:|---:|---:|
| validation | 12 | 0.3851 | 88.47% | 88.28% |
| test | 12 | 0.3662 | 89.00% | 88.82% |

encoder 在 probe 前后由 checksum 检查保持不变。完整 30-epoch validation 曲线保存在
`metrics.csv`；测试集只在选定 validation checkpoint 后进行最终评估。

## 8. 测试与实现门禁

执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

最终结果：`14 passed`。

已覆盖的 Hebbian 专项测试包括：

- 小张量 Oja 候选更新与手算结果一致；
- `compute_local_update()` 不修改权重；
- `apply_local_update()` 后每个 filter 的 L2 norm 为 1；
- greedy epoch 只改变 active layer；
- encoder 参数不需要 autograd gradient；
- 原有 shared-model shape、split、freeze 和 reproducibility 测试继续通过。

## 9. 产物清单

正式 run 目录保存：

- `config_resolved.yaml`：完整解析后的配置；
- `metadata.json`：Python/PyTorch/device、参数量、shape、初始 hash；
- `encoder_enc1_end.pt`、`encoder_enc2_end.pt`、`encoder_enc3_end.pt`：逐层快照；
- `encoder_hebbian.pt`：最终 Hebbian encoder；
- `model_best.pt`、`model_last.pt`：带 decoder 的 autoencoder checkpoint；
- `linear_probe.pt`：最佳 frozen linear probe；
- `metrics.csv`：30 个 encoder epochs、20 个 decoder train/validation rows、
  reconstruction test、30 个 probe validation rows 和最终 validation/test；
- `hebbian_training_summary.json`：逐层末期诊断和 checksum；
- `reconstructions_original_then_reconstructed.png`：原图/重建图；
- `hebbian_training_diagnostics.png`：逐层机制与 decoder 曲线；
- `bp_hebbian_seed0_comparison.png`：同协议 BP/Hebbian 对比图。

## 10. 可复现命令

```powershell
.\.venv\Scripts\python.exe -m training.train_representation --config configs/hebbian_main.yaml
.\.venv\Scripts\python.exe -m evaluation.evaluate_reconstruction --config configs/hebbian_main.yaml --run-dir results/<hebbian-run-id>
.\.venv\Scripts\python.exe -m training.train_linear_probe --config configs/hebbian_main.yaml --run-dir results/<hebbian-run-id>
.\.venv\Scripts\python.exe -m evaluation.plot_run_metrics --hebbian-run results/<hebbian-run-id> --bp-run results/<bp-run-id>
```

probe rerun会先删除旧的 `linear_probe`/`linear_probe_final` rows，再写入完整新结果；
encoder、decoder 和 reconstruction 历史不会被删除。这样终端中断后重跑不会留下重复行。

## 11. 当前完成边界与下一步

本次已经完成单 seed、单架构的 Hebbian 初版本闭环，但以下项目仍属于后续研究，不应
写成已完成：

1. 使用 tuning seed 42、只看 validation 的正式 LR/winner-fraction 搜索；
2. paired seeds `[0,1,2,3,4]` 与均值、95% confidence interval；
3. random encoder control，确认相对随机表示的增益；
4. latent geometry、LDA/separability、effective rank 与 per-layer representation；
5. noise、non-stationary data、latent-dimension 和 architecture-asymmetry sweep；
6. frozen snapshot 上同 batch 的 BP reference、cosine/alignment、scale-matched bias 与
   Hebbian/BP 独立 SNR；
7. 针对 enc3 winner concentration 的 homeostasis、adaptive competition 或第二种
   Hebbian variant 消融。

最优先的下一项是第 6 项的 update analysis 与第 3 项 random encoder control；它们分别
直接回答 Q4，并判断当前 89% 是否真正来自 Hebbian 学习而非卷积随机特征。之后再启动
多 seed 主实验。
