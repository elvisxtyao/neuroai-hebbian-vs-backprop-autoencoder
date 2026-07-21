# Phase 0-v1 公共骨架完成报告

> 后续状态（2026-07-20）：本文记录的是 Phase 0 骨架交付时点。显式
> WTA/Oja Hebbian 更新、逐层训练、冻结 decoder、linear probe 和 seed-0
> 正式运行现已完成；最新状态与结果见 `HEBBIAN_IMPLEMENTATION_REPORT.md`。

日期：2026-07-20
工作区：`D:\Microlearning`
状态：**公共骨架已完成并通过实际测试**

## 1. 交付结论

已完成一套可供 BP 与 Hebbian 共用的 `phase0-v1` 基础框架。两条学习规则共用同一份数据划分、模型 forward、初始化、linear probe、表示提取和结果 schema，只通过 `training.learning_rule` 切换 trainer。

本次交付没有用占位 Hebbian 算法冒充正式实现。BP trainer 可运行；Hebbian trainer 保留显式 `compute_local_update/apply_local_update` 实现入口，在正式局部规则完成前会明确拒绝运行。

## 2. 已冻结的基础 settings

- MNIST：官方 training 60k 分层划分为 train 50k / validation 10k；官方 test 10k；
- split seed：`0`；paired model seeds：`[0,1,2,3,4]`；tuning seed：`42`；
- 输入：`ToTensor()`，严格位于 `[0,1]`，不做 z-score；
- batch size：`128`，`num_workers=0`，不丢弃最后一个 batch；
- Encoder：`1→16→32→64` 三层 Conv，输出 `64×1×1`；
- Decoder：三层 ConvTranspose，最终 Sigmoid 输出 `1×28×28`；
- 无 BatchNorm、Dropout、pooling、target clamping；
- Reconstruction：pixel-mean MSE；
- BP：Adam `lr=1e-3`，10 epochs；
- Linear probe：train-feature standardization + 单层 Linear；encoder 全程冻结；
- Probe：SGD `lr=0.1, momentum=0.9, weight_decay=1e-4`，30 epochs；
- 正式入口：validated YAML + Python module CLI；notebook 不作为正式结果入口。

可直接发送给队友的完整标准见 `PHASE0_STANDARD_V1.md`。

## 3. 已完成模块

| 模块 | 完成内容 |
|---|---|
| Config | YAML inheritance、required-field validation、跨规则一致性约束 |
| Data | MNIST 下载、稳定 sample ID、分层 split、manifest 校验、DataLoader |
| Model | 共享 ConvEncoder/ConvDecoder/ConvAutoencoder、逐层表示接口 |
| Initialization | paired deterministic initialization、state checksum |
| BP | 共享模型上的 MSE/Adam trainer、validation checkpoint |
| Hebbian hook | 独立 trainer 接口；禁止 custom-autograd/未验证占位更新 |
| Probe | train-only feature standardization、单层分类器、冻结校验 |
| Evaluation | h1/h2/z 对齐提取、accuracy/macro-F1/CE |
| Results | resolved config、metadata、统一 metrics.csv、非覆盖 run 目录 |
| Tests | shape、参数量、split、配置、复现、freeze、无标签泄漏 |

## 4. 实际验证结果

### 环境

```text
Python: 3.11
torch: 2.13.0+cpu
torchvision: 0.28.0+cpu
numpy: 2.4.6
pytest: 8.4.2
```

依赖安装在项目隔离环境 `.venv`，不会依赖系统 Python。

### 自动测试

```text
11 passed
```

覆盖：

- h1/h2/z/reconstruction shape；
- 标准参数量；
- 相同 seed 初始化相同、不同 seed 初始化不同；
- probe step 不改变 encoder；
- 分层 split 可复现且无重叠；
- BP/Hebbian 配置共用 data/model；
- representation trainer 不接收 label；
- target clamping 在两种规则中均关闭。

### Synthetic BP smoke test

```text
h1: 16×16×14×14
h2: 16×32×7×7
z:  16×64×1×1
BP MSE: 0.08318465 → 0.08290488 → 0.08275205
paired model hash: d5c874a63a2a9144304fd979764ba9b32d9a68f0fc93f1a0c903178d69e19c89
```

Smoke loss 有限且连续下降，输出范围保持在 `[0,1]`。

### 真实 MNIST 数据门禁

```text
train / validation / test = 50000 / 10000 / 10000
train-validation overlap = 0
split seed = 0
train-label checksum = e474323ac02fc156161b0bbb1f8c9cfeb327a74bcd9f9680a8385269a24f566c
```

三个 DataLoader 均验证：

```text
batch shape = 128×1×28×28
pixel range = [0.0, 1.0]
batch sample IDs unique = 128
```

## 5. 队友使用方式

安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

验证：

```powershell
python -m pytest
python -m scripts.smoke_test --config configs/bp_main.yaml
```

BP 训练与 probe：

```powershell
python -m training.train_representation --config configs/bp_main.yaml
python -m training.train_linear_probe --config configs/bp_main.yaml --run-dir results/<run-id>
```

队友不应复制模型 class；应在共享模型上使用 `learning_rule=bp`，并在提交说明中写明 `phase0-v1 compliant`。

## 6. 明确未包含的后续工作

以下内容不属于公共骨架完成条件，因此本次没有伪装成已完成：

- 正式 Hebbian local update、WTA/Oja 和 greedy layer-wise training；
- 完整10-epoch BP baseline 数值结果；
- 五个 paired seeds 正式实验；
- noise、representation geometry、architecture sweep；
- BP reference、cosine、bias 和 SNR 分析。

下一开发步骤应是实现并单测显式 Hebbian layer，而 BP 队友可同时在当前骨架上运行正式 baseline。
