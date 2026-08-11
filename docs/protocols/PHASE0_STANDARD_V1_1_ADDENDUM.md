# Phase 0 v1.1 正式实验附录

状态：冻结

生效日期：2026-07-23

父协议：`PHASE0_STANDARD_V1.md`（`phase0-v1`）

Canonical source ref：`phase0-v1.1-formal`

本附录只修订正式实验治理和复现门禁。父协议中未被明确覆盖的模型、
数据、训练预算、probe 和评估定义继续有效。发生冲突时，本附录优先。

## 1. 冻结修订

| 项目 | Phase 0 v1.1 决策 |
|---|---|
| BP learning rate | Adam `lr=0.003`、`weight_decay=0.0`；覆盖父协议中的 `lr=0.001` |
| Hebbian selected config | `lr=0.0005`、`winner_fraction=0.10`；其 representation health 尚须通过 Stage 1 |
| Tuning seed | `42`，仅用于 train/validation 选择 |
| Formal paired seeds | `[0,1,2,3,4]` |
| Formal configs | `configs/formal/bp_phase0_v1_1.yaml` 与 `configs/formal/hebbian_phase0_v1_1.yaml` |
| Test policy | validation 完成所有选择并恢复最佳 checkpoint 后，才允许一次正式 test evaluation |
| Canonical source | Git ref `phase0-v1.1-formal` 指向唯一正式源快照 |

无需重新进行 BP learning-rate tuning。现有 seed 0–1 Q1 输出以及此前
`results/` 中的运行均保持 preliminary，不得改名、复制或汇入正式统计。

## 2. 数据和复现身份

- 数据：MNIST，官方训练集分层划分为 train 50,000 / validation 10,000，
  官方 test 10,000。
- Split manifest：`data/splits/mnist_split_v1.npz`。
- Split SHA-256：
  `e7e92e0252a4ffd8b80651b9fe630f4914b563d2b6b802c0c397a8cf1c31ee54`。
- Formal seeds：`0,1,2,3,4`；每个 seed 必须完整配对 BP、Hebbian 和
  random-encoder control。
- Representation subset seed：`17`；noise seed：`2026`。
- 所有正式运行必须在 metadata 中保存 resolved-config hash、split hash、
  Git commit、环境版本、初始状态 hash、seed、rule 和 protocol block。

环境快照和确定性设置见
`environment/phase0_v1_1_environment.md`。依赖版本见
`environment/phase0_v1_1_requirements.txt`。

## 3. 初始化配对规则

对每个 formal seed：

1. BP、Hebbian 和 random control 均调用同一
   `ConvAutoencoder(latent_dim=64, seed=seed)`。
2. 三者训练前的完整 `state_dict` hash 必须相同；不相同则该 seed 立即失败。
3. Encoder hidden、decoder hidden 使用 Kaiming uniform；最终 decoder
   output 使用 Xavier uniform；全部 bias 初始化为 0。
4. BP 使用该完整初始状态进行联合 autoencoder 训练。
5. Hebbian encoder 从配对 encoder 状态开始逐层局部训练；decoder 从配对
   decoder 初始状态开始，并只在 encoder 冻结后用 BP 训练。
6. Random control 保留配对随机 encoder，不进行 representation training。
7. Linear probe 对每个 rule/seed 独立创建，但均用相同 seed 初始化、
   相同训练特征标准化、相同 SGD 配置和相同 validation checkpoint 选择规则。
8. Encoder、decoder 和 probe 的训练前/后 hash 必须写入运行证据；本应冻结的
   模块 hash 发生变化时，该运行失败。

## 4. Test-set access policy

超参数、epoch、checkpoint、collapse threshold、architecture、latent dimension
和分析选择只能依赖 train/validation。Tuning 模式不得构造、提取或记录 test
representation/metric。

正式 probe 的顺序固定为：

```text
extract train/validation
→ fit probe on train
→ choose checkpoint on validation
→ restore best validation checkpoint
→ first access to test
→ one final test evaluation
```

任何在选择结束前读取 test、根据 test 重新运行、或用 test 决定报告方案的结果
均失去 formal 资格，必须保留为 preliminary 并重新运行。

## 5. Artifact 目录和命名

```text
results/
  preliminary/                         # 开发、历史、单 seed、失败门禁
    <study>/<run-id>/
  formal/
    phase0_v1_1/
      stage1_representation_health/
      stage2_q4_seed42/
      stage3_q1/
      stage4_q4_multiseed/
      stage6_q2/
      stage7_q3/
      stage8_q5_dimension/
      stage9_q5_q6_asymmetry/
verification/
  phase0_v1_1/
    pytest_full.log
```

Formal run ID 沿用代码生成格式
`<YYYYMMDDTHHMMSSZ>_<rule>_seed<n>`（同秒冲突时追加整数 suffix）；其
formal 身份由 `results/formal/phase0_v1_1/<study>/` 父目录、resolved config
中的 `protocol.evidence_tier=formal` 和 canonical Git ref 共同确定。
Preliminary run ID 必须含 `preliminary` 或位于 `results/preliminary/`。
现有直接位于旧 `results/` 的 legacy runs 一律视为 preliminary。失败、暂停和
探索性运行不得迁移到 `results/formal/`。每个 formal run 至少包含：

```text
config_resolved.yaml
metadata.json
run_status.json
metrics.csv
resume_checkpoint.pt
checkpoints/
```

分析阶段还必须保存 sample-ID manifest、原始 CSV/JSON、图、统计摘要和生成命令。
`results/` 保持 Git ignored；正式结果通过独立 artifact manifest 追踪，不混入
source commit。

## 6. 阶段门禁

- Stage 0：本附录、formal configs、schema、环境记录、完整测试日志与 clean
  canonical source snapshot 全部一致。
- Stage 1：用固定 2,000-image validation subset 验证 expected/observed
  sparsity、dead-unit ratio、winner concentration、entropy、variance 和
  effective rank。当前 `active_neuron_ratio≈winner_fraction` 不能单独判为
  collapse。
- Stage 1B：仅在 Stage 1 FAIL 时允许；只能使用 train/validation 修复或重新选择
  Hebbian config。
- Stage 2：Stage 1 PASS 后才能执行 Q4 seed42 工具验证。
- Stage 3 及以后：Stage 0–2 全部通过后才允许生成 formal Q1、多 seed、dimension
  或 architecture runs。

Stage 0 不将任何现有结果升级为正式证据，也不回答 Q1–Q6 的最终研究结论。

## 7. Canonical snapshot 与验收

正式实验只能从 Git ref `phase0-v1.1-formal` 指向的 commit 启动。精确 commit
hash 使用以下命令解析：

```powershell
git rev-parse phase0-v1.1-formal
```

该 ref 必须指向包含本附录、formal configs、schema、状态文件和不可变测试日志
的 clean commit。完整测试日志位于
`verification/phase0_v1_1/pytest_full.log`。若 source、config 或协议发生变化，
必须发布新 addendum/ref，不能移动此 ref 后继续沿用旧正式结果。
