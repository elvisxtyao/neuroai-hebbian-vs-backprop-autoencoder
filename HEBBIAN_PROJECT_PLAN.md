# Backpropagation、Hebbian Learning 与 Minimal Hybrid Credit Assignment：执行、记录与验收计划

> 项目：Comparing Backpropagation, Hebbian Learning, and Minimal Hybrid Credit Assignment in a Three-Layer Convolutional Autoencoder
> 本文档范围：Full BP、Full Hebbian、Hybrid-HHB/Hybrid-HBB 与匹配随机控制的实现、训练、评估和机制分析
> 当前状态：`Stage 2C seed-42 diagnostic complete; Hybrid-HHB confirmation pending`
> 最后更新：2026-07-27

> 文档角色：本文件保存研究设计、公式、WBS ID 和验收条件。实时完成状态只在
> `PROJECT_STATUS.md` 维护；本文件中的长 WBS checkbox 不再作为唯一状态源。

当前交付已完成显式 convolutional WTA/Oja、逐 filter 权重归一化、三层 greedy
训练、共享 frozen linear probe、逐层诊断、BP/Hebbian 基线，以及 seed-42
Hybrid-HHB/Hybrid-HBB depth diagnostic。正式输出与未完成边界见
`PROJECT_STATUS.md`。后续主线是先确认最小 Hybrid intervention，再在冻结的
三层框架内比较 0/1/2/3 个 Hebbian encoder layers；单 seed 诊断不得替代正式
multi-seed 结论。

---

## 1. 项目目标与边界

### 1.1 核心目标

在稳定、可复现的 **3-layer convolutional autoencoder** 中比较 Full BP、
Full Hebbian 与 Minimal Hybrid Credit Assignment，并判断浅层 Hebbian
features 是否具有超越匹配随机前缀的价值。主研究轴是在总深度、参数 shape、
初始化、数据和评估协议不变时，将 encoder 中 Hebbian layers 的数量从 0
逐步增加到 3：

1. 比较分类、重建、学习速度与计算成本；
2. 比较 h1、h2 与 bottleneck z 的 latent representation；
3. 判断 BP encoder suffix 能否稳定补偿低秩 Hebbian prefix；
4. 判断一层或两层 Hebbian prefix 是否优于匹配的 frozen-random prefix；
5. 比较噪声、随机种子、latent dimension 与 architecture asymmetry 下的鲁棒性；
6. 分析局部 Hebbian update、BP gradient 及 Hybrid rule boundary 的机制差异。

三层是主实验的**固定控制框架**，不是“Hebbian 必须使用三层”的理论声明。
改变 encoder 总层数会同时改变容量、感受野与空间压缩路径，必须留到独立的
architecture-depth ablation；不能与本计划的 learning-rule allocation 主比较混用。

### 1.2 最重要的公平比较原则

> 在每个主实验中，除逐层学习规则及其必需的训练目标外，保持数据、总网络
> 结构、初始化、分类器和评估协议一致。无法匹配总 dataset passes 时，匹配
> per-layer exposure，并完整报告 total passes、samples seen 与 wall-clock。

所有差异必须写入实验配置，不能只在代码中隐含。调参只使用 validation set，test set 仅用于最终评估。

重建比较必须同时报告两种口径：

- **System reconstruction：** 使用该方法实际训练流程得到的 decoder，衡量最终
  系统性能；允许 Hybrid 的 BP suffix 与 decoder 按冻结协议联合适应。
- **Standardized decoder reconstruction：** encoder 完成后冻结全部 encoder，
  从相同 paired decoder initialization 重新训练新 decoder；所有方法使用相同
  optimizer、learning rate、数据顺序、epoch 数和 validation checkpoint selection。
  该指标用于比较 encoder representation 本身包含多少可恢复信息。

Hybrid system reconstruction 的改善可能同时来自更丰富的 z 与 BP
suffix–decoder joint adaptation，因此不能单独用它证明 representation 已修复。
涉及“encoder 可恢复信息”的主张必须以 standardized decoder reconstruction
为主要重建证据，并结合独立的 representation metrics。

### 1.3 必须修正原规划的地方

原规划适合作为开发路线，但与正式研究问题存在几处不一致。本项目采用以下修订：

- **卷积和 autoencoder 不能是 stretch goal。** MLP 可用于单元测试和快速验证，但正式主实验必须是 3 层卷积编码器，并包含明确的 decoder/reconstruction 协议。
- **明确“3-layer”的含义。** 本文默认指 encoder 有 3 个可学习卷积层，decoder 镜像地有 3 个转置卷积层；不能把整个 encoder–decoder 合计 3 层与之混用。
- **区分总深度与 Hebbian depth。** `BBB/HBB/HHB/HHH` 的 encoder 均为同一
  三层结构，只改变每层规则；不能把 HBB 误写成“一层 encoder”，也不能把
  HHB 的结果归为 Full Hebbian。
- **分类和重建分开评估。** `classification cross-entropy` 与 `reconstruction loss` 分开记录，不能统称为 final test loss。
- **BP 也必须使用 frozen encoder + 相同 linear probe。** 否则比较的是 end-to-end classifier 与 linear probe，而不是两种表示学习规则。
- **Decoder 协议必须透明。** 若 decoder 使用 BP，则结论应表述为“Hebbian-trained encoder/representation”，不能声称整个模型均由纯 Hebbian rule 训练。
- **Hybrid 声明必须准确。** “Minimal Hybrid”表示在 encoder 中加入修复
  bottleneck 所需的最小 BP credit assignment；它不是纯 biologically plausible
  model，也不能用来替代 HHH 基线。
- **Random control 必须匹配 prefix。** Full Random 只给出整体下界；判断一层
  或两层 Hebbian 是否有价值，必须分别比较 `HBB vs RBB` 和 `HHB vs RRB`。
- **Architecture asymmetry 必须可测量。** 主分析优先改变 encoder 内部通道分配；单纯改变一个在 encoder 冻结后训练的 decoder，不应改变 Hebbian latent representation。
- **Cosine similarity 不是完整的 bias。** 它称为 update alignment；偏差和 SNR 另行定义。
- **Epoch 不能作为唯一学习速度单位。** 同时报告 samples seen、wall-clock time 和 normalized AULC。

### 1.4 教程的使用范围与迁移原则

教程 notebook 只作为以下内容的**概念来源与教学原型**：

- 公共模型框架中切换不同 learning rule 的思想；
- 逐层观察并记录候选权重更新；
- 在相同模型状态上构造 BP reference；
- 使用 cosine similarity、update variance 和 SNR 分析学习机制。

正式项目不直接复制 notebook，也不把 notebook 作为组内 API、训练入口或结果来源。迁移时遵循：

| 教程内容 | 正式项目中的处理 |
|---|---|
| 同一模型比较不同 learning rule | 保留，重写为共享的 3-layer `ConvEncoder`/`Decoder` |
| custom autograd Hebbian backward | 移除，改为 `torch.no_grad()` 下的显式局部 `local_update()` |
| target clamping/label-dependent encoder activity | 从主实验移除，标签只用于 linear probe 与评估 |
| 教学用 BP reference | 重写为冻结 snapshot 上的 reconstruction-gradient reference |
| notebook 内临时数组/打印 | 改为统一的逐层 update record schema |
| notebook 数据划分 | 不继承，使用固定、可保存、无 test leakage 的 split manifest |
| 教学版 cosine/SNR | 按第 7 阶段公式重新实现并编写合成数据测试 |

教程来源、版本或文件 hash 应写入 `docs/tutorial_migration.md`。项目运行不得依赖按顺序执行 notebook cells；教程代码只有在被迁移到模块、通过测试并纳入配置后，才算正式实现。

---

## 2. 研究问题到实验的映射

| 问题 | 主实验 | 核心指标 | 最小产出 |
|---|---|---|---|
| Q1 分类与重建性能 | 固定三层架构，比较 BBB/HHH/HHB/HBB 与匹配随机控制；统一 frozen probe，并分别报告 system/standardized reconstruction | accuracy、macro-F1、CE、两种 reconstruction MSE、AULC、samples seen、wall-clock | 学习曲线、均值 ± 95% CI、预注册 paired contrasts |
| Q2 latent representation | 对固定 class-balanced subset 提取 BBB/HHH/HHB/HBB/RBB/RRB 的 h1/h2/z | linear probe、k-NN、separability、effective/stable rank、class covariance、h2→z compensation | 每层 PCA/UMAP、spectrum 与跨 seed 指标表 |
| Q3 鲁棒性 | frozen clean-trained models → matched noisy evaluation；比较 Full 与 Hybrid rules | noisy accuracy、absolute/relative degradation、representation stability、reconstruction degradation | severity curve + paired seed variance |
| Q4 权重更新 | frozen snapshots 上比较 Hebbian candidate、BP reference，以及 Hybrid rule boundary 两侧的 updates | alignment、norm ratio、alpha-star、scale-matched bias、variance、SNR | method × layer × snapshot 曲线 |
| Q5 维度与非对称 | 对 BBB/HHH/HHB/HBB 做 latent dimension 与 encoder-width sweep | accuracy、两种 reconstruction、robustness、separability、sensitivity | rule × architecture interaction 与敏感度表 |
| Q6 非对称与表示 | 对 Q5 配置重复 h1/h2/z 分析，定位 Hybrid compensation 随结构变化的位置 | CKA、linear probe、effective rank、class geometry、compensation ratio | layerwise representation comparison |

建议把 non-stationary learning 和 CIFAR-10 设为扩展实验；只有完成 MNIST 主矩阵后再开始。

---

## 3. 模型与训练协议

### 3.1 主模型

主实验使用 MNIST，输入为 `1 × 28 × 28`。所有方法共享同一个 3-layer
`ConvAutoencoder`；learning rule 按 encoder layer 配置，不能为不同规则复制
独立 forward model。推荐基线结构如下，最终以冻结的公共配置为准：

```text
x
 └─ Conv1 + activation + competition   -> h1
     └─ Conv2 + activation + competition -> h2
         └─ Conv3 + activation           -> z
             ├─ Linear probe -> class logits
             └─ Deconv1 -> Deconv2 -> Deconv3 -> x_hat
```

建议用 stride convolution 代替 max-pooling，以便 decoder 的空间尺寸可精确逆转。每一层必须记录输入/输出 shape；不得依赖手工猜测 output padding。

共享 forward API 不接收 label 或 target-clamp 参数：

```python
h1, h2, z = encoder.forward_features(x, return_all_layers=True)
x_hat = decoder(z)
```

训练规则与模型结构分离：

```python
trainer = build_trainer(
    layer_rules=["hebbian" | "bp" | "frozen_random"] * 3,
    model=model,
    config=config,
)
trainer.train_batch(x)  # 主表示学习阶段不接收 y
```

这样 learning-rule 切换不会同时改变 padding、activation、bottleneck 或 decoder shape。若教程中存在 target clamping，它只能作为明确标注的 supervised ablation，不能进入 Q1–Q6 主实验。

### 3.2 “Latent dimension”的定义

一般卷积 bottleneck 同时有 channels 与 spatial size。`phase0-v1` 使用第三层 `k=7` 将 `7×7` feature map 映射为 `L×1×1`，因此：

```text
latent_dim = C_z × 1 × 1 = C_z = L
```

主配置为 `L=64`，dimension sweep 为 `[16,32,64,128]`。若未来版本改变 spatial bottleneck，必须恢复报告 `C_z×H_z×W_z`，不得继续把 channel 数单独称为 latent dimension。

### 3.3 Hebbian encoder 的主学习规则

开发基线采用局部 competitive Hebbian/Oja-style update。对第 `l` 层卷积核，输入 patch 为 `x_l`，post-synaptic activity 为 `y_l`：

```math
\Delta W_l = \eta\, g(y_l)\left(x_l - y_l W_l\right)
```

其中 `g(y_l)` 由 WTA 或 soft competition 得到。第一版只选择一种主规则，并将其他规则作为消融实验，避免边实现边更换主定义。

必须明确并记录：

- WTA 是 per spatial location、per sample 还是 per batch；
- winner 数量或 top-k 比例；
- winner/loser 的更新系数；
- weight normalization 的轴与频率；
- bias 是否存在、如何更新；
- 更新发生在 optimizer step 前还是后；
- 深层训练是 simultaneous local update 还是 greedy layer-wise training。

推荐先采用 **greedy layer-wise training**：训练 Conv1 后冻结，生成 h1；再训练 Conv2；最后训练 Conv3。这比同时更新三层更容易定位表示塌缩和深度退化。

正式实现不通过 custom `autograd.Function.backward()` 注入 Hebbian update。每个 Hebbian layer 提供显式接口：

```python
with torch.no_grad():
    delta_w, diagnostics = layer.compute_local_update(pre_activity, post_activity)
    layer.apply_local_update(delta_w)
```

`compute_local_update()` 只计算候选更新，`apply_local_update()` 才改变参数。分析阶段可以只记录候选更新而不修改模型，避免 Q4 的统计采样改变 checkpoint。BP 模式继续使用标准 autograd 和 optimizer，两种模式共用完全相同的 forward parameters 和 `state_dict` 命名。

### 3.4 BP 基线

BP 组模型使用相同 encoder/decoder 结构，通过 reconstruction objective 训练。训练完成后冻结 encoder，并使用与 Hebbian 完全相同的 linear probe 协议。可另加 supervised BP CNN 作为上界，但不能替代公平的 BP-AE baseline。

BP 模式应由共享模型的 `learning_rule: bp` 配置触发，而不是从 notebook 复制另一份网络。BP reference 和 BP baseline 需要区分：

- **BP baseline：** 正常训练完成的 BP autoencoder，用于 Q1–Q3/Q5–Q6；
- **BP reference：** 在 Hebbian checkpoint 副本上临时计算的 reconstruction negative gradient，只用于 Q4，不执行 optimizer step。

### 3.5 Decoder、公平重建与“生物合理性”声明

在编码器学习规则之外，decoder 是最主要的混杂因素。按以下顺序选择并固定协议：

1. **System reconstruction：** 保留各方法实际训练得到的 decoder。BBB 使用
   end-to-end BP；HHH 使用冻结 encoder 后训练的 BP decoder；HHB/HBB 可按
   冻结配置联合训练 BP suffix 与 decoder。该口径衡量系统结果，但包含训练
   协议差异。
2. **Standardized decoder reconstruction：** 每个 encoder 完成后冻结整个
   encoder，从同一 paired decoder initialization 重新训练新 decoder。所有方法
   使用相同 MSE、Adam `lr=0.003`、train batches、batch order、epoch budget
   和 validation-only checkpoint selection；decoder gradient 不得进入 encoder。
3. **机制对照：** 可额外报告 tied-weight transpose-convolution reconstruction，
   不更新 encoder 权重。
4. **可选扩展：** 若实现局部 decoder rule，单独作为实验，不替换上述两种口径。

每个 reconstruction row 必须携带
`reconstruction_protocol: system | standardized_decoder`。System reconstruction
可用于最终系统性能；Standardized decoder reconstruction 是比较 encoder 可恢复
信息的主要重建指标。若两者结论不同，必须报告差异，不得选择性呈现。

这一设计仍不是“纯 Hebbian 端到端 autoencoder”。HHH 的 decoder、Hybrid 的
BP suffix/decoder 以及所有 linear probes 均使用 BP；摘要和图注必须如实描述。

### 3.6 正式模型矩阵与必要对照

| ID | Enc1 | Enc2 | Enc3 | Decoder | 研究作用 |
|---|---|---|---|---|---|
| `BBB` / Full BP | BP | BP | BP | BP | 全局 credit-assignment reference |
| `HBB` / Hybrid-HBB | Hebbian | BP | BP | BP | 一层 Hebbian depth；与 RBB 比较其净价值 |
| `HHB` / Hybrid-HHB | Hebbian | Hebbian | BP | BP | 最小 BP 深层干预；主要 confirmation candidate |
| `HHH` / Full Hebbian | Hebbian | Hebbian | Hebbian | BP | 纯 Hebbian encoder baseline |
| `RBB` | frozen random | BP | BP | BP | HBB 的 matched random-prefix control |
| `RRB` | frozen random | frozen random | BP | BP | HHB 的 matched random-prefix control |
| Full Random | frozen random | frozen random | frozen random | BP | 整体表示学习下界 |

`BBB/HBB/HHB/HHH` 构成固定三层结构中的 0/1/2/3 Hebbian-depth 梯度。
`RBB/RRB` 用于回答 Hebbian prefix 是否超越随机 features；Full Random 不能
替代这两个因果控制。可选 supervised BP CNN 只作为任务性能上界，不参与
autoencoder representation 主统计比较。

---

## 4. Phase 0-v1 强制实验标准

本节由 Hebbian 负责人发布，作为 Full BP、Full Hebbian、Hybrid 与随机前缀
控制的共同主基线。队友无需重新设计主架构；若发现实现问题，应提交变更说明
并升级版本号。未经记录的偏离只能算 exploratory run，不能进入主比较表。

### 4.1 已冻结的总览

| 项目 | `phase0-v1` 标准 | 状态 |
|---|---|---|
| 主数据集 | `torchvision.datasets.MNIST` | 🔒 |
| 数据划分 | 官方 train 60k → stratified train 50k/val 10k；官方 test 10k | 🔒 |
| split seed | `0`，保存原始 MNIST index 到 `mnist_split_v1.npz` | 🔒 |
| 输入预处理 | `ToTensor()`，范围 `[0,1]`，主实验不做 z-score | 🔒 |
| 主架构 | `conv3_ae_v1`，3 Conv encoder + 3 ConvTranspose decoder | 🔒 |
| 主 latent dimension | `64`；sweep 为 `[16, 32, 64, 128]` | 🔒 |
| activation | Encoder/decoder hidden：ReLU；reconstruction output：Sigmoid | 🔒 |
| normalization/dropout/pooling | 全部关闭；使用 stride convolution | 🔒 |
| encoder bias | `False`；decoder bias：`True` | 🔒 |
| learning-rule switch | 同一 forward model；仅 trainer/update rule 不同 | 🔒 |
| target clamping | 主实验永久关闭 | 🔒 |
| paired model seeds | `[0, 1, 2, 3, 4]` | 🔒 |
| batch size | train/probe/eval 均为 `128` | 🔒 |
| reconstruction objective | 像素平均 MSE，目标为 `[0,1]` 原图 | 🔒 |
| linear probe | frozen encoder + train-feature standardization + 单层 Linear | 🔒 |
| decoder protocol | BP：joint AE；Hebbian：训练 encoder 后冻结，再训练同构 BP decoder | 🔒 |
| BP reference | frozen Hebbian snapshot 上的 raw reconstruction negative gradient | 🔒 |
| 主噪声 | Gaussian；salt-and-pepper 和 pixel masking 为预注册补充实验 | 🔒 |
| 统计单位 | 5 个 paired seeds；报告逐 seed、mean±SD、paired difference 95% CI | 🔒 |
| 正式运行入口 | Python module/CLI + validated YAML；禁止 notebook cell 作为入口 | 🔒 |

### 4.2 固定模型 shape

主架构不使用 BatchNorm、LayerNorm、Dropout 或 pooling。`L` 是 bottleneck dimension，主实验取 `L=64`。

| Stage | Operation | Parameters | Output shape |
|---|---|---|---|
| Input | — | — | `B × 1 × 28 × 28` |
| h1 | Conv2d + ReLU | `1→16, k=3, s=2, p=1, bias=False` | `B × 16 × 14 × 14` |
| h2 | Conv2d + ReLU | `16→32, k=3, s=2, p=1, bias=False` | `B × 32 × 7 × 7` |
| z | Conv2d + ReLU | `32→L, k=7, s=1, p=0, bias=False` | `B × L × 1 × 1` |
| d1 | ConvTranspose2d + ReLU | `L→32, k=7, s=1, p=0, bias=True` | `B × 32 × 7 × 7` |
| d2 | ConvTranspose2d + ReLU | `32→16, k=4, s=2, p=1, bias=True` | `B × 16 × 14 × 14` |
| x_hat | ConvTranspose2d + Sigmoid | `16→1, k=4, s=2, p=1, bias=True` | `B × 1 × 28 × 28` |

这一定义使 `latent_dim=L`，无需再区分 channel dimension 与 flatten dimension。Encoder 的 3 层指三个可学习 Conv2d；ReLU 不计入层数。Decoder 同理为三个可学习 ConvTranspose2d。

所有权重使用 paired seed 初始化。Encoder 和 decoder hidden weights 使用 `kaiming_uniform_(mode="fan_in", nonlinearity="relu")`；最后一个 Sigmoid reconstruction layer 使用 `xavier_uniform_`，所有 bias 初始化为 0。若后续单独研究初始化，必须作为 ablation，不修改 v1 主结果。

### 4.3 公共模型与 trainer API

```python
model.encode(x, return_all_layers=False)
model.decode(z)
model.reconstruct(x)
trainer.train_batch(x)
rule.compute_local_update(pre_activity, post_activity)
rule.apply_local_update(delta_w)
extract_representations(model, dataloader, layers, sample_ids)
save_checkpoint(path, model, config, metadata)
load_checkpoint(path, model)
```

`trainer.train_batch(x)` 在表示学习阶段不接收 label。`learning_rule` 与
`encoder_layer_rules` 只决定 trainer/rule，不决定另一套模型 class。
`return_all_layers=True` 时按固定 key 返回 `h1`、`h2`、`z`；不得只返回匿名
list。所有 representation 文件必须包含 `sample_id`、`label`、`seed`、
`model_type`、`encoder_layer_rules`、`architecture_id` 和 `checkpoint_epoch`。

### 4.4 数据划分、DataLoader 与预处理

教程中的临时划分不直接继承。MNIST 官方 60,000 张 training images 使用 `split_seed=0` 做分层无放回划分：50,000 train、10,000 validation；官方 10,000 张 test 保持不变。保存三个 split 的原始 dataset indices 与标签校验和到 `data/splits/mnist_split_v1.npz`。任何模型都不得根据 test metric 选择超参数或 checkpoint。

主实验只使用 `ToTensor()` 映射至 `[0,1]`，不做 z-score。原因是保持 reconstruction target、Hebbian pre-synaptic activity 和噪声尺度一致。数据加载器必须返回稳定的原始 `sample_id`，以保证 representation、noise 和 update analysis 能按样本对齐。

```yaml
batch_size: 128
train_shuffle: true
eval_shuffle: false
drop_last: false
num_workers: 0
pin_memory: false
```

每个 paired model seed 使用同一 DataLoader generator seed 和同一 batch order。`num_workers=0` 是 v1 的可复现标准；性能优化只能在确认 batch IDs 不变后另开配置。

### 4.5 固定训练协议

#### Hebbian encoder

- Rule：competitive Oja-style local update；
- Activity：`y = ReLU(preactivation)`；
- Competition：在每个 sample、每个 spatial location 上沿 channel 维执行 top-k，`k=max(1, ceil(0.10 × C_out))`；
- Candidate update：仅使用当前层 input patches 与 masked post-synaptic activity；
- Stabilization：每次 `apply_local_update()` 后，对每个 output filter 做 L2 normalization，`eps=1e-8`；
- Bias：encoder 无 bias；
- Training order：greedy Conv1 → freeze，Conv2 → freeze，Conv3 → freeze；
- Budget：每层 10 epochs，因此每个 encoder weight tensor 接触 10 次完整 training split；
- Phase 0 v1.1 frozen LR：`5e-4`；旧 validation grid
  `[1e-4, 5e-4, 1e-3, 5e-3]` 只保留为 tuning provenance；
- Labels：encoder trainer 不接收 label，`target_clamping=false`。

正式公式仍以第 3.3 节为准。top-k、LR 或 normalization 的变化均属于消融或 validation search，不能悄悄改变主规则。

#### BP autoencoder baseline

- 同一 paired initial encoder/decoder state；
- End-to-end reconstruction training；
- Loss：pixel-mean MSE；
- Optimizer：Adam，Phase 0 v1.1 `lr=0.003,
  betas=(0.9,0.999), weight_decay=0`；
- Budget：10 epochs，使每个 encoder layer 与 Hebbian 对应层具有相同的 per-layer data exposure；
- Checkpoint：最低 validation MSE，test 不参与选择。

Hebbian 的 greedy encoder 共需 30 个 dataset passes，而 BP 同时更新三层，只需 10 passes。报告中必须同时给出 per-layer exposure、总 dataset passes 和 wall-clock，不能只用“epoch 数”宣称谁学习更快。

#### Decoder protocols

HHH encoder 三层训练完成后冻结，其 system decoder 从 paired initialization
开始训练。HHB/HBB 的 system decoder 可与 BP suffix 联合训练。除此之外，每个
方法都必须在冻结完整 encoder 后，从 paired initialization 训练 standardized
decoder。两类 decoder 均使用 pixel-mean MSE、Adam、10 epochs 和
validation-only checkpoint selection；standardized decoder 必须统一使用
Phase 0 v1.1 `lr=0.003`，且 gradient 不得传入 encoder。

#### Linear probe

- 输入：flatten 后的 `z∈R^L`；
- 先用 train split 计算每个 latent feature 的 mean/std，并应用 `(z-mean)/(std+1e-6)`；
- 单层 `Linear(L,10,bias=True)`，无 hidden layer；
- CrossEntropyLoss；
- SGD：`lr=0.1, momentum=0.9, weight_decay=1e-4`；
- 30 epochs，batch size 128；
- checkpoint 根据 validation accuracy 选择；
- encoder checksum 在 probe 训练前后必须完全一致；
- probe seed 与 model seed 相同，所有方法使用相同 probe initialization protocol。

### 4.6 Noise、representation 与 update-analysis 固定样本

- Gaussian：`sigma=[0.0,0.1,0.2,0.3,0.4]`；
- Salt-and-pepper：`ratio=[0.0,0.1,0.2,0.3,0.4]`，被选像素等概率置 0 或 1；
- Pixel masking：`ratio=[0.0,0.1,0.2,0.3,0.4]`，独立像素置 0；
- 所有噪声先在 `[0,1]` 空间加入，再 clip 至 `[0,1]`；
- `noise_seed=2026`，噪声由 `sample_id/noise_type/severity` 确定，与 model seed 无关；
- 所有方法必须读取相同的 noisy sample，不能在各自循环中临时随机生成。

Representation 主分析使用固定的 2,000 张 test images，每类 200 张，`representation_subset_seed=17`，索引保存为 manifest。Update analysis 使用 training split 上固定的 50 个 mini-batches，batch size 128，`update_batch_seed=31415`；所有 snapshot 和 learning rules 复用相同 batch IDs。

Q4 snapshot 固定为 Hebbian greedy training 的 `conv1_end`、`conv2_end`、`conv3_end`。每个 snapshot 复制后冻结 encoder，使用 paired decoder initialization 训练 10 epochs reference decoder，再计算：

```text
ΔW_BP_ref = -∇W MSE(decoder(encoder(x)), x)
```

BP reference 不执行 optimizer step，不包含 Adam moments、momentum 或 weight decay。Hebbian candidate update 同样只计算不应用。Cosine 在相同 layer/snapshot/batch 上计算；SNR 对 50 个 frozen-state mini-batch candidate updates 分别为 Hebbian 和 BP reference 计算。

### 4.7 公共配置的固定 schema

教程中的 cell-level 常量必须进入 resolved config，不能散落在训练脚本中。最低字段如下：

```yaml
source:
  tutorial_id: null          # 路径、URL、版本或 hash；仅作 provenance
  migrated_from_notebook: true

data:
  dataset: MNIST
  split_manifest: data/splits/mnist_split_v1.npz
  split_seed: 0
  input_range: [0.0, 1.0]
  normalization: none
  batch_size: 128
  num_workers: 0

model:
  architecture: conv3_ae_v1
  encoder_channels: [16, 32, 64]
  latent_dim: 64
  latent_dims_sweep: [16, 32, 64, 128]
  normalization: none
  dropout: 0.0
  target_clamping: false

training:
  learning_rule: hybrid      # bp | hebbian | hybrid | frozen_random
  encoder_layer_rules:       # 不改变 forward architecture
    enc1: hebbian
    enc2: hebbian
    enc3: bp
  seed: 0
  paired_seeds: [0, 1, 2, 3, 4]
  tuning_seed: 42
  confirmation_seeds: [43, 44]
  max_tuning_trials_per_rule: 8
  hebbian_epochs_per_layer: 10
  bp_epochs: 10
  decoder_epochs: 10
  reconstruction_protocols:
    - system
    - standardized_decoder
  standardized_decoder:
    freeze_entire_encoder: true
    paired_initialization: true
    validation_selection: min_reconstruction_mse

hebbian:
  lr: 5.0e-4
  winner_fraction: 0.10
  update: oja
  filter_l2_normalize: true
  normalization_epsilon: 1.0e-8

backprop:
  optimizer: adam
  lr: 0.003
  betas: [0.9, 0.999]
  weight_decay: 0.0
  reconstruction_loss: mse_pixel_mean

probe:
  type: linear
  freeze_encoder: true
  standardize_features: true
  epochs: 30
  optimizer: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4

noise:
  seed: 2026
  gaussian_sigma: [0.0, 0.1, 0.2, 0.3, 0.4]
  salt_pepper_ratio: [0.0, 0.1, 0.2, 0.3, 0.4]
  pixel_mask_ratio: [0.0, 0.1, 0.2, 0.3, 0.4]

update_analysis:
  bp_reference: reconstruction_raw_negative_gradient
  checkpoints: [conv1_end, conv2_end, conv3_end]
  freeze_snapshot: true
  include_optimizer_state: false
  num_batches: 50
  batch_size: 128
  batch_seed: 31415
  epsilon: 1.0e-12
```

运行开始时对 config 做 schema validation，并保存包含默认值、实际 shape、parameter count、split hash 和 code version 的 `config_resolved.yaml`。出现未知字段直接报错，不能静默忽略。`learning_rule` 是训练器策略，不得在模型构造时暗中选择另一套 encoder class。

### 4.8 结果与统计标准

主表同时包含每个 seed 的原始值以及 `mean ± SD`。模型差异以 paired seed difference 为基本单位，并对 seed-level paired differences 给出 bootstrap 95% CI；由于只有 5 个 seeds，不把单一 p-value 作为主要结论。

以下 trial budget 记录已完成的原始 BP/HHH validation tuning provenance。
超参数开发固定使用单独的 `tuning_seed=42`，不能把 formal seeds
`[0,1,2,3,4]` 中表现最好者用于选配置。原始两种规则各允许最多 8 个
validation trials；HHB/HBB 继承冻结规则，不重新开启 tuning：

- Hebbian：先搜索 `hebbian_lr=[1e-4,5e-4,1e-3,5e-3]`，再以最佳 LR 搜索 `winner_fraction=[0.05,0.10,0.20]`；
- BP：搜索 Adam `lr=[3e-4,1e-3,3e-3]`，再以最佳 LR 搜索 `weight_decay=[0,1e-5,1e-4]`；
- Linear probe 超参数固定，不参与两边的 trial budget。

每个 trial 都只使用 validation split：BP run 内 checkpoint 按 validation reconstruction MSE 选择；trial 间最终选择统一按 validation linear-probe accuracy。Hebbian trial 使用训练结束 encoder 的 validation linear-probe accuracy。选定后冻结完整配置，从头运行 paired seeds `[0,1,2,3,4]`。最终 test 每个冻结 run 只评估一次。

**允许修改的方式：** 修复 bug 不改变版本号，但需记录 commit；任何会改变数据、shape、loss、training budget、probe 或 noise realization 的修改必须升级为 `phase0-v2`，并重新运行双方受影响的主实验。

### 4.9 发给 BP 队友的最低交付要求

BP 实现必须复用 shared model/data/evaluation modules，只新增或调用 BP trainer。提交主结果前需提供：

- 同一 seed 下，未训练 BP/Hebbian 模型的 encoder/decoder `state_dict` hash 相同；
- 对固定 input batch，训练前的 `h1/h2/z/x_hat` 数值相同；
- BP 训练只由 `learning_rule=bp` 切换，不修改 architecture config；
- BP encoder/decoder 联合训练 10 epochs，保存最低 validation MSE checkpoint；
- 使用共享 frozen linear-probe、noise 和 representation scripts；
- 输出第 10 节规定的完整文件，不接受只有截图或 notebook output 的结果；
- 在 PR/提交说明中列出任何与 `phase0-v1` 的偏离；无偏离时明确写 `phase0-v1 compliant`。

交接验收命令最终应统一为：

```text
python -m training.train_representation --config configs/bp_main.yaml
python -m training.train_linear_probe --run-dir <bp_run_dir>
python -m evaluation.evaluate_clean --run-dir <bp_run_dir>
python -m evaluation.evaluate_noise --run-dir <bp_run_dir>
```

命令名称在实现前可调整一次，但 BP/Hebbian 必须共享入口参数和结果 schema。

---

## 5. 分阶段执行计划与验收标准

### Phase 0 — 冻结定义与接口（阻塞后续主实验）

- [x] 发布“3-layer”定义与逐层 shape；
- [x] 冻结 decoder 训练协议与论文措辞；
- [x] 建立教程/参考 notebook 的 provenance 与“保留思想—必须重写—禁止进入主实验”迁移表；
- [ ] 补充原始 Hebbian 教程的真实路径/URL、版本、hash 和许可信息；
- [x] 冻结共享 `ConvEncoder`/`Decoder` forward API，BP/Hebbian 只切换 trainer/update rule；
- [x] 生成并保存 `mnist_split_v1.npz`，校验 50k/10k/10k 样本无重叠；
- [x] 冻结 normalization、paired seeds、主架构与初始化；
- [x] 冻结主 latent dimension 与 sweep；
- [x] 冻结 linear probe、noise generator 与固定分析样本；
- [x] 明确主实验 `target_clamping: false`；
- [x] 实现无标签泄漏自动测试；
- [x] 区分正常训练的 BP baseline 与仅用于 Q4 的 BP reference；
- [x] 写明公共 config、结果和逐层 update record schema；
- [x] 将 schema 实现为运行时 validation；
- [x] 规定 notebook 不能作为训练入口，正式结果只能由模块化 CLI/config 产生；
- [ ] 将第 4 节发送给队友并取得“按 `phase0-v1` 实现”的书面确认。

**Done when：** 第 4 节所有 🔒 标准已由代码和测试实现；同一模型配置和初始 `state_dict` 可以分别交给 BP/Hebbian trainer；同一 batch 的 forward output 与各层 shape 完全一致；切换规则不会改变模型结构；无 label 进入 encoder 表示学习路径；split/config/update schemas 已落盘；队友在 Git 或书面消息中确认按 `phase0-v1` 执行。

### Phase 1 — Hebbian layer 最小实现与测试

- [ ] 实现 `HebbianLinear` 作为公式和测试原型；
- [x] 实现 convolutional Hebbian local update；
- [x] 将教程 custom autograd 思想迁移为 `compute_local_update()`/`apply_local_update()`；
- [x] 实现 per-sample/per-location channel top-k WTA；
- [x] 实现 Oja stabilization 与逐 filter L2 normalization；
- [x] 确保 Hebbian encoder 权重不被 autograd/optimizer 意外更新；
- [x] 记录 post activity、candidate update norm、weight norm、sparsity 与 winner frequency；
- [ ] 增加 preactivation distribution 的正式记录；
- [x] 验证“只计算候选更新”不会改变 layer state；
- [x] 验证主训练路径不读取 label，也不存在 target clamping。

**测试：**

- [ ] shape 与普通 Conv2d 一致；
- [x] 固定输入时更新方向与手算结果一致；
- [ ] `hebbian_lr=0` 时权重不变；
- [x] `compute_local_update()` 后、`apply_local_update()` 前权重不变；
- [x] classifier backward 不改变 frozen encoder；
- [ ] 相同 forward state 可同时生成 Hebbian candidate update 与 BP reference；
- [ ] 100–500 steps 后无 NaN/Inf，weight norm 有界；
- [ ] 固定 seed 可复现；
- [ ] dead-neuron 与 single-winner collapse 可被自动检测。

**Done when：** 所有单元测试通过，并保存一份短训练的诊断曲线。

### Phase 2 — 3 层 Hebbian encoder 与 MNIST smoke test

- [x] 构建 3 层卷积 encoder；
- [x] 完成 greedy layer-wise training；
- [x] 冻结 encoder 并验证 checksum 不再变化；
- [x] 训练统一 linear probe；
- [x] 训练/绑定 decoder 并输出 reconstruction；
- [x] 运行 untrained AE 与 random-frozen-encoder/decoder-only reconstruction control；
- [x] 衡量 decoder 对随机 encoder 的补偿程度，并与 BP/Hebbian 重建比较；
- [x] 保存 layer-end encoder、autoencoder 与 probe checkpoints；
- [ ] 保存固定分析 subset 的 h1/h2/z representations。

**表示塌缩门禁：**

- [ ] active-neuron ratio 高于预注册阈值；
- [ ] winner distribution 不由单个神经元长期垄断；
- [ ] activation variance 非零；
- [ ] latent effective rank 明显高于 1；
- [ ] linear probe 明显高于 random encoder 和随机猜测。

**Done when：** 单 seed、单架构完整跑通，结果可从 config + checkpoint 复现。

### Phase 3 — 超参数选择（仅 validation）

按顺序搜索，避免一次改变多个因素：

1. 固定主架构与 latent dimension，搜索 Hebbian learning rate；
2. 搜索 competition/top-k；
3. 搜索 normalization/stabilization；
4. 最后搜索 latent dimension。

起始候选可为：

```yaml
hebbian_lr: [1e-4, 5e-4, 1e-3, 5e-3]
winner_fraction: [0.05, 0.10, 0.20]
stabilization: oja_plus_filter_l2_normalization
latent_dims: [16, 32, 64, 128]
```

卷积 WTA 的 winner 数量依赖 feature map 大小，优先用比例而不是固定 activation value。若计算预算有限，先做粗搜索，再在最佳邻域细搜。

**公平性门禁：** BP 与 Hebbian 的调参 trial 数必须相同或在报告中明确预算差异；测试集不得参与选择。

#### Stage 1B — Hebbian repair/reselection（已冻结）

在 Stage 1 health gate 失败后，使用 tuning seed 42 和同一 2,000-image
validation subset 运行了两组预注册的小型 repair matrix：

- v1：stateless channel RMS / channel-standardized competition；
- v2：在 v1 competition 基础上，仅对 Hebbian local-update patch 加
  dataset-independent batch/spatial centering；forward representation 不变。

每个候选必须同时满足 `h1/h2/z` unchanged health gate 全部通过，以及
validation linear-probe accuracy ≥ `0.8863`。两轮共八个候选均完成，
`test_samples_accessed=0`，但没有候选同时满足两项条件。

**Stage 1B outcome: COMPLETED — NO CANDIDATE PASSED.**

Stage 1B 到此冻结：不增加 v3/v4 候选，不选择 replacement config，不把
seed-42 validation 结果写成正式多 seed 结论。完整表格、hash 和限制见
`docs/stage1b_hebbian_repair.md`。

### Phase 4 — Q1：正式 clean performance 与 Hebbian-depth value

本阶段只在 Stage 2D confirmation 通过后启动。固定三层结构，核心正式矩阵为
`BBB/HHH/HHB/HBB/Full Random`；为回答“一层或两层 Hebbian 是否有价值”，
加入 matched controls `RBB/RRB`。正式 seeds 为 `[0,1,2,3,4]`，所有方法按
seed 配对初始化、数据顺序、decoder/probe 初始化和 validation selection。

- [ ] 对完整模型矩阵运行 5 个 paired seeds；
- [ ] 保存 encoder、system decoder、standardized decoder 与 probe 的逐 epoch/step 指标；
- [ ] 报告 accuracy、macro-F1、CE、两种 reconstruction MSE、AULC、
  samples-to-threshold、samples seen 和 wall-clock；
- [ ] 分开报告 encoder、decoder 与 probe 的学习预算；
- [ ] 计算预注册 paired differences、effect sizes 与 bootstrap 95% CI；
- [ ] 每个冻结 run 只在最终阶段读取 test 一次。

主要 contrasts 按以下顺序解释：

1. `HHB − HHH`：最小 BP Enc3 intervention 是否修复 Full Hebbian；
2. `HBB − HHB`：将 Enc2 从 Hebbian 改为 BP 的增量影响；
3. `HBB − RBB`：单层 Hebbian prefix 的净价值；
4. `HHB − RRB`：两层 Hebbian prefix 的净价值；
5. `BBB − HHB`：最小 Hybrid 与 Full BP 的剩余差距。

截至 2026-07-23 的 paired seeds 0–1 报告
`docs/q1_clean_performance.md` 只覆盖旧的 BP/Hebbian/random matrix，用于
验证统计与恢复管线。它不满足新的 Hybrid/matched-control 设计，不得与新
formal seeds 拼接或作为正式结论。

学习速度定义：

```math
\mathrm{AULC}=\frac{1}{T}\sum_{t=1}^{T} A_t
```

除 epoch-AULC 外，同时以 samples seen 或 wall-clock 对齐曲线。阈值时间只在
被比较方法都能达到同一阈值时报告。

### Phase 5 — Q2：逐层 latent representation

对完全相同的固定 class-balanced subset 和 sample order，提取
`BBB/HHH/HHB/HBB/RBB/RRB` 的 input、h1、h2、z：

- [ ] PCA（主图，确定性强）；
- [ ] UMAP（辅助图，固定 seed 与参数）；
- [ ] 每层 linear probe 和 k-NN accuracy；
- [ ] within/between-class distance 与 separability ratio；
- [ ] silhouette score；
- [ ] effective rank、active-neuron ratio、sparsity；
- [ ] covariance spectrum、stable rank、between/within-class covariance；
- [ ] 计算 `ER(z)/ER(h2)` 与其他预注册 h2→z compensation 指标；
- [ ] 可选 CKA，用于跨模型/架构表示相似性；
- [ ] confusion matrix 与易混类别分析。

重点检验 HHB 的 BP Enc3 是否跨 seed 将低秩 Hebbian h2 转换为更高秩 z，
以及 HBB 是否从 h2 开始恢复。不要只根据 t-SNE/UMAP 的视觉分离下结论。
所有降维图必须共享样本、颜色、预处理和随机种子，并配套定量指标。

### Phase 6 — Q3：噪声与配置鲁棒性

主实验先采用 clean train → noisy test：

```yaml
gaussian_sigma: [0.0, 0.1, 0.2, 0.3, 0.4]
salt_pepper_ratio: [0.0, 0.1, 0.2, 0.3, 0.4]
mask_ratio: [0.0, 0.1, 0.2, 0.3, 0.4]
```

- [ ] 在 `[0,1]` 像素空间加噪并 clip，再执行 normalization；
- [ ] 所有方法使用相同 sample-level noise realization；
- [ ] 每个 severity 报告 accuracy、macro-F1 与 degradation；
- [ ] 对 system/standardized reconstruction 分别报告 degradation；
- [ ] 计算 clean/noisy representation cosine similarity；
- [ ] 计算预测分布 JS divergence；
- [ ] 画 accuracy–severity 和 representation-stability–severity 曲线；
- [ ] 比较 Full BP、Full Hebbian、HHB 与 HBB 的 seed/config 扰动方差。

```math
D=A_{clean}-A_{noise},\qquad RD=\frac{A_{clean}-A_{noise}}{A_{clean}}
```

### Stage 1C — effective-rank metric audit（Q4 前置门禁）

**目的：** 只审计 Stage 1/1B 使用的 representation-health rank 定义是否
在样本轴、特征轴、WTA 位置、centering 和数值稳定性上合理。这个阶段
不训练模型、不应用 local update、不训练新 probe、不重新调参，也不访问
test set。Stage 1B 保持冻结，不因本审计自动重开。

**冻结输入：**

- Stage 1 使用的同一 seed-42 Hebbian checkpoint；
- `data/splits/mnist_validation_health_v1.npz`；
- 同一 2,000-image validation subset、相同 sample order，必须恰好每类
  200 张；
- 已冻结的 linear probe；只重新报告 accuracy，不更新其参数。

**表示位置与变换：**

1. `z_pre_wta`：进入 top-k competition 前、用于确定 winner 的 activation；
2. `z_post_wta`：对同一 activation 应用冻结 top-k mask 后的表示；
3. `z_post_wta_centered`：对完整 2,000-image dataset 的每个 feature
   dimension 去均值；
4. `z_post_wta_l2`：对每个样本的 post-WTA `z` 做 L2 normalization；
5. `z_post_wta_class_centered`：按真实类别减去对应 class centroid，仅作
   within-class geometry audit，不作为无监督训练输入；
6. `h1`、`h2`、`z`：保存 singular-value spectrum。卷积层同时审计：
   `(sample × spatial-location, channel)` 的 channel-health view，以及
   `(sample, channel × height × width)` 的 sample-representation view；
   两者不得混名或混用阈值。

**必须保存：**

- covariance eigenvalue spectrum；
- participation-ratio effective rank；
- stable rank；
- rank ratio（rank metric / 明确记录的 feature dimension）；
- per-class effective rank；
- between-class covariance rank；
- within-class covariance rank；
- frozen linear-probe accuracy；
- 每种 view/transform 的 shape、axis meaning、centering flag、dtype、
  epsilon/threshold 和样本 manifest hash。

所有矩阵分解使用 float64。Covariance 必须沿 feature dimension 计算，
即输入矩阵约定为 rows=samples/observations、columns=features；需要
centered covariance 的指标必须先做 dataset-level feature centering。
数值秩和含 epsilon 的公式必须报告无 epsilon 基线与一组相对尺度敏感性
结果，确认结论不是由 epsilon 主导。

**特别 QA：**

- [x] sample axis 和 feature axis 与保存 metadata 完全一致；
- [x] convolutional flattening 的两种 view 分开计算、分开命名；
- [x] covariance 的维度是 feature × feature；
- [x] dataset-level centering 在 covariance/SVD 前显式执行并记录；
- [x] epsilon/数值阈值改变时主要结论稳定；
- [x] subset 有 2,000 个唯一 sample IDs，且每类恰好 200 张；
- [x] checkpoint 与 frozen probe 的分析前后 hash 不变；
- [x] `test_samples_accessed=0`；
- [x] 所有 spectrum、rank table、probe metric 和 audit report 齐全。

**解释规则：**

- 若 `z_pre_wta` rank 高而 `z_post_wta` rank 低，支持“WTA 本身造成
  rank 压缩”；
- 只有当 `z_pre_wta` 与 `z_post_wta` 都接近 1，才更支持“filters
  本身高度重复”；
- 若结论依赖轴选择、未中心化或 epsilon，则 Stage 1 rank 结论记为
  方法学未通过，不得据此判定 filter collapse；
- Stage 1C 只验证 metric 和机制定位，不选择模型，也不把单 checkpoint
  结果升级为 Q2 的正式多 seed 结论。

**Done when：** 上述 QA 全部通过，保存可复算的 spectra/tables/report，
并在 `PROJECT_STATUS.md` 中记录 metric-validity decision。完成后才能决定
以哪个既有 frozen snapshot 进入 Q4 tooling gate；不得静默选用新的
Hebbian 配置。

截至 2026-07-23，Stage 1C 已完成且 metric validity 为 PASS。
`z_pre_wta` participation rank=`1.0186`，analysis-only `z_post_wta`
rank=`1.0000`；低秩在 WTA 前已经存在，WTA 进一步压缩但不是初始成因。
同一 subset 的 frozen standardized probe accuracy=`0.9040`，因此该 rank
应解释为 raw-covariance anisotropy/channel redundancy，不能单独等同于
分类信息消失。完整证据见 `docs/stage1c_effective_rank_audit.md`。

### Phase 7 — Q4：权重更新机制

教程中的逐层 update、BP reference、cosine 和 SNR 仅作为分析流程原型；正式分析必须在**冻结的模型 snapshot** 上重算。在选定训练 checkpoint 上复制相同 encoder/decoder 状态，对预先保存的一组 mini-batch IDs 分别计算：

- `ΔW_Hebb`：局部 Hebbian 候选更新；
- `ΔW_BP`：同一 encoder/decoder 状态下 reconstruction objective 对 encoder 的负梯度更新。

正式多 seed Q4 除 HHH layer-end snapshots 外，还必须覆盖 HHB 与 HBB：

- HHH：Enc1/Enc2/Enc3 的 local candidate 与 matched BP reference；
- HHB：Enc1/Enc2 local candidates，以及 BP Enc3 的 raw gradient；
- HBB：Enc1 local candidate，以及 BP Enc2/Enc3 的 raw gradients；
- BBB：相同层的 raw BP gradients，作为全局 credit-assignment reference。

分析重点是 Hebbian prefix 与 BP suffix 的 rule boundary，而不是把 Hybrid
更新合并成一个平均方向。必须分别报告每层指标，并检验 Enc3 的方向、SNR
或更新尺度是否与 h2→z representation recovery 同步；相关性不得表述为因果。

两者都只作为 candidate update 记录，采样期间不得调用 `apply_local_update()`、`optimizer.step()` 或更新 BatchNorm running statistics。主 BP reference 定义为不包含 momentum、Adam state 和 weight decay 的 raw negative gradient；如需比较实际 optimizer step，另存为辅助指标并明确命名。

所有指标按 layer 报告，必要时也按 filter 报告：

```math
\mathrm{Alignment}=\frac{\langle\Delta W_H,\Delta W_{BP}\rangle}
{\|\Delta W_H\|_2\|\Delta W_{BP}\|_2+\epsilon}
```

```math
\mathrm{NormRatio}=\frac{\|\Delta W_H\|_2}{\|\Delta W_{BP}\|_2+\epsilon}
```

由于 Hebbian candidate 和 BP raw gradient 的整体尺度由不同规则决定，`phase0-v1` 将 **scale-matched relative bias** 作为主 bias 指标。先在同一 layer/snapshot 上计算 mini-batch mean updates `μ_H` 与 `μ_BP`：

```math
\alpha^*=\frac{\langle\mu_H,\mu_{BP}\rangle}{\|\mu_H\|_2^2+\epsilon},
\qquad
\mathrm{ScaleMatchedBias}=
\frac{\|\alpha^*\mu_H-\mu_{BP}\|_2}
{\|\mu_{BP}\|_2+\epsilon}
```

同时报告 `Alignment` 和 `NormRatio`。未经缩放的 relative difference 可作为辅助指标，但必须命名为 `RawRelativeDifference`，不能直接解释为规则的方向偏差，更不能用 cosine 代替 bias。

在固定 checkpoint 上抽取多个 mini-batches：

```math
\mathrm{SNR}=\frac{\|\mathbb{E}[\Delta W]\|_2^2}
{\mathbb{E}[\|\Delta W-\mathbb{E}[\Delta W]\|_2^2]+\epsilon}
```

Hebbian 与 BP 的 SNR 分别计算。均值和方差的统计单位是“同一 frozen checkpoint 上的不同 mini-batch candidate updates”，不是连续训练 step；否则权重状态变化会与 batch variance 混合。主结果报告线性 SNR，可选地另报 `10 log10(SNR)`，但图表必须标明单位。

- [x] 固定 checkpoint，不能在采样 batch 之间持续更新权重；
- [x] 固定并保存 mini-batch IDs，BP/Hebbian 使用完全相同的数据；
- [x] 关闭 target clamping，BP reference 只依赖 reconstruction loss；
- [x] 记录 `rule/layer/checkpoint/batch_ids/update_norm/cosine` 及 SNR 所需充分统计量；
- [x] 对合成更新向量编写测试：同向 cosine=1、反向=-1、正交=0、零方差时 SNR 行为可控；
- [x] 在 `conv1_end/conv2_end/conv3_end` checkpoints 重复；
- [x] 画 snapshot/layer–alignment、norm、bias、SNR panels；
- [ ] 分析这些指标与 accuracy、separability、robustness 的相关性；
- [ ] 在正式 paired seeds 上扩展到 BBB/HHH/HHB/HBB，并单列 Hybrid boundary；
- [x] 明确 correlation 不代表 causation。

截至 2026-07-23，Stage 2 / Q4 seed-42 tooling gate 已通过。该运行在
Stage 1C 授权的 failure-case snapshots 上使用 50 个固定 batches，完成
raw BP direction、raw/effective Hebbian delta、alignment、norm ratio、
`alpha*`、scale-matched bias 和跨 batch SNR；分析 optimizer step 为 0，
source/analysis checksums 前后一致，test access 为 0。该 PASS 仅验证工具和
单个失败案例，不覆盖跨 seed 统计或 `P7-CORR-01`，完整记录见
`docs/q4_update_mechanism_seed42.md`。

#### Stage 2B — notebook-inspired output-filter update-centering audit

本阶段是一次有限的 validation-only、seed-42 机制实验，不重开 Stage 1B，
不访问 test set，不修改 gate threshold，也不允许根据结果追加候选。保留原始
Oja + WTA failure-case baseline，只测试一个预注册候选：

```python
raw_update_centered = (
    raw_update - raw_update.mean(dim=0, keepdim=True)
)
```

Conv2d 权重为
`[out_channels, in_channels, kernel_height, kernel_width]`，因此输出
filter 维度固定为 `dim=0`。centering 必须作用于完整 raw Hebb–Oja
candidate，并发生在 learning-rate scaling 之前；compute/apply 继续分离，
apply 后仍执行 per-filter L2 normalization。

- [x] 审计 supplied notebook 的默认 update-centering 语义与 Oja 注释分支；
- [x] 验证 Conv2d output-channel axis 和 centering 顺序；
- [x] synthetic tests 覆盖共同方向消除、残差保留、零均值、零更新、
  compute non-mutation 与 resume determinism；
- [x] 先重新核验已通过的 Stage 2 Q4 baseline 完整性；
- [x] 只运行一个 seed-42 output-filter-centered candidate；
- [x] 比较 validation accuracy、effective rank、winner coverage、
  max winner share、alignment、bias 与 SNR；
- [x] 确认 test access 为 0，分析前后 checksum 不变；
- [x] 保持 Stage 1B 冻结，不新增候选、不修改门禁。

截至 2026-07-25，本阶段结论为
`COMPLETED — DOES_NOT_RESOLVE_FAILURE`。candidate validation accuracy
为 `0.1944`，低于冻结 floor `0.8863`；`h1/h2/z` health gate 全部失败，
`z` effective rank 从 `1.0186` 降至 `1.0000`。Enc3 raw/effective
alignment 分别为 `-0.1078/-0.1033`，且 `alpha*` 为负，因此该操作不只是
改变 update scale，也没有改善 representation health。candidate
`eligible_to_replace_baseline=false`。完整证据见
`docs/output_filter_centering_mechanism.md`。

后续冻结决策树审计将该结果唯一分类为
`BRANCH D — FREEZE AS FAILURE-CASE BASELINE`，并正式记录
`COMMON-MODE UPDATE REMOVAL: NOT SUFFICIENT`。不得启动新的 repair
candidate。原始 Oja + WTA config 只保留为
`health-gate failure-case baseline`；后续若继续，只能在另行批准并预注册后
执行正式多 seed failure replication。见
`docs/hebbian_failure_case_protocol_addendum.md`。

#### Stage 2C — Hybrid Hebbian–BP Depth Ablation

经用户单独授权，在不修改 original full-Hebbian failure-case baseline 的
前提下，预注册两个且仅两个 validation-only、seed-42 controls：

- `Hybrid-HHB`：Enc1/Enc2 Hebbian，Enc3/decoder BP；
- `Hybrid-HBB`：Enc1 Hebbian，Enc2/Enc3/decoder BP。

同时从同一个 clean implementation commit 重跑 Full BP 与 Full Hebbian
references，确保 split、对应层/decoder/probe initialization、BP `lr=0.003`
和 validation checkpoint policy 完全配对。所有 frozen layers 必须排除在
optimizer parameter groups 外且 checksum 不变。本阶段只提供 diagnostic
localization 与 Stage-3 candidate recommendation，不运行 confirmation 或
formal seeds。完整预注册规则见 `docs/hybrid_depth_ablation_protocol.md`。

截至 2026-07-25，本阶段已完成且 integrity/pairing gate 为 PASS。Full BP、
Full Hebbian、Hybrid-HHB、Hybrid-HBB validation accuracy 分别为
`0.9226/0.9063/0.9097/0.9161`；z effective rank 分别为
`11.8533/1.0186/10.0850/12.9754`。HHB 修复 z，HBB 进一步修复 h2，
但两者完整 applicable health gate 仍被早期 Hebbian layers 阻断，因此按预注册
规则选择 Outcome D。两者均达到 confirmation eligibility；按“更小 BP
intervention 优先”冻结 `Hybrid-HHB` 为 confirmation candidate。不得据此
宣称跨 seed 结论或直接进入 Stage 3。完整记录见
`docs/hybrid_depth_ablation_results.md`。

#### Stage 2D — Hybrid-HHB validation-only confirmation gate

在进入正式 Phase 4 前，用预注册 seeds `[43,44]` 确认 seed-42 localization
是否可复现。配置完全继承 Stage 2C 的 `Hybrid-HHB`：Enc1/Enc2 使用冻结的
Oja/WTA/L2 Hebbian 规则，Enc3 使用 BP，BP learning rate=`0.003`。不得
重新调参、修改 health threshold、访问 test set，或根据 seed 43 的结果改变
seed 44。

每个 seed 必须同时运行 paired `BBB/HHH/HHB` references，并满足：

- validation linear-probe accuracy `>=0.8863`；
- **standardized decoder** reconstruction MSE
  `<=1.25 ×` paired BBB standardized-decoder MSE；
- `ER_z(HHB) >= 2.0` 且 `ER_z(HHB) >= 2 × ER_z(HHH)`；
- `ER_z(HHB) / (ER_h2(HHB)+epsilon) >= 2.0`，表明 BP Enc3 能补偿低秩
  Hebbian h2，而不是只复现 system-decoder joint adaptation；
- system reconstruction 同时完整报告，但不作为 representation-repair
  的单独证据；
- pairing、frozen-layer checksum、resume determinism、artifact completeness
  全部 PASS，且 `test_samples_accessed=0`。

两个 confirmation seeds 都通过才可将 HHB 标记为
`CONFIRMED FOR FORMAL STAGE 3/PHASE 4`。任一 seed 失败则状态为
`CONFIRMATION FAILED`，停止正式矩阵并报告异质性；不得追加第三个
confirmation seed 来覆盖失败结果。

### Phase 8 — Q5/Q6：维度与 architecture asymmetry

#### 8.1 Latent dimension

对每个可实现的 flatten latent dimension，优先运行
`BBB/HHH/HHB/HBB × 5 paired seeds`，并报告 clean/noisy accuracy、两种
reconstruction、probe、separability、effective rank 和训练时间。RBB/RRB
只在与“Hebbian prefix 是否有价值”直接相关的关键维度上运行，避免无解释价值
的完整笛卡尔积。

#### 8.2 Architecture asymmetry 的主定义

为了让 Q5/Q6 真正影响分类 latent，主实验把非对称定义为 **3 层 encoder 内部参数/通道的前后分配**，而不是只改变冻结 encoder 后面的 decoder。

`phase0-v1` 在固定 `L=64`、相同 kernel/stride/receptive field 下冻结三种配置。括号内为 encoder channels：

- `early_heavy`：`[64, 28, 64]`，encoder weight parameters = `104,512`；
- `balanced`：`[16, 32, 64]`，encoder weight parameters = `105,104`；
- `late_heavy`：`[4, 33, 64]`，encoder weight parameters = `104,712`。

三者 encoder weight parameter range/mean 小于 0.6%，因此主分类/representation 比较近似固定 encoder 容量、bottleneck 和 receptive field。由于 decoder 的后两层使用 `k=4` 而 encoder 对应层使用 `k=3`，完整 autoencoder 参数量约为 `222,109 / 213,953 / 210,414`，range/mean 约 5.5%；因此 asymmetry 实验的 reconstruction 指标只作辅助结果，并必须同时报告 decoder 与 total parameter count。实际参数量由代码写入 metadata，并断言 encoder 参数差异小于 1%；以上手工数值只作为 schema 预期值。

可补充定义 encoder–decoder 参数不对称指数：

```math
AI=\log\frac{P_{encoder}}{P_{decoder}}
```

但若 Hebbian encoder 在 decoder 训练前已经冻结，则 decoder-side asymmetry 对其 latent 的“无影响”是协议的直接结果，不能解释成 Hebbian 天生更鲁棒。该实验应作为机制对照单独报告。

#### 8.3 敏感性

除 range/mean 外，优先报告相对于 balanced baseline 的变化和 rule × architecture interaction：

```math
\mathrm{Sensitivity}(M)=
\frac{\max_a M(a)-\min_a M(a)}{|M(a_{balanced})|+\epsilon}
```

- [ ] 比较 accuracy、robustness、separability 和 effective rank；
- [ ] 检验 method × architecture interaction，并分别报告 Full 与 Hybrid；
- [ ] 对每层画 representation 指标；
- [ ] 用 CKA 或 centered geometry 比较结构变化前后的表示；
- [ ] 结合 dead neuron、winner entropy、Hybrid compensation 和 update SNR
  解释 HHH/HHB/HBB 的敏感性差异。

### Phase 9 — 扩展实验（主实验完成后）

- [ ] CIFAR-10；
- [ ] non-stationary classes；
- [ ] 第二种 Hebbian variant；
- [ ] noisy training；
- [ ] 局部 decoder learning rule。

### 5.1 详细任务分解（WBS 与 Task ID 设计）

上面的 Phase checklist 是阶段验收摘要；下面保留 Task ID 作为实施和证据映射。实时状态统一在 `PROJECT_STATUS.md` 维护，避免本文件数百个 checkbox 与运行报告产生冲突。任务完成时仍应产生一个可观察的文件、测试、日志或决策记录。

#### P0 — 标准、数据与接口

- [x] `P0-DOC-01` 创建 `docs/tutorial_migration.md`；
- [ ] `P0-DOC-02` 记录教程路径、来源、版本、hash 和访问日期；
- [x] `P0-DOC-03` 建立教程内容到正式模块的逐项映射表；
- [x] `P0-DOC-04` 标记教程中禁止进入主实验的 custom autograd 与 target clamping；
- [ ] `P0-DATA-01` 创建 MNIST split 生成脚本；
- [ ] `P0-DATA-02` 使用 `split_seed=0` 生成 stratified 50k/10k train/val indices；
- [ ] `P0-DATA-03` 保存 `mnist_split_v1.npz` 和标签校验和；
- [ ] `P0-DATA-04` 测试 train/val/test 无重叠且数量正确；
- [ ] `P0-DATA-05` 测试相同 seed 重建的 split hash 相同；
- [ ] `P0-CFG-01` 创建公共 YAML 默认配置；
- [ ] `P0-CFG-02` 实现 config schema 和必填字段检查；
- [ ] `P0-CFG-03` 实现未知字段直接报错；
- [x] `P0-CFG-04` 保存包含默认值的 resolved config；
- [x] `P0-CFG-05` 在 metadata 中写入 code version、split hash 和 parameter count；
- [ ] `P0-API-01` 定义 `encode/decode/reconstruct` 方法签名；
- [ ] `P0-API-02` 定义 `h1/h2/z` 的固定返回 key；
- [ ] `P0-API-03` 定义 trainer 与 learning-rule interface；
- [x] `P0-API-04` 定义 checkpoint save/load contract；
- [x] `P0-API-05` 定义 representation/update result schema；
- [ ] `P0-TEST-01` 测试 BP/Hebbian 初始 `state_dict` hash 相同；
- [ ] `P0-TEST-02` 测试切换 learning rule 不改变模型 parameter names/shapes；
- [ ] `P0-TEST-03` 测试 representation trainer 不接收 label；
- [ ] `P0-TEAM-01` 将第 4 节发送给 BP 队友；
- [ ] `P0-TEAM-02` 记录队友的 `phase0-v1 compliant` 确认或偏离项。

#### P1 — Hebbian learning rule 与单层实现

- [ ] `P1-BASE-01` 定义 `LocalLearningRule` 抽象接口；
- [x] `P1-BASE-02` 区分 `compute_local_update()` 和 `apply_local_update()`；
- [ ] `P1-LIN-01` 实现 `HebbianLinear` forward；
- [ ] `P1-LIN-02` 实现 Linear pre/post activity cache；
- [ ] `P1-LIN-03` 实现 Linear Oja candidate update；
- [x] `P1-WTA-01` 实现 channel-wise top-k mask；
- [x] `P1-WTA-02` 处理 `k=max(1,ceil(fraction×channels))` 边界；
- [x] `P1-WTA-03` 记录 winner count、frequency 和 entropy；
- [x] `P1-CONV-01` 使用 unfold 或等价方法提取 convolution patches；
- [x] `P1-CONV-02` 实现 Conv2d Oja candidate update；
- [x] `P1-CONV-03` 按 batch 与 spatial positions 正确求均值；
- [x] `P1-CONV-04` 实现 per-output-filter L2 normalization；
- [x] `P1-CONV-05` 在 `torch.no_grad()` 下应用 update；
- [x] `P1-CONV-06` 确保 Hebbian encoder parameters 不进入 BP optimizer；
- [x] `P1-LOG-01` 定义 layer update diagnostics 数据结构；
- [x] `P1-LOG-02` 记录 update norm、weight norm 和 activation statistics；
- [ ] `P1-TEST-01` 用小矩阵手算验证 Linear update；
- [x] `P1-TEST-02` 用小卷积输入手算验证 Conv update；
- [x] `P1-TEST-03` 测试 `hebbian_lr=0` 权重不变；
- [x] `P1-TEST-04` 测试 compute 阶段不改变权重；
- [x] `P1-TEST-05` 测试 apply 阶段只改变目标层；
- [x] `P1-TEST-06` 测试每个 filter 更新后 norm≈1；
- [x] `P1-TEST-07` 测试固定 seed 得到相同 update；
- [x] `P1-TEST-08` 测试 500 steps 内无 NaN/Inf；
- [x] `P1-TEST-09` 测试 label 不影响 Hebbian candidate update。

#### P2 — 3 层模型、训练器与 smoke test

- [ ] `P2-MODEL-01` 实现 Conv1，并断言输出 `16×14×14`；
- [ ] `P2-MODEL-02` 实现 Conv2，并断言输出 `32×7×7`；
- [ ] `P2-MODEL-03` 实现 Conv3，并断言输出 `L×1×1`；
- [ ] `P2-MODEL-04` 实现三个 decoder layers 并恢复 `1×28×28`；
- [ ] `P2-MODEL-05` 实现 paired deterministic initialization；
- [ ] `P2-MODEL-06` 实现参数量与 shape summary；
- [x] `P2-TRAIN-01` 实现 Conv1 local training stage；
- [ ] `P2-TRAIN-02` 实现 Conv1 freeze 和 checksum 验证；
- [x] `P2-TRAIN-03` 实现 Conv2 local training stage；
- [ ] `P2-TRAIN-04` 实现 Conv2 freeze 和 checksum 验证；
- [x] `P2-TRAIN-05` 实现 Conv3 local training stage；
- [x] `P2-TRAIN-06` 保存 `conv1_end/conv2_end/conv3_end` checkpoints；
- [x] `P2-TRAIN-07` 实现 epoch/step/samples-seen 计数；
- [ ] `P2-PROBE-01` 提取 train/val/test latent features；
- [ ] `P2-PROBE-02` 仅用 train features 拟合 mean/std；
- [ ] `P2-PROBE-03` 实现 frozen single-layer linear probe；
- [ ] `P2-PROBE-04` 根据 validation accuracy 保存 probe checkpoint；
- [ ] `P2-PROBE-05` 验证 probe 前后 encoder checksum 不变；
- [x] `P2-DEC-01` 从 paired initialization 创建 Hebbian decoder；
- [x] `P2-DEC-02` detach/freeze encoder 后训练 decoder；
- [x] `P2-DEC-03` 保存 reconstruction grid 和 validation MSE；
- [x] `P2-DEC-04` 冻结 paired random encoder，仅训练同构 BP decoder；
- [x] `P2-DEC-05` 比较 untrained、decoder-only、Hebbian 与 BP reconstruction；
- [x] `P2-SMOKE-01` 在小数据子集跑通全流程；
- [x] `P2-SMOKE-02` 在完整 MNIST 上运行 seed 0、L=64；
- [x] `P2-SMOKE-03` 重新加载 checkpoint 并复现相同 test metrics；
- [x] `P2-GATE-01` 计算 active-neuron ratio；
- [x] `P2-GATE-02` 计算 winner entropy；
- [x] `P2-GATE-03` 计算 activation variance 和 effective rank；
- [ ] `P2-GATE-04` 与 random encoder probe 和 10% chance baseline 比较；
- [x] `P2-GATE-05` 对 collapse gate 给出 pass/fail 和原因。

#### P2B — Stage 1B Hebbian repair/reselection

- [x] `P2B-CFG-01` 预注册 v1 stateless competition candidate matrix；
- [x] `P2B-RUN-01` 完成 v1 四个 validation-only candidates；
- [x] `P2B-CFG-02` 在启动前预注册 v2 centered-local-input matrix；
- [x] `P2B-RUN-02` 完成已启动的 v2 四个 candidates；
- [x] `P2B-QA-01` 验证八个 trials 的 config、checkpoint、probe、health
  JSON 和 trial rows 齐全；
- [x] `P2B-QA-02` 验证 health extraction 前后 state hash 不变；
- [x] `P2B-QA-03` 验证 selection records 的 test access 均为 0；
- [x] `P2B-SEL-01` 按预注册 health + accuracy rule 判定，无 eligible trial；
- [x] `P2B-FREEZE-01` 冻结
  `COMPLETED — NO CANDIDATE PASSED`，停止 v3/v4，不生成 replacement config；
- [x] `P2B-NOTE-01` 保存完整结果、hash、限制与机制观察。

#### P2C — Stage 2C Hybrid depth diagnostic（已完成）

- [x] `P2C-CFG-01` 预注册 BBB/HHH/HHB/HBB seed-42 validation-only matrix；
- [x] `P2C-QA-01` 验证相同 clean source、split、初始化、probe 和零 test access；
- [x] `P2C-RUN-01` 完成四个 paired runs 与完整 checkpoints；
- [x] `P2C-REP-01` 在固定 2,000-image subset 计算 h1/h2/z health metrics；
- [x] `P2C-DEC-01` 按预注册规则选择 Outcome D；
- [x] `P2C-FREEZE-01` 冻结 HHB 为 confirmation candidate，不升级单 seed 结论。

#### P2D — Hybrid-HHB confirmation（下一阻塞阶段）

- [ ] `P2D-PROTO-01` 将 seeds `[43,44]`、阈值、paired references 和停止规则
  写入不可变 protocol；
- [ ] `P2D-DEC-01` 实现并核验 system reconstruction 输出；
- [ ] `P2D-DEC-02` 为 BBB/HHH/HHB 从 paired initialization 训练 standardized decoder；
- [ ] `P2D-QA-01` 验证 standardized decoder optimizer/data/epoch/selection 完全一致；
- [ ] `P2D-RUN-01` 完整运行 seed 43，不因结果修改 seed 44；
- [ ] `P2D-RUN-02` 完整运行 seed 44，不追加第三个 confirmation seed；
- [ ] `P2D-GATE-01` 检查 accuracy、standardized reconstruction、z-rank 和
  h2→z compensation；
- [ ] `P2D-QA-02` 检查 pairing、checksum、resume、artifact completeness 和
  `test_samples_accessed=0`；
- [ ] `P2D-DECIDE-01` 仅在两个 seeds 都通过时批准正式 Phase 4。

#### P3 — Validation-only 超参数选择

- [x] `P3-CFG-01` 创建 tuning seed 42 的 Hebbian LR configs；
- [x] `P3-RUN-01` 依次运行 4 个 Hebbian LR trials；
- [x] `P3-SEL-01` 用 validation linear-probe accuracy 选择 LR；
- [x] `P3-CFG-02` 基于最佳 LR 创建 3 个 winner-fraction configs；
- [x] `P3-RUN-02` 依次运行 winner-fraction trials；
- [x] `P3-SEL-02` 选择 Hebbian 最终 validation config；
- [x] `P3-CFG-03` 创建 BP LR 与 weight-decay trial configs；
- [x] `P3-CHECK-01` 核对 BP/Hebbian trial 数均不超过 8；
- [x] `P3-CHECK-02` 检查所有 tuning outputs 不含 test metrics；
- [x] `P3-LOG-01` 保存完整 trial table，包括失败 trials；
- [x] `P3-FREEZE-01` 生成最终 `hebbian_main.yaml`；
- [x] `P3-FREEZE-02` 接收并校验最终 `bp_main.yaml`；
- [x] `P3-FREEZE-03` 保存两份 resolved config hash；
- [x] `P3-FREEZE-04` 在决策日志中记录选择理由。

#### P4 — Q1 formal clean performance

旧 seeds 0–1 BP/Hebbian/random 输出只验证了运行、恢复和统计管线；不得作为
新矩阵的已完成 seed。

- [ ] `P4-MATRIX-01` 生成 BBB/HHH/HHB/HBB/Full-Random seeds 0–4 manifest；
- [ ] `P4-MATRIX-02` 生成 RBB/RRB seeds 0–4 matched-control manifest；
- [ ] `P4-RUN-01` 完成全部 encoder/representation training；
- [ ] `P4-RUN-02` 完成所有 frozen linear probes；
- [ ] `P4-RUN-03` 保存各方法 system reconstruction；
- [ ] `P4-RUN-04` 完成全部 standardized-decoder training/evaluation；
- [ ] `P4-QA-01` 检查每个 run 的 config、hash、checkpoint、日志和 test-access audit；
- [ ] `P4-QA-02` 检查 paired seeds 使用相同初始化、batch order 和 probe protocol；
- [ ] `P4-QA-03` 检查 standardized decoders 使用相同 initialization、optimizer、
  data、epoch 和 validation selection；
- [ ] `P4-METRIC-01` 汇总 accuracy、macro-F1 和 classification CE；
- [ ] `P4-METRIC-02` 分开汇总 system/standardized reconstruction MSE；
- [ ] `P4-METRIC-03` 汇总 encoder/decoder/probe dataset passes、samples seen 与 wall-clock；
- [ ] `P4-METRIC-04` 计算 epoch/samples-seen/wall-clock AULC；
- [ ] `P4-STAT-01` 输出五个预注册 paired contrasts；
- [ ] `P4-STAT-02` 计算 mean±SD、effect size 和 paired bootstrap 95% CI；
- [ ] `P4-FIG-01` 绘制 learning curves 与 Hebbian-depth dose plot；
- [ ] `P4-TABLE-01` 生成 clean performance 与双 reconstruction 主表；
- [ ] `P4-NOTE-01` 写出 Q1 结论、限制及 Hybrid 非纯 Hebbian 声明。

#### P5 — Q2 layerwise representation

- [ ] `P5-DATA-01` 生成每类 200 张的 2,000-image subset manifest；
- [ ] `P5-DATA-02` 验证 BBB/HHH/HHB/HBB/RBB/RRB 使用相同 sample IDs 和顺序；
- [ ] `P5-EXT-01` 提取并保存 `input/h1/h2/z`；
- [ ] `P5-EXT-02` 保存 labels、sample IDs、seed 和 checkpoint ID；
- [ ] `P5-QA-01` 验证表示数组无 NaN/Inf 且 shape 正确；
- [ ] `P5-METRIC-01` 计算每层 activation sparsity；
- [ ] `P5-METRIC-02` 计算每层 active-neuron ratio；
- [ ] `P5-METRIC-03` 计算每层 effective rank；
- [ ] `P5-METRIC-03B` 计算 stable rank、covariance spectrum 与 rank ratio；
- [ ] `P5-METRIC-04` 计算 within-class distance；
- [ ] `P5-METRIC-05` 计算 between-class distance 与 separability ratio；
- [ ] `P5-METRIC-06` 计算 silhouette score；
- [ ] `P5-METRIC-07` 训练每层统一 linear probe；
- [ ] `P5-METRIC-08` 计算每层 k-NN accuracy；
- [ ] `P5-METRIC-09` 计算易混类别 centroid distances；
- [ ] `P5-METRIC-10` 计算 h2→z compensation ratio 并做 paired-seed 汇总；
- [ ] `P5-FIG-01` 用固定参数绘制 PCA；
- [ ] `P5-FIG-02` 用固定 seed/参数绘制 UMAP；
- [ ] `P5-FIG-03` 绘制 confusion matrices；
- [ ] `P5-OPT-01` 可选计算跨规则 layerwise CKA；
- [ ] `P5-TABLE-01` 生成 method × layer 指标表；
- [ ] `P5-NOTE-01` 写出 Q2 的结果摘要与限制。

#### P6 — Q3 noise robustness

- [ ] `P6-NOISE-01` 实现 deterministic Gaussian noise；
- [ ] `P6-NOISE-02` 实现 deterministic salt-and-pepper noise；
- [ ] `P6-NOISE-03` 实现 deterministic pixel masking；
- [ ] `P6-NOISE-04` 使用稳定 hash 绑定 sample ID、type 和 severity；
- [ ] `P6-TEST-01` 测试相同 noise key 输出逐像素一致；
- [ ] `P6-TEST-02` 测试不同模型读取相同 noisy tensors；
- [ ] `P6-TEST-03` 测试输出始终在 `[0,1]`；
- [ ] `P6-RUN-01` 完成 Gaussian severity × model × seed 评估；
- [ ] `P6-RUN-02` 完成 salt-and-pepper 补充评估；
- [ ] `P6-RUN-03` 完成 pixel masking 补充评估；
- [ ] `P6-METRIC-01` 计算 noisy accuracy 与 macro-F1；
- [ ] `P6-METRIC-02` 计算 absolute/relative degradation；
- [ ] `P6-METRIC-03` 计算 clean/noisy representation cosine；
- [ ] `P6-METRIC-04` 计算 prediction JS divergence；
- [ ] `P6-METRIC-05` 分开计算 system/standardized reconstruction degradation；
- [ ] `P6-STAT-01` 汇总 seed-level paired degradation；
- [ ] `P6-FIG-01` 绘制 accuracy–severity curves；
- [ ] `P6-FIG-02` 绘制 representation-stability curves；
- [ ] `P6-NOTE-01` 写出 Q3 的结果摘要与限制。

#### P6C — Stage 1C effective-rank metric audit（Q4 前置）

- [x] `P6C-DATA-01` 复用并验证同一 2,000-image validation manifest；
- [x] `P6C-EXT-01` 提取 `z_pre_wta` 与同源 `z_post_wta`；
- [x] `P6C-EXT-02` 构造 dataset-centered、per-sample L2-normalized 和
  class-centered `z`；
- [x] `P6C-EXT-03` 保存 `h1/h2/z` singular-value spectra；
- [x] `P6C-AXIS-01` 分离并验证 convolutional channel-health view 与
  sample-representation flatten view；
- [x] `P6C-QA-01` 验证 rows=observations、columns=features，covariance
  shape 为 feature × feature；
- [x] `P6C-QA-02` 验证 dataset-level centering 明确且可复算；
- [x] `P6C-QA-03` 运行 float64 epsilon/数值阈值敏感性检查；
- [x] `P6C-QA-04` 验证每类 200 张、sample IDs 唯一且顺序匹配；
- [x] `P6C-QA-05` 验证 checkpoint/probe hash 不变且 test access 为 0；
- [x] `P6C-METRIC-01` 保存 covariance eigenvalue spectra；
- [x] `P6C-METRIC-02` 保存 participation-ratio rank、stable rank 和 rank ratio；
- [x] `P6C-METRIC-03` 保存 per-class effective rank；
- [x] `P6C-METRIC-04` 保存 between-class 与 within-class covariance rank；
- [x] `P6C-METRIC-05` 报告原 frozen linear-probe accuracy；
- [x] `P6C-DEC-01` 按 pre/post-WTA 规则记录 metric-validity 和机制判断；
- [x] `P6C-NOTE-01` 保存审计报告并更新 live status；不训练、不调参、不重开 Stage 1B。

#### P7 — Q4 update mechanism

- [x] `P7-SNAP-01` 加载并验证 `conv1_end` frozen snapshot；
- [x] `P7-SNAP-02` 加载并验证 `conv2_end` frozen snapshot；
- [x] `P7-SNAP-03` 加载并验证 `conv3_end` frozen snapshot；
- [x] `P7-BATCH-01` 生成并保存 50 个固定 update-analysis batch IDs；
- [x] `P7-DEC-01` 为每个 snapshot 从 paired initialization 训练 reference decoder；
- [x] `P7-REF-01` 实现 reconstruction raw negative gradient；
- [x] `P7-REF-02` 明确排除 optimizer state、momentum 和 weight decay；
- [x] `P7-HEBB-01` 在不应用更新时生成 Hebbian candidate；
- [x] `P7-REC-01` 定义 update record dtype、shape 和 metadata；
- [x] `P7-REC-02` 保存 layer/snapshot/batch/rule/update norm；
- [x] `P7-REC-03` 保存 snapshot hash 并验证分析前后不变；
- [x] `P7-TEST-01` 测试同向 cosine=1；
- [x] `P7-TEST-02` 测试反向 cosine=-1；
- [x] `P7-TEST-03` 测试正交 cosine=0；
- [x] `P7-TEST-04` 测试零向量与 epsilon 行为；
- [x] `P7-TEST-05` 测试 constant updates 的 SNR 边界；
- [x] `P7-METRIC-01` 计算 batch-level alignment；
- [x] `P7-METRIC-02` 计算 norm ratio；
- [x] `P7-METRIC-03` 计算 mean updates 与最优缩放 `alpha*`；
- [x] `P7-METRIC-04` 计算 scale-matched relative bias；
- [x] `P7-METRIC-05` 分别计算 Hebbian/BP SNR；
- [x] `P7-METRIC-06` 按 layer 与 snapshot 汇总不确定性；
- [x] `P7-FIG-01` 绘制 alignment/norm/bias/SNR panels；
- [ ] `P7-CORR-01` 与 accuracy/separability/robustness 做探索性相关分析；
- [x] `P7-NOTE-01` 写出 seed-42 failure-case Q4 结果并避免因果措辞。
- [ ] `P7-HYBRID-01` 在正式 paired seeds 上分析 HHB/HBB rule boundaries；
- [ ] `P7-HYBRID-02` 与 BBB/HHH matched snapshots 做逐层比较；

#### P8 — Q5/Q6 dimension 与 asymmetry

- [ ] `P8-DIM-01` 生成 `L=[16,32,64,128]` configs；
- [ ] `P8-DIM-02` 验证每个 config 的 bottleneck shape；
- [ ] `P8-DIM-03` 运行 BBB/HHH/HHB/HBB dimension × seeds；
- [ ] `P8-DIM-04` 在关键 dimensions 运行 RBB/RRB matched controls；
- [ ] `P8-DIM-05` 汇总 clean/noisy/probe/separability/effective-rank；
- [ ] `P8-ARCH-01` 创建 early-heavy config；
- [ ] `P8-ARCH-02` 创建 balanced config；
- [ ] `P8-ARCH-03` 创建 late-heavy config；
- [ ] `P8-ARCH-04` 自动计算 encoder/decoder/total parameter counts；
- [ ] `P8-ARCH-05` 断言 encoder parameter range/mean <1%；
- [ ] `P8-ARCH-06` 运行 BBB/HHH/HHB/HBB architecture × seeds；
- [ ] `P8-ARCH-07` 在关键 architectures 运行 RBB/RRB matched controls；
- [ ] `P8-METRIC-01` 计算每个 metric 的 relative-to-balanced change；
- [ ] `P8-METRIC-02` 计算 sensitivity score；
- [ ] `P8-STAT-01` 检验 learning-rule × architecture interaction；
- [ ] `P8-REP-01` 提取每个 architecture 的 h1/h2/z；
- [ ] `P8-REP-02` 比较 layerwise geometry 与 CKA；
- [ ] `P8-FIG-01` 绘制 latent-dimension interaction plots；
- [ ] `P8-FIG-02` 绘制 architecture interaction plots；
- [ ] `P8-NOTE-01` 分别写出 Q5 与 Q6 的结果和机制解释。

#### P9 — 扩展实验门禁

- [ ] `P9-GATE-01` 确认 Q1–Q6 主结果和必要图表已完成；
- [ ] `P9-GATE-02` 记录剩余时间、GPU/CPU 预算和优先级；
- [ ] `P9-PLAN-01` 为 CIFAR-10 建立独立 config/version，不修改 v1；
- [ ] `P9-PLAN-02` 为 non-stationary classes 写出数据顺序和 forgetting metric；
- [ ] `P9-PLAN-03` 为第二 Hebbian rule 写出单一变量消融方案；
- [ ] `P9-PLAN-04` 为 noisy training 区分 train corruption 与 test corruption；
- [ ] `P9-PLAN-05` 为 local decoder rule 明确学习信号和 biological-plausibility 声明；
- [ ] `P9-DECIDE-01` 只选择通过资源评审的扩展实验；
- [ ] `P9-REPORT-01` 未执行的扩展明确列为 future work，不混入主结论。

---

## 6. 最小可交付范围（MVP）与正式主实验

### 开发 MVP

- [ ] `HebbianLinear` 公式验证；
- [ ] 单层 `HebbianConv2d`；
- [ ] MNIST 单 seed smoke test；
- [ ] frozen encoder + linear probe；
- [ ] collapse diagnostics。

### Summer Camp 正式最小结果

- [ ] 3 层 convolutional encoder/autoencoder 协议；
- [ ] BBB、HHH、HHB、HBB 与 Full Random 核心对照；
- [ ] RBB/RRB matched random-prefix controls，用于判断浅层 Hebbian 的净价值；
- [ ] 至少 5 paired seeds；
- [ ] clean classification + system/standardized reconstruction；
- [ ] 逐层 representation 定量分析；
- [ ] Gaussian noise severity curve；
- [ ] 至少 3 个 latent dimensions；
- [ ] 至少 3 个 encoder asymmetry 配置；
- [ ] conv1/conv2/conv3-end update alignment、scale-matched bias 和 SNR；
- [ ] 完整 config、checkpoint、metrics 和复现实验命令。

Salt-and-pepper、masking、CIFAR-10 与 non-stationary learning 可在时间不足时降为补充材料。

---

## 7. 实验记录模板

每次正式运行增加一行；失败实验也要保留并注明原因。

| Run ID | Date | Git commit | Model | Layer rules | Reconstruction protocol | Dataset | Arch ID | Latent dim | Seed | Config path | Status | Key result | Notes |
|---|---|---|---|---|---|---|---|---:|---:|---|---|---|---|
| example | YYYY-MM-DD | hash | HHB | H/H/B | system + standardized | MNIST | balanced | 64 | 0 | `configs/...yaml` | planned | — | — |

### 单次运行检查

- [ ] Git commit 与 working tree 状态已记录；
- [ ] config 已复制到结果目录；
- [ ] seed 覆盖 Python/NumPy/PyTorch/DataLoader；
- [ ] train/val/test sample IDs 已保存；
- [ ] 最佳 checkpoint 只依据 validation metric 选择；
- [ ] 无 NaN/Inf；
- [ ] weight norm、activation、sparsity、winner entropy 已记录；
- [ ] encoder 冻结测试通过；
- [ ] system 与 standardized decoder checkpoints/metrics 分开保存；
- [ ] standardized decoder 前后 encoder checksum 不变；
- [ ] representations 带 sample ID 和 label；
- [ ] 主表示学习配置中 `target_clamping=false`，encoder trainer 未读取 label；
- [ ] 运行由 CLI + resolved config 启动，不依赖 notebook state；
- [ ] update analysis 使用 frozen snapshot 和已保存的 batch IDs；
- [ ] test 结果没有用于调参；
- [ ] 失败原因和异常日志已记录。

---

## 8. 每周进展记录

| 周次/日期 | 本周目标 | 已完成 | 阻塞项 | 决策/变更 | 下周动作 |
|---|---|---|---|---|---|
| Week 1 | 冻结接口；HebbianConv2d 测试 |  |  |  |  |
| Week 2 | 3 层 encoder smoke test |  |  |  |  |
| Week 3 | 调参与 5 seeds clean runs |  |  |  |  |
| Week 4 | representation 与 noise |  |  |  |  |
| Week 5 | dimension/asymmetry sweep |  |  |  |  |
| Week 6 | update analysis、图表与报告 |  |  |  |  |

### 决策日志

| Date | Decision | Reason | Alternatives rejected | Impact |
|---|---|---|---|---|
| 2026-07-17 | 正式主实验使用 3 层卷积 encoder；MLP 仅作开发验证 | 与研究题目和深度退化假设一致 | 以 MLP 作为正式模型 | 增加早期实现工作，但避免研究对象偏离 |
| 2026-07-17 | 分类统一使用 frozen encoder + linear probe | 隔离 representation quality | BP end-to-end classifier vs Hebbian probe | 提升公平性 |
| 2026-07-17 | update cosine 命名为 alignment，主 bias 使用 scale-matched relative bias | 避免统计概念和更新尺度混淆 | 将 cosine 直接称为 bias | Q4 结论更严谨 |
| 2026-07-17 | 教程只作教学原型，不作为组内代码标准 | notebook 含教学简化和隐式状态 | 直接复制 notebook | 需要模块化迁移与测试 |
| 2026-07-17 | BP/Hebbian 共用 ConvAutoencoder，只切换 trainer/update rule | 隔离 learning rule 变量 | 维护两套模型 | 减少结构漂移 |
| 2026-07-17 | Hebbian 使用显式 local update，不使用 custom autograd backward | 便于控制更新时点并记录候选更新 | 在 backward 中替换梯度 | Q4 分析更可验证 |
| 2026-07-17 | 主实验移除 target clamping | 防止标签信息进入 encoder | 保留教程的 clamping | 保持无监督表示比较 |
| 2026-07-23 | 发布 Phase 0 v1.1 addendum；BP Adam learning rate 冻结为 `0.003` | validation-only tuning 已选定该值；避免重跑 BP tuning，并补齐正式复现治理 | 沿用父协议 `0.001`；重新 tuning BP | 正式运行改用 `configs/formal/` 和 canonical source ref |
| 2026-07-23 | 在 Q1/Q4 多 seed 前先执行 representation health gate | `active_neuron_ratio` 接近 `winner_fraction` 时，旧 collapse threshold 可能把预期 top-k 稀疏性误判为病理 collapse | 直接继续 Q1 seeds；仅依据单一 active-ratio 阈值判定 | Stage 1 必须同时检查 winner concentration、dead units、entropy、variance 与 effective rank |
| 2026-07-23 | Formal probe 在 validation checkpoint 冻结后才首次读取 test | 防止 representation extraction 提前接触 test | 训练 probe 前一次性缓存 train/validation/test | 所有正式 test 只作最终一次评估；tuning 完全不构造 test features |
| 2026-07-23 | Stage 1 representation health gate 判定当前 Hebbian selected config 为 FAIL | seed-42 `z` 在 2,000 张 validation 图上固定由同一 7/64 units 获胜，effective rank=1.0186；Q1 seeds 0–1 重复 | 将 `0.109375≈winner_fraction` 直接解释为正常稀疏；继续 Q4/Q1 | 必须先执行 Stage 1B；collapse 定义分离 per-location density、dataset-wide coverage 与 representation rank |
| 2026-07-23 | Stage 1B 冻结为 `COMPLETED — NO CANDIDATE PASSED` | 两轮八个预注册 validation-only candidates 均未同时通过 unchanged health gate 和 accuracy floor；test access 为 0 | 继续增加 v3/v4；从失败候选中强行选择一个 | 不生成 replacement formal config；保留完整负结果并停止 repair tuning |
| 2026-07-23 | 在 Q4 前增加 Stage 1C effective-rank metric audit | Stage 1B 显示 winner coverage 可提高但 rank 仍低，需要区分 WTA 压缩、filter 重复和 metric/axis 问题 | 重新训练或继续调参；未经审计直接解释 rank≈1 | 复用既有 checkpoint 和 2,000-image validation subset，仅审计 pre/post-WTA、centering、axes、spectra、epsilon 和 class covariance |
| 2026-07-23 | Stage 1C 完成，metric validity=PASS，机制分类为 `PRE_AND_POST_WTA_NEAR_ONE` | pre-WTA PR=1.0186、post-WTA PR=1.0000；主结果不受 epsilon 主导；frozen probe 在同 subset accuracy=0.9040 | 将低 rank 归因于 WTA；将低 PR 直接等同于没有分类信息 | Stage 1/1B 的 raw-covariance anisotropy 结论保留但缩窄解释；现有 seed-42 checkpoint 仅作为 Q4 failure-case snapshot |
| 2026-07-23 | Stage 2 / Q4 seed-42 tooling gate 完成并通过 | 三个 frozen layer-boundary snapshots、50 个固定 batches、raw BP 与 raw/effective Hebbian updates、完整 metrics/tensors、62 tests、零分析 optimizer step、零 test access、checksum 不变 | 把单失败案例当作正式多 seed Q4；把 decoder 训练 step 混同为分析 optimizer step | Q4 工具可复用；seed-42 仅提供 failure-case mechanism evidence，正式跨 seed Q4 与相关分析仍未完成 |
| 2026-07-25 | 完成 notebook-inspired output-filter update-centering 单候选审计；候选判定 `DOES_NOT_RESOLVE_FAILURE` | validation accuracy=0.1944，三层 health gate 全失败，Enc3 alignment 变为负值；70 tests、零 test access、分析 checksum 不变 | 把 bias 数值略降解释为方向改善；继续追加候选；替换原始 baseline | 候选不具备 replacement 资格；Stage 1B 和门禁保持冻结，结果仅作为单 seed 机制负证据 |
| 2026-07-25 | 按冻结 follow-up 决策树选择 Branch D | performance/health/direction 均无支持，完整性检查虽通过但不改变联合门禁失败 | B1/B2/B3/C 的额外修复；进入 Stage 3 | 停止 current Oja repair；原始配置仅作 health-gate failure-case baseline |
| 2026-07-25 | Stage 2C Hybrid depth ablation 完成，选择 Outcome D | 两个 hybrid 均过 performance floor 且修复深层 rank，但完整 health gate 仍被早期 Hebbian layers 阻断；Full BP 也未通过全部严格 health checks | 把单 seed 当正式结论；修改 threshold；自动运行 confirmation | 冻结 Hybrid-HHB 为最小 BP intervention confirmation candidate，需另行批准 |
| 2026-07-27 | 项目扩展为 Full BP、Full Hebbian 与 Minimal Hybrid Credit Assignment 比较 | seed-42 显示 BP suffix 可定位并修复深层表示瓶颈；固定三层结构可形成 0/1/2/3 Hebbian-depth 梯度 | 用 HHB 替代 HHH；直接缩短 encoder 总深度 | HHH 保留为原始基线；新增 HBB、HHB、RBB、RRB，并先执行 seeds 43/44 confirmation |
| 2026-07-27 | Reconstruction 同时报告 system 与 standardized-decoder protocols | Hybrid 的 BP suffix 可与 decoder 联合适应，system MSE 不能单独归因于 encoder information | 只报告训练流程自带 decoder；只比较分类 | encoder-representation 重建主张以 paired standardized decoder 为主，并保留 system performance |

---

## 9. 推荐代码结构

```text
project/
├── configs/
│   ├── common_mnist.yaml
│   ├── hebbian_main.yaml
│   ├── bp_main.yaml
│   ├── latent_sweep.yaml
│   ├── asymmetry_sweep.yaml
│   └── noise_eval.yaml
├── schemas/
│   ├── config_schema.py
│   └── result_schema.py
├── data/
│   └── splits/
│       └── mnist_split_v1.npz
├── models/
│   ├── conv_autoencoder.py
│   ├── encoder.py              # BP/Hebbian 共享 forward
│   ├── decoder.py              # BP/Hebbian 共享 forward
│   └── linear_probe.py
├── learning_rules/
│   ├── base.py
│   ├── hebbian.py              # 显式 compute/apply local update
│   └── backprop.py             # 标准 autograd/optimizer
├── training/
│   ├── train_representation.py # 由 config 切换 learning rule
│   ├── train_decoder.py
│   └── train_linear_probe.py
├── evaluation/
│   ├── evaluate_clean.py
│   ├── evaluate_noise.py
│   ├── extract_representations.py
│   └── representation_metrics.py
├── analysis/
│   ├── update_recorder.py
│   ├── bp_reference.py
│   └── update_metrics.py
├── visualization/
│   ├── plot_learning_curves.py
│   ├── plot_latent_geometry.py
│   ├── plot_robustness.py
│   └── plot_update_metrics.py
├── tests/
│   ├── test_hebbian_update.py
│   ├── test_shapes.py
│   ├── test_freeze.py
│   ├── test_no_label_leakage.py
│   ├── test_update_metrics.py
│   └── test_reproducibility.py
├── docs/
│   └── tutorial_migration.md
├── prompts/
│   ├── README.md
│   └── CP-xxx_short_name.md
├── results/
└── HEBBIAN_PROJECT_PLAN.md
```

---

## 10. 结果目录与文件规范

```text
results/{experiment_id}/{model}/{arch_id}/seed_{seed}/
├── config_resolved.yaml
├── metadata.json
├── split_manifest_ref.json
├── metrics.csv
├── encoder.pt
├── decoder_system.pt
├── decoder_standardized.pt
├── linear_probe.pt
├── representations.npz
├── update_records.npz
├── update_metrics.csv
└── logs.txt
```

`metrics.csv` 至少包含：

```text
run_id, model, layer_rules, seed, split, stage, epoch, step, samples_seen,
reconstruction_protocol, reconstruction_loss, classification_ce, accuracy, macro_f1,
weight_norm, update_norm, activation_mean, activation_sparsity,
active_neuron_ratio, winner_entropy, wall_time_sec
```

`update_records.npz` 至少保存或可重构：

```text
run_id, rule, layer, checkpoint_id, checkpoint_step, batch_ids,
candidate_update, update_norm, bp_reference_loss, epsilon,
snapshot_hash, target_clamping, optimizer_state_included
```

如果完整 update tensor 过大，可保存每层均值、平方和、样本数以及可复查的小规模 batch tensor；但必须足以重算 cosine、bias、variance 和 SNR。`metadata.json` 还需记录教程 provenance，正式结果不能只存在 notebook 输出 cell 中。

禁止覆盖旧结果。`experiment_id`、resolved config、Git commit 和环境版本必须足够支持复现。

---

## 11. 最终图表与表格清单

- [ ] Table 1：共同架构、训练协议与参数量；
- [ ] Table 2：BBB/HHH/HHB/HBB 与随机控制的 clean classification，
  mean ± 95% CI；
- [ ] Table 3：system 与 standardized-decoder reconstruction 并列表；
- [ ] Figure 1：Full BP/Full Hebbian/Hybrid learning curves；
- [ ] Figure 2：h1/h2/z 的 PCA/UMAP；
- [ ] Figure 3：逐层 separability、linear probe、effective rank；
- [ ] Figure 4：accuracy vs noise severity；
- [ ] Figure 5：representation stability vs noise severity；
- [ ] Figure 6：latent dimension × learning rule；
- [ ] Figure 7：architecture × learning rule interaction；
- [ ] Figure 8：update alignment、scale-matched bias、norm ratio 与 SNR；
- [ ] Figure 9：Hebbian depth（0/1/2/3）与 matched-random contrasts；
- [ ] Appendix：每 seed 结果、失败运行、超参搜索空间和额外 reconstruction。

每张主图都应标明 n、误差条定义、seed、数据 split 与统计单位。

---

## 12. 风险与应对

| 风险 | 早期信号 | 应对 |
|---|---|---|
| 深层 Hebbian 表示塌缩 | effective rank≈1、winner entropy 低 | greedy layer-wise、homeostasis、kernel normalization、降低 LR |
| 比较不公平 | BP 使用 end-to-end 分类而 Hebbian 使用 probe | 两者统一 frozen encoder + 同一 probe |
| 卷积 latent dimension 含糊 | 只报告 channel 数 | 报告 `C×H×W` 与 flatten 总维度 |
| Decoder/suffix 联合适应混杂重建结论 | HHB/HBB system MSE 明显改善，但 BP suffix 与 decoder 联合训练 | 同时报告 system 与 paired standardized-decoder reconstruction；encoder-information 主张以后者为主 |
| 用 Full Random 证明浅层 Hebbian 有价值 | HBB/HHB 只与完全随机 encoder 比较 | 增加 RBB/RRB，分别匹配 BP suffix 与训练预算 |
| 把 Hybrid 当成纯 Hebbian | 图表把 HHB/HBB 合并进 Hebbian 结果 | 逐层标注 H/B/R；HHH 始终单列，Hybrid 只解释为 credit-assignment intervention |
| 测试集泄漏 | 根据 test accuracy 选超参 | validation-only selection；test 最后一次评估 |
| 标签泄漏到 Hebbian encoder | forward/trainer 接收 y 或启用 target clamping | 主训练接口只接收 x；配置断言与无标签测试 |
| custom autograd 隐式改写梯度 | backward 行为难以追踪，optimizer 可能二次更新 | 使用显式 compute/apply local update；Hebbian 参数不进入 optimizer |
| notebook 隐式状态 | 重启 kernel 后结果变化、cell 顺序影响输出 | 教程只作来源；正式运行全部使用模块化 CLI + resolved config |
| SNR 混入训练漂移 | 采样 batch 之间模型权重继续改变 | frozen snapshot 上采样 candidate updates，固定 batch IDs |
| BP reference 定义漂移 | 有时含 momentum/decay，有时只用 gradient | 主 reference 固定为 raw reconstruction negative gradient |
| 计算量失控 | dimensions × architectures × noise × seeds 全笛卡尔积 | 先在单 seed 筛选，再冻结配置做 5 seeds confirmatory runs |
| 只靠可视化下结论 | UMAP 看似分离但 probe 很差 | 图形必须搭配定量指标 |
| “生物合理”表述过强 | decoder/probe 仍使用 BP | 明确哪些层用 Hebbian、哪些组件用 BP |

---

## 13. 最终验收（Definition of Done）

- [ ] Hebbian rule、competition、normalization 和训练顺序有公式与实现说明；
- [ ] 教程路径/版本/hash 与迁移映射已记录，但主实验不依赖 notebook 执行；
- [ ] 正式模型确为约定的 3-layer convolutional architecture；
- [ ] BBB/HHH/HHB/HBB/RBB/RRB 共用同一 forward model、parameter shapes 与
  paired initial `state_dict`，只改变逐层训练规则或 frozen-random 状态；
- [ ] Hebbian update 为显式 local update，未通过 custom autograd backward 注入；
- [ ] 主表示学习关闭 target clamping，标签只进入 frozen linear probe 与评估；
- [ ] 与 BP 使用相同数据、encoder shape、latent size、probe、seed 和评估样本；
- [ ] 所有主结果至少包含 5 个 paired seeds 与不确定性；
- [ ] HHB seeds 43/44 validation-only confirmation 均通过后才启动正式主矩阵；
- [ ] Q1–Q6 每题至少有一个定量实验和一个可解释结论；
- [ ] clean/noisy、classification/reconstruction 指标没有混用；
- [ ] system 与 standardized-decoder reconstruction 分开记录、并列报告；
- [ ] HBB/RBB 与 HHB/RRB matched contrasts 已完成，Hebbian-value 结论不依赖 Full Random；
- [ ] representation 分析覆盖 h1、h2、z，而不只 bottleneck；
- [ ] update analysis 在 frozen snapshot 和相同 batch IDs 上完成，并正确区分 BP baseline/reference、alignment、bias 与 SNR；
- [ ] 结果可由固定 config 和 checkpoint 重现；
- [ ] 失败实验、限制与仍使用 BP 的组件已如实报告；
- [ ] 主要 AI coding prompts 已按 Task ID 归档，记录输出文件、人工修改和验证结果；
- [ ] prompt 记录中不含 token、密码、个人数据或大段二进制内容；
- [ ] 图表、表格和代码路径可交接给组员；
- [ ] README 中有一条命令可运行 smoke test，一条命令可复现主实验。

---

## 14. 下一步（按顺序执行）

1. 为 Stage 2D 写入不可变 protocol：seeds 43/44、BBB/HHH/HHB paired
   references、双 reconstruction 与全部阈值；
2. 在不访问 test、不调参的条件下完成两个 HHB confirmation seeds；
3. 仅在两者均通过时冻结正式 BBB/HHH/HHB/HBB/RBB/RRB configs 和
   seeds 0–4 manifest；
4. 完成 Q1 formal clean performance、system reconstruction 与
   standardized-decoder reconstruction；
5. 复用同一 checkpoints 完成 Q2 layerwise representation 与 Q3 robustness；
6. 将 Q4 frozen-snapshot tooling 扩展到 HHB/HBB rule boundaries 和 paired seeds；
7. 最后执行 BBB/HHH/HHB/HBB latent-dimension 与 architecture-asymmetry matrices；
8. 补充缺失的 tutorial provenance 与 BP 队友书面确认；这些不改变主实验门禁顺序；
9. 每次 task 状态变化立即更新 `PROJECT_STATUS.md` 和对应证据，不在项目末尾补记。

---

## 15. Prompt Used

本节用于规划和记录会实质影响代码、实验设计或分析结果的 AI coding prompts。普通语法询问、文字润色和不产生项目变更的对话无需记录。每个 prompt 必须关联第 5.1 节的 Task ID，完整内容保存到 `prompts/`，本节只维护索引和主要模板。

### 15.1 Prompt 使用规则

- Prompt ID 使用 `CP-001`、`CP-002` 等稳定编号，修改 prompt 不复用旧编号；
- 一个 prompt 尽量只覆盖一个紧密相关的 task group；
- prompt 必须要求先读取本计划及相关文件，不能仅凭聊天上下文编码；
- prompt 不得自行改变 `phase0-v1` 的数据、shape、loss、seed、训练预算或指标定义；
- prompt 必须写出预期文件、禁止事项和验证命令；
- AI 输出不能因为“看起来合理”就标记 accepted，必须运行相应测试；
- 若 AI 生成代码后有人工修改，记录修改原因，而不是覆盖原 prompt 记录；
- prompt 和结果记录不得包含密码、API token、个人数据或完整模型权重；
- 状态统一使用 `planned / used / revised / accepted / rejected`。

### 15.2 Prompt registry

| Prompt ID | WBS Task IDs | 目标 | Prompt file | Status | 主要输出 | 验证记录 |
|---|---|---|---|---|---|---|
| CP-001 | P0-CFG-01—05, P0-API-01—05 | 项目 scaffold、schema 与公共接口 | `prompts/CP-001_scaffold_schema.md` | planned | configs/schemas/interfaces | 待填写 |
| CP-002 | P0-DATA-01—05 | 固定 MNIST split manifest | `prompts/CP-002_mnist_split.md` | planned | split script/manifest tests | 待填写 |
| CP-003 | P2-MODEL-01—06 | 共享 3-layer ConvAutoencoder | `prompts/CP-003_shared_autoencoder.md` | planned | model modules/shape tests | 待填写 |
| CP-004 | P1-BASE/LIN/WTA/CONV | 显式 Hebbian local update | `prompts/CP-004_hebbian_rule.md` | planned | learning rule modules | 待填写 |
| CP-005 | P1-TEST-01—09 | Hebbian 数值与安全测试 | `prompts/CP-005_hebbian_tests.md` | planned | unit tests | 待填写 |
| CP-006 | P2-TRAIN-01—07 | greedy layer-wise trainer | `prompts/CP-006_greedy_trainer.md` | planned | trainer/checkpoints/logging | 待填写 |
| CP-007 | P2-PROBE/DEC | frozen probe 与 decoder | `prompts/CP-007_probe_decoder.md` | planned | training/evaluation modules | 待填写 |
| CP-008 | P5-* | representation extraction/metrics | `prompts/CP-008_representation.md` | planned | arrays/metrics/plots | 待填写 |
| CP-009 | P6-* | deterministic noise evaluation | `prompts/CP-009_noise_robustness.md` | planned | corruptions/evaluation/tests | 待填写 |
| CP-010 | P7-* | BP reference、cosine、bias、SNR | `prompts/CP-010_update_analysis.md` | planned | update records/metrics/tests | 待填写 |
| CP-011 | P8-* | dimension/asymmetry sweeps | `prompts/CP-011_sweeps.md` | planned | configs/runner/summary | 待填写 |
| CP-012 | P0-TEAM-*, P4-QA-* | `phase0-v1` compliance review | `prompts/CP-012_compliance_review.md` | planned | review report | 待填写 |

### 15.3 单个 prompt 文件模板

```markdown
---
prompt_id: CP-XXX
task_ids: [P?-???-??]
date: YYYY-MM-DD
tool_model: <tool/model/version if known>
status: planned
context_files:
  - HEBBIAN_PROJECT_PLAN.md
expected_outputs: []
---

# Exact prompt

<粘贴实际提交给 coding agent 的完整 prompt，不做事后改写>

# Result

- Files created:
- Files modified:
- Important decisions:

# Verification

- Commands run:
- Passed:
- Failed:
- Artifacts inspected:

# Human changes

- Manual edits after generation:
- Reason:

# Final disposition

- Status: accepted | revised | rejected
- Follow-up prompt IDs:
```

### 15.4 所有 coding prompts 共用的约束前缀

下面的前缀应放在 CP-001—CP-012 开头，再附加具体任务要求：

```text
Read HEBBIAN_PROJECT_PLAN.md before making changes. Work only on the listed WBS Task IDs.
Treat Section 4 phase0-v1 as immutable. Do not change the dataset split, architecture shapes,
latent definition, losses, seeds, training budgets, linear-probe protocol, noise realization,
or update-metric formulas unless the prompt explicitly asks for a versioned plan change.

Inspect existing files before editing and preserve unrelated work. Do not use notebook state as
a runtime dependency. Hebbian updates must be explicit local updates under torch.no_grad(), not a
custom autograd backward replacement. Representation training must not read labels or use target
clamping. Keep BP and Hebbian forward architectures shared.

Before finishing, run the smallest relevant tests, report exact commands and results, list changed
files, and identify any remaining assumptions or failures. Do not mark a WBS task complete unless
its observable artifact and acceptance test exist.
```

### 15.5 主要 coding prompt 模板

#### CP-001 — Scaffold、config schema 与接口

```text
Tasks: P0-CFG-01 through P0-CFG-05 and P0-API-01 through P0-API-05.
Create the minimal project directories, validated YAML config schema, resolved-config writer,
metadata schema, and typed interfaces for model, trainer, checkpoint, representation, and update
records. Unknown config fields must raise an error. Do not implement training algorithms yet.
Add focused tests for defaults, required fields, unknown fields, and serialization round trips.
```

#### CP-002 — MNIST split manifest

```text
Tasks: P0-DATA-01 through P0-DATA-05.
Implement a deterministic stratified split of the official MNIST training set into 50,000 train
and 10,000 validation samples with split_seed=0; preserve the official 10,000 test set. Save original
dataset indices, label counts, and integrity hashes. Add tests for sizes, disjointness, class balance,
index bounds, and identical hashes across repeated generation. Do not download data inside tests.
```

#### CP-003 — Shared 3-layer ConvAutoencoder

```text
Tasks: P2-MODEL-01 through P2-MODEL-06.
Implement one shared ConvEncoder/Decoder/ConvAutoencoder matching the exact layer table in Section
4.2. Support latent dimension L without changing spatial bottleneck 1x1. Return named h1/h2/z
representations. Implement deterministic paired initialization and parameter/shape summaries.
Add tests for L in [16,32,64,128], exact output shapes, reconstruction shape, parameter names, and
state-dict equality between learning-rule modes. Do not put learning updates inside model.forward().
```

#### CP-004 — Explicit Hebbian learning rule

```text
Tasks: P1-BASE-01 through P1-CONV-06 and P1-LOG-01 through P1-LOG-02.
Implement the phase0-v1 competitive Oja-style local rule for Linear and Conv2d layers. Separate
candidate computation from application, apply channel-wise top-k competition per sample/spatial
location, average updates over batch and spatial positions, and L2-normalize each output filter.
Return structured diagnostics. Keep all parameter mutation under torch.no_grad() and ensure Hebbian
encoder parameters cannot be updated by a BP optimizer.
```

#### CP-005 — Hebbian rule tests

```text
Tasks: P1-TEST-01 through P1-TEST-09.
Write focused unit tests using tiny deterministic tensors. Compare Linear and Conv candidate updates
with independently hand-computed references; test lr=0, compute-without-mutation, layer isolation,
filter norms, reproducibility, 500-step numerical stability, and invariance to shuffled/replaced
labels. Tests must fail if custom autograd or label-dependent behavior is introduced.
```

#### CP-006 — Greedy layer-wise trainer

```text
Tasks: P2-TRAIN-01 through P2-TRAIN-07.
Implement Conv1→freeze, Conv2→freeze, Conv3→freeze Hebbian training with 10 epochs per layer.
Save conv1_end, conv2_end, and conv3_end checkpoints, checksums, resolved configs, per-step diagnostics,
samples_seen, and wall-clock time. Resume must restore the exact stage and counters. Add a tiny-data
integration test proving frozen earlier layers do not change during later stages.
```

#### CP-007 — Frozen linear probe and decoder

```text
Tasks: P2-PROBE-01 through P2-PROBE-05 and P2-DEC-01 through P2-DEC-03.
Implement feature extraction, train-only feature standardization, a single Linear(L,10) probe with
the fixed SGD protocol, and checksum-based encoder freeze validation. Implement the Hebbian decoder
protocol from paired initialization using detached frozen features, MSE, Adam, and 10 epochs. Save
validation-selected checkpoints, test metrics, and a deterministic reconstruction grid.
```

#### CP-008 — Representation analysis

```text
Tasks: P5-DATA-01 through P5-NOTE-01.
Implement the fixed 2,000-sample stratified manifest, extraction of input/h1/h2/z with sample IDs,
and layerwise sparsity, active-neuron ratio, effective rank, within/between distances, separability,
silhouette, linear-probe, and k-NN metrics. Add deterministic PCA and UMAP plotting with shared sample
order and colors. Validate shapes/NaNs and produce a tidy layer×rule metrics table.
```

#### CP-009 — Deterministic noise robustness

```text
Tasks: P6-NOISE-01 through P6-NOTE-01.
Implement Gaussian, salt-and-pepper, and independent pixel masking corruptions using a stable seed
derived from noise_seed, sample_id, noise type, and severity. Apply noise in [0,1] and clip. Add
reproducibility/parity/range tests, then evaluate fixed clean-trained checkpoints and compute accuracy,
macro-F1, absolute/relative degradation, representation cosine, and prediction JS divergence.
```

#### CP-010 — Update mechanism analysis

```text
Tasks: P7-SNAP-01 through P7-NOTE-01.
Implement frozen-snapshot update analysis using the fixed 50 batch IDs. Train the reference decoder
per snapshot, compute raw reconstruction negative gradients without optimizer state, compute Hebbian
candidates without applying them, and persist auditable update records. Implement tested alignment,
norm ratio, alpha*, scale-matched bias, and separate Hebbian/BP SNR. Assert snapshot hashes remain
unchanged and generate layer×snapshot result panels.
```

#### CP-011 — Dimension and architecture sweeps

```text
Tasks: P8-DIM-01 through P8-NOTE-01.
Generate validated configs and a resumable runner for latent dimensions [16,32,64,128] and the three
frozen architecture IDs. Compute and assert encoder parameter counts, never overwrite runs, and emit
a complete run manifest. Aggregate clean/noisy/probe/separability/effective-rank metrics, relative-to-
balanced changes, sensitivity, rule×architecture interactions, and layerwise representation results.
```

#### CP-012 — phase0-v1 compliance review

```text
Tasks: P0-TEAM-01 through P0-TEAM-02 and P4-QA-01 through P4-QA-02.
Review the BP and Hebbian implementations against Section 4 without modifying code. Check shared
parameter names/shapes, initial hashes, forward outputs, split hash, batch order, training budgets,
probe/noise/evaluation reuse, and result schemas. Produce an evidence-backed checklist with file/line
references and commands. Classify each item as compliant, non-compliant, or not verifiable.
```

### 15.6 Prompt execution log

每次实际使用 prompt 后追加一行；同一个 Prompt ID 的重试也必须分别记录 run number。

| Date | Prompt ID | Run | Task IDs | Tool/model | Input commit | Output commit/files | Tests | Human edits | Result |
|---|---|---:|---|---|---|---|---|---|---|
| YYYY-MM-DD | CP-XXX | 1 | P?-???-?? | — | — | — | — | — | planned |
