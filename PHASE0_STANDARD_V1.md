# Phase 0-v1 公共实验标准

这份标准用于 BP 与 Hebbian 两条实现。主实验必须复用相同的数据、模型 forward、初始化、linear probe 和评估入口；唯一核心差异是 encoder 的学习规则。

## 1. 最基础 settings

| 项目 | 固定值 |
|---|---|
| Dataset | `torchvision.datasets.MNIST` |
| Split | 官方 train 60k → stratified train 50k / validation 10k；官方 test 10k |
| Split seed | `0`，索引保存为 `data/splits/mnist_split_v1.npz` |
| Input | `ToTensor()`，范围 `[0,1]`，不做 z-score |
| Batch size | `128` |
| DataLoader | train shuffle；val/test 不 shuffle；`drop_last=false`；`num_workers=0` |
| Paired seeds | `[0,1,2,3,4]`；调参使用独立 seed `42` |
| Main model | `conv3_ae_v1`，3 Conv encoder + 3 ConvTranspose decoder |
| Main latent dim | `64`；dimension sweep `[16,32,64,128]` |
| Encoder hidden | ReLU，无 bias、BatchNorm、Dropout、pooling |
| Decoder hidden/output | ReLU / Sigmoid；hidden/output bias 开启 |
| Reconstruction | `[0,1]` 输入作为 target，pixel-mean MSE |
| Target clamping | 主实验关闭；label 不进入 representation training |
| Model initialization | paired seed；hidden Kaiming uniform；output Xavier uniform；bias=0 |
| BP training | Adam `lr=1e-3`，10 epochs，最低 validation MSE checkpoint |
| Hebbian training | 显式 local update；不通过 custom autograd backward 注入 |
| Linear probe | frozen encoder；train-feature standardization；单层 Linear(64,10) |
| Probe training | SGD `lr=0.1, momentum=0.9, weight_decay=1e-4`，30 epochs |
| Main metrics | reconstruction MSE、classification CE、accuracy、macro-F1 |
| Official results | validated YAML + Python CLI；notebook 不能作为正式入口 |

## 2. 固定模型 shape

主配置 `L=64`：

```text
B×1×28×28
  -> Conv(1,16,k3,s2,p1,bias=False) + ReLU       = h1: B×16×14×14
  -> Conv(16,32,k3,s2,p1,bias=False) + ReLU      = h2: B×32×7×7
  -> Conv(32,L,k7,s1,p0,bias=False) + ReLU       = z:  B×L×1×1
  -> ConvT(L,32,k7,s1,p0,bias=True) + ReLU       = B×32×7×7
  -> ConvT(32,16,k4,s2,p1,bias=True) + ReLU      = B×16×14×14
  -> ConvT(16,1,k4,s2,p1,bias=True) + Sigmoid    = B×1×28×28
```

“3-layer”指 encoder 的三个可学习 Conv2d。`latent_dim=L`。

## 3. 公共接口

```python
model.encode(x, return_all_layers=False)
model.decode(z)
model.reconstruct(x)
trainer.train_batch(x)  # representation training 不接收 label
extract_representations(model, loader, layers=("h1", "h2", "z"))
```

`return_all_layers=True` 返回带固定 key 的 `{"h1", "h2", "z"}` 字典。BP/Hebbian 必须共享同一个模型 class 和初始 `state_dict`。

## 4. 队友交付要求

- 训练前，相同 seed 的 BP/Hebbian model hash 相同；
- 固定 batch 的训练前 `h1/h2/z/x_hat` 相同；
- BP 仅通过 `learning_rule=bp` 切换，不复制另一套 encoder；
- 使用公共 split、probe、representation 和 evaluation modules；
- 输出 resolved config、metadata、metrics 和 checkpoints；
- 主结果不得来自 notebook cell；
- 提交说明写明 `phase0-v1 compliant`，或逐条列出偏离。

任何改变数据划分、模型 shape、loss、训练预算、probe 或噪声 realization 的修改，都必须升级标准版本并重新运行双方受影响的实验。

