# DDF-Net 数据接入说明 (Adapter Guide)

本文说明如何用 **DDF-Net** 生成 BCI Competition IV-2a 的合成 MI-EEG 样本，并把
它们接入本项目的双重门槛 (`adapters/load_ddf_net_synthetic.py`) 进行质量筛选。

## 1. DDF-Net 是什么

DDF-Net（*Diffusion enhancement-based dual-domain fusion attention motor imagery
electroencephalography classification network*, Engineering Applications of
Artificial Intelligence, 2026）是一个面向运动想象 EEG 分类的模型：

- 用 **去噪扩散概率模型 (DDPM)** 预训练网络，并在潜在空间对 MI-EEG 数据集做
  **扩散增强**；
- 融合频域/时域双域注意力，提取 μ/β 节律表征；
- 在 BCI IV-2a 上 90.66%、IV-2b 上 91.31%（论文报告）。

仓库与预训练权重：

- 代码：https://github.com/HyperSystemAndImageProc/DDF-Net
- 论文：https://www.sciencedirect.com/science/article/abs/pii/S0952197626002356

> ⚠️ 本仓库当前无法直接抓取 DDF-Net 的 GitHub README（网络受限），因此下面的
> 推理步骤是**基于典型 DDPM-EEG 仓库的通用流程**。请以 DDF-Net 仓库 README 中的
> 实际命令为准，替换对应的脚本名 / 权重路径 / 参数名。

## 2. 环境准备

```bash
git clone https://github.com/HyperSystemAndImageProc/DDF-Net.git
cd DDF-Net
# 以仓库 README 为准；通常是一个 requirements.txt / environment.yml
pip install -r requirements.txt
```

## 3. 获取预训练权重

- 预训练权重通常在仓库中或通过 README 中给出的链接下载（例如 Google Drive /
  Zenodo）。下载后按 README 放到指定目录（常见为 `checkpoints/` 或 `weights/`）。
- **确认权重对应的数据集是 BCI IV-2a**（22 通道、250 Hz），与本项目一致。

## 4. 本地推理生成合成样本

典型步骤（脚本名以仓库为准）：

```bash
# 1) 准备 BCI IV-2a 的 .gdf 原始数据（见根目录 README 的"数据下载说明"）
# 2) 运行 DDPM 采样 / 生成脚本，输出合成样本
python generate.py --dataset bciiv2a --checkpoint checkpoints/ddfnet_bciiv2a.pt \
                   --n-samples 144 --out samples/ddf_bciiv2a.npy
```

生成的 `.npy` 通常形状为 `[n_epochs, 22, n_times]`（或 `[n_epochs, n_times, 22]`
/ 扁平 `[n_epochs, 22*n_times]`）。本项目的 adapter 会自动归一化这三种形状。

- 若合成样本是**按类别分别保存**的（如 `left.npy` / `right.npy`），adapter 也支持
  自动赋标签（见下）。

## 5. 接入本项目门槛

```bash
# 单文件 + 标签
py adapters/load_ddf_net_synthetic.py \
    --syn samples/ddf_bciiv2a.npy \
    --syn-labels samples/ddf_bciiv2a_labels.npy \
    --subject A01 --normalize trace

# 或：左右两个文件，自动赋 left/right 标签
py adapters/load_ddf_net_synthetic.py \
    --syn samples/left.npy --syn-right samples/right.npy \
    --subject A01 --normalize trace
```

脚本会：

1. 载入真实被试（默认 A01T）校准双重门槛；
2. 对合成批计算 **Covariance 门 / Discriminative 门 / 双门** 的通过率；
3. 汇报下游 CSP+LDA 准确率：**Unfiltered → Filtered-Cov → Filtered-Dual**。

预期：双门应剔除 DDF-Net 输出中的异常样本，且筛选后的增强收益显著优于未筛选。
