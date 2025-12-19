# The Unified Model – Nucleic Acid Part 🧬  
**核酸序列表征学习 / 基础模型评测 / 剪接位点识别**

<p align="center">
  <img src="https://img.shields.io/badge/type-research%20project-lightgrey.svg" />
  <img src="https://img.shields.io/badge/modality-nucleic%20acid-blue.svg" />
  <img src="https://img.shields.io/badge/model-nucleotide--transformer-green.svg" />
  <img src="https://img.shields.io/badge/status-ongoing-orange.svg" />
</p>

---

## 🧠 这个仓库是什么 / What is this?

这是 **The Unified Model** 项目中专门针对 **核酸（DNA/RNA）模态** 的一个研究子模块。

核心目标很简单：

> **系统性地评估“核酸基础模型（foundation models）”学到的表示，  
到底能在真实的生物下游任务中做什么、做得有多好。**

目前以 **Nucleotide Transformer** 作为教师模型，  
以 **HS3D（人类剪接位点识别）** 作为主要 benchmark，  
构建了一套 **统一的数据处理 → 嵌入提取 → 下游评测流程**。

你可以把它理解为：

- 一套 **核酸版的 unified representation learning pipeline**
- 或者一个 **“核酸基础模型下游能力体检工具”**

---

## 🎯 研究目标 / Objectives

这个模块主要解决以下几个问题：

- **核酸序列如何被统一表示？**  
  不同下游任务是否可以共用同一套 embedding？

- **大型核酸基础模型是否真的“有用”？**  
  它们在高度不平衡、真实生物任务中的表现如何？

- **能否形成一个可扩展的统一框架？**  
  方便后续接入新的数据集、新任务、新模型。

---

## 🧪 使用的教师模型 / Teacher Model

当前采用的教师模型为：

- **Model**：`InstaDeepAI/nucleotide-transformer-500m-human-ref`
- **类型**：核酸 Transformer 基础模型
- **Embedding 维度**：1280
- **Tokenizer**：模型原生 nucleotide tokenizer

在本项目中：

- 使用教师模型 **冻结参数**
- 提取序列级表示（`[CLS] token embedding`）
- 作为统一的下游 DNN 输入特征

> 关注点不在“再训练一个更强的模型”，  
> 而在 **“这个基础模型已经学到了什么”**。

---

## 🧬 数据集：HS3D（Human Splice Site Dataset）

### 数据集简介

HS3D 是一个经典的人类 **剪接位点识别（splice site recognition）** 数据集。

- **模态**：DNA
- **任务类型**：二分类 / 可扩展至多分类
- **序列字符集**：A / C / G / T / N
- **窗口长度**：140 bp

### 数据格式（统一化后）

```text
hs3d_train.csv
hs3d_test.csv
```
### 📄 数据格式说明 / Data Format

在统一预处理后，HS3D 数据集以 CSV 形式组织，主要包含训练集与测试集：

```text
hs3d_train.csv
hs3d_test.csv
```

🧪 标签设计 / Label Definition

在**原始二分类任务（正 / 负）**的基础上，本项目进一步探索 多类别剪接信号建模，将样本细分为以下类别：

donor
剪接供体位点（5' splice site）

acceptor
剪接受体位点（3' splice site）

decoy
与真实剪接位点在局部模式上相似，但不发生剪接的干扰序列

background
不包含任何剪接信号的普通背景序列

该多分类设定用于：

更细粒度地刻画不同类型剪接信号的序列特征，
并评估核酸基础模型在复杂生物语义区分任务中的表示能力。

