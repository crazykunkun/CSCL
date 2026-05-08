# CSCL 频域增强项目进展总结

本文档用于记录当前 CSCL 频域增强模型的工作进展、核心逻辑架构、评测结果和后续遗留问题，方便在新的对话窗口或后续开发中继续推进。

## 1. 当前工作进展

目前已在 `CSCL/code/MultiModal-DeepFake-main` 中完成一版频域流增强模型开发。

主要新增和修改内容如下：

- 新增频域分支代码：
  - `models/frequency_branch.py`
- 修改主模型接入逻辑：
  - `models/CSCL.py`
- 新增频域增强模型说明文档：
  - `docs_frequency_guided_cscl_system.md`
- 新增架构图文件：
  - `frequency_guided_cscl_arch.mmd`
  - `frequency_guided_cscl_arch.svg`
- 新增指标树状图文件：
  - `results/metrics_tree.mmd`
  - `results/metrics_tree.svg`

当前频域流主要使用四类高频线索：

1. `SRM` 残差
2. `DCT` 高频能量
3. `FFT` 高通响应
4. `Haar` 小波高频响应

已完成一版训练实验：

- 训练数据：从 `test_2_0.json` 中抽取 `6000` 条样本
- 训练轮数：`10`
- 主要 loss 权重：
  - `Loss_sim_wgt: 5`
  - `Loss_freq_wgt: 1`
- 新模型 checkpoint：
  - `results/logfrequency_test2_6000_10epoch_sim5_freq1/checkpoint_best.pth`

已完成多个测试集评测：

- 原始测试集：`test.json`
- 去掉纯文本篡改后的测试集：`test_2_0.json`
- 去掉 500 个模糊文本篡改候选后的测试集：`test_3_0.json`

## 2. 重要文件路径

项目根目录：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main
```

数据集元数据目录：

```text
/root/autodl-tmp/datasets/DGM4/metadata
```

原模型 checkpoint：

```text
/root/autodl-tmp/model/checkpoint_49.pth
```

新频域模型 checkpoint：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/results/logfrequency_test2_6000_10epoch_sim5_freq1/checkpoint_best.pth
```

原模型 test3.0 评测结果：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/results/eval_checkpoint49_test3_0.json
```

新模型 test3.0 评测结果：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/results/eval_frequency_test2_6000_10epoch_sim5_freq1_best_test3_0_bs64.json
```

test3.0 汇总结果：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/results/result_summary_test3_0.json
```

模糊文本篡改候选文件：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/results/ambiguous_text_tamper_candidates.json
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/results/ambiguous_text_tamper_candidates.csv
```

频域流系统说明文档：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/docs_frequency_guided_cscl_system.md
```

项目进展总结文档：

```text
/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main/project_progress_summary.md
```

## 3. 原模型核心架构

原始 CSCL 模型整体结构如下：

```text
图像 RGB 输入
  → ViT / METER 图像编码器
  → 图像 token

文本输入
  → RoBERTa / METER 文本编码器
  → 文本 token

图像 token + 文本 token
  → 图文融合
  → 一致性学习模块
  → 多任务输出
      1. 真假二分类
      2. 多标签篡改类型检测
      3. 图像 bbox 定位
      4. 文本 token 定位
```

原模型的核心优势是利用图文一致性学习，同时完成检测和定位任务。

## 4. 新版频域增强架构

新版模型在原始 RGB 图像流之外新增频域流，整体结构如下：

```text
图像 RGB 输入
  ├─ RGB 主干：METER / ViT 图像编码器
  │     → RGB token
  │
  └─ 频域流
        → SRM 残差
        → DCT 高频能量
        → FFT 高通响应
        → Haar 小波高频响应
        → CNN 编码器
        → 频域 token / 频域 score

RGB token + 频域 token
  → Cross-Attention / Gate 融合
  → 频域增强后的图像 token

增强图像 token + 文本 token
  → 原 CSCL 一致性学习结构
  → 多任务输出
```

频域流设计目标：

- 用高频线索增强伪造区域定位能力。
- 让模型额外关注图像中的边缘异常、压缩异常、纹理不连续和局部伪造痕迹。
- 频域信息主要服务于图像定位和图文一致性。
- 当前实现中，频域流没有直接改变真假二分类头，因此新旧模型的 `AUC / ACC / EER` 完全一致是可以解释的。

## 5. 频域流内部逻辑

### 5.1 SRM 残差

SRM 残差用于提取图像中的高频残差信息。

其作用是抑制自然图像中的低频语义内容，突出局部异常纹理、边缘断裂和伪造痕迹。

### 5.2 DCT 高频能量

DCT 用于分析局部图像块的频率分布。

图像篡改、压缩、拼接和生成痕迹往往会改变局部高频分布，因此 DCT 高频能量可以作为伪造区域的辅助线索。

### 5.3 FFT 高通响应

FFT 高通响应用于从全局频率角度观察图像。

它强调高频成分，抑制低频结构，有助于发现局部纹理异常和生成伪影。

### 5.4 Haar 小波高频响应

Haar 小波将图像分解为不同方向的高频子带。

它可以捕捉水平、垂直和对角方向上的突变信息，对边缘异常和局部篡改区域有一定帮助。

### 5.5 CNN 编码与 token 化

四类频域特征被组合后输入轻量 CNN 编码器。

CNN 的作用包括：

- 融合不同频域特征。
- 提取局部空间模式。
- 将频域响应映射到与 RGB token 对齐的特征空间。
- 输出频域 token 和 frequency score。

### 5.6 Gate / Cross-Attention 融合

频域 token 与 RGB token 通过 cross-attention 或 gate 机制融合。

其基本思想如下：

```text
RGB token + 频域 token → Gate 网络 → gate 权重
融合特征 = RGB token + gate × 频域增强特征
```

Gate 不是硬规则滤波，而是学习一个连续权重，用来控制频域信息注入强度。

如果某个位置的频域信息对伪造定位有帮助，gate 值应更高；如果某个位置主要是背景纹理、压缩噪声或无效高频响应，gate 值应更低。

## 6. 测试集构建逻辑

### 6.1 test.json

`test.json` 是原始完整测试集。

样本数约为：

```text
50705
```

### 6.2 test_2_0.json

`test_2_0.json` 是去掉只涉及文本篡改后的测试集。

构建目的：

- 减少纯文本篡改样本对图像 IOU 分析的干扰。
- 更聚焦图像区域定位能力。

样本数：

```text
43780
```

### 6.3 test_3_0.json

`test_3_0.json` 是从原始 `test.json` 中剔除 500 条模糊文本篡改候选后得到的测试集。

剔除依据是精确匹配以下字段：

```text
id + image + fake_cls + text + fake_text_pos
```

不能只按 `id` 删除，因为 `id` 在数据集中并不唯一。

不能只按 `image` 删除，因为同一图片可能同时存在 `orig` 样本和篡改样本。

最终结果：

```text
source_count: 50705
removed_count: 500
test3_count: 50205
```

文件路径：

```text
/root/autodl-tmp/datasets/DGM4/metadata/test_3_0.json
```

## 7. 模糊文本篡改候选逻辑

已生成 500 条疑似 `text_swap` 与 `text_attribute` 边界模糊的候选样本。

输出文件：

```text
results/ambiguous_text_tamper_candidates.json
results/ambiguous_text_tamper_candidates.csv
```

筛选依据包括：

1. `text_swap` 标签，但篡改词像人名、地名、数字等实体。
2. `text_swap` 标签，但篡改比例较低，更像局部属性修改。
3. `text_attribute` 标签，但篡改比例较高，更像全局替换。
4. `text_attribute` 标签，但连续篡改 span 较长。
5. 原模型对 `text_swap` 和 `text_attribute` 的 logits 非常接近。

候选统计：

```text
候选总数：500
带模型分数：453

text_swap 但篡改词像实体/数字：269
模型 text_swap/text_attribute logits 很接近：169
text_attribute 但篡改比例高：49
text_attribute 但连续 span 长：43
```

注意：这些样本只是“候选样本”，不是自动改标签结论。若要正式修改数据集，需要人工复核并记录依据。

## 8. test3.0 上的真实评测结果

### 8.1 原模型结果

checkpoint：

```text
/root/autodl-tmp/model/checkpoint_49.pth
```

评测结果：

```text
AUC_cls      96.0300
ACC_cls      89.8257
EER_cls      10.3105
MAP          92.1025
OF1          85.4784
CF1          85.2556
IOU_score    82.8547
IOU_ACC_50   89.3576
IOU_ACC_75   85.9576
IOU_ACC_95   43.8542
F1_tok       76.8651
```

### 8.2 新频域模型结果

checkpoint：

```text
results/logfrequency_test2_6000_10epoch_sim5_freq1/checkpoint_best.pth
```

评测结果：

```text
AUC_cls      96.0300
ACC_cls      89.8257
EER_cls      10.3105
MAP          92.3126
OF1          85.3364
CF1          85.1377
IOU_score    81.4419
IOU_ACC_50   88.3498
IOU_ACC_75   83.7148
IOU_ACC_95   42.4858
F1_tok       76.7780
```

### 8.3 新旧模型变化

```text
MAP        +0.2101
OF1        -0.1420
CF1        -0.1179
IOU_score  -1.4128
IOU50      -1.0078
IOU75      -2.2428
IOU95      -1.3684
F1_tok     -0.0871
```

结论：

- 当前频域流没有提升 IOU。
- 新模型只轻微提升了 `MAP`。
- 召回相关指标略有改善，但精度和定位指标下降。
- 当前频域融合方式可能引入了额外噪声，削弱了 bbox 定位能力。

## 9. 指标解释

### 9.1 AUC_cls / ACC_cls / EER_cls

这些是真假二分类指标。

- `AUC_cls`：二分类 ROC 曲线下面积。
- `ACC_cls`：真假分类准确率。
- `EER_cls`：等错误率，越低越好。

### 9.2 mAP

`mAP` 是多标签篡改类型检测的平均精度。

它衡量模型对多个篡改类型的排序和识别能力。

### 9.3 OP / OR / OF1

`OP / OR / OF1` 是 overall 口径下的多标签指标。

- `OP`：overall precision。
- `OR`：overall recall。
- `OF1`：overall F1。

它会把所有类别、所有样本的预测结果合并后计算整体 precision、recall 和 F1。

### 9.4 CP / CR / CF1

`CP / CR / CF1` 是 class-wise 口径下的多标签指标。

- `CP`：class-wise precision。
- `CR`：class-wise recall。
- `CF1`：class-wise F1。

它会先分别计算每个类别的指标，再对类别取平均。

### 9.5 IOU_score / IOUm

`IOU_score` 或 `IOUm` 是 bbox 定位平均 IOU。

它衡量预测框和真实框的平均重叠程度。

### 9.6 IOU_ACC_50 / IOU50

`IOU_ACC_50` 表示 IOU 大于等于 0.5 的样本比例。

### 9.7 IOU_ACC_75 / IOU75

`IOU_ACC_75` 表示 IOU 大于等于 0.75 的样本比例。

### 9.8 IOU_ACC_95 / IOU95

`IOU_ACC_95` 表示 IOU 大于等于 0.95 的样本比例。

这是更严格的定位指标。

### 9.9 F1_tok

`F1_tok` 是文本 token 定位 F1。

它衡量模型是否准确找出文本中被篡改的 token 或单词位置。

## 10. 当前尚未解决的问题

### 10.1 频域流没有带来 IOU 提升

当前结果显示：

```text
IOU_score / IOU50 / IOU75 / IOU95 全部下降
```

这说明当前频域信息没有有效服务 bbox 定位。

可能原因包括：

- 频域特征与 RGB token 空间没有充分对齐。
- gate 没有真正过滤噪声。
- 频域 loss 过弱或定义不够直接。
- 频域分支学习目标和 bbox 定位目标不一致。
- 频域增强破坏了原模型已有的定位表征。

### 10.2 Loss_freq 很低且不下降

之前观察到 `Loss_freq` 初始就很低，并且多轮训练变化不明显。

可能说明：

- 当前 `Loss_freq` 过容易。
- 频域 score target 不够有效。
- loss 没有提供足够梯度。
- 频域分支没有真正影响最终 bbox。

### 10.3 真假二分类完全不变

当前新旧模型的 `AUC / ACC / EER` 完全一致。

这在当前结构下可以解释，因为频域融合没有进入 binary classification head。

如果后续希望频域流提升整体真假检测能力，需要确认频域特征是否接入二分类路径。

### 10.4 新模型提升 mAP 但降低 OF1 / CF1

当前现象是：

```text
mAP 上升，但 OF1 / CF1 下降
```

这说明模型排序能力可能略好，但固定阈值下分类结果变差。

后续可以考虑：

- 校准多标签分类阈值。
- 对不同类别设置不同 threshold。
- 输出 per-class threshold。

### 10.5 频域分支与 bbox 定位目标弱耦合

当前频域流更像是“附加特征增强”。

如果核心目标是提升 IOU，需要更直接的 bbox 监督，例如：

- 用 GT bbox 生成 patch-level mask。
- 监督频域 attention map。
- 让频域 score 预测伪造区域热图。
- 将频域特征接入 bbox regression head 前的关键 token。

### 10.6 数据策略仍不稳定

当前使用测试集抽样训练只是诊断实验，不适合作为最终论文实验主依据。

后续应回到训练集，或构造合理验证集。

`test_2_0` 和 `test_3_0` 可用于分析，但不能作为最终实验唯一依据。

### 10.7 模糊文本篡改候选需要人工复核

当前 500 条候选只是规则和模型分歧筛选结果。

如果要正式修改数据集，需要人工确认并记录修改依据。

## 11. 下一步开发建议

### 11.1 做频域分支有效性诊断

优先输出 frequency score heatmap。

需要检查：

- 频域响应是否落在 GT bbox 内。
- 高频响应是否只是集中在背景纹理或边缘区域。
- frequency score 与 bbox IOU 是否有相关性。

如果频域响应和 bbox 无关，继续训练当前结构意义不大。

### 11.2 加强 bbox 相关监督

不要只依赖当前 `Loss_freq`。

建议用 GT bbox 生成 patch-level mask：

```text
bbox 内 patch → 1
bbox 外 patch → 0
```

然后用 BCE 或 focal loss 监督 frequency localization map。

### 11.3 改变频域融合位置

当前频域增强可能注入太早或太泛。

可以尝试：

- 只在 bbox head 前融合频域特征。
- 不影响二分类、多标签和文本侧任务。
- 让频域流成为 bbox 定位专用增强分支。

### 11.4 做频域特征消融实验

建议分别测试：

```text
只 SRM
只 DCT
只 FFT
只 Haar
SRM + DCT
四路全开
```

通过消融判断哪类频域特征真正对 IOU 有帮助。

### 11.5 控制频域融合强度

建议降低 gate 初始值，让模型初始状态接近原模型。

目标是：

```text
新模型初始性能 ≈ 原模型性能
训练后只学习频域增量
```

这样可以避免一开始破坏 pretrained METER 表征。

### 11.6 复核外部结果

如果别人给出的表显示 IOU 提升，但无法提供以下内容，则不应直接采信：

- checkpoint
- config
- test set
- eval JSON
- git diff
- 评测日志
- 指标计算代码

必须能在同一测试集、同一脚本、同一 checkpoint 下复现。

## 12. 新对话窗口继续开发建议开场

可以在新对话窗口中输入：

```text
我在 /root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main 开发 CSCL 频域增强模型。
当前新增了 models/frequency_branch.py，并在 models/CSCL.py 中接入了频域流。
最新模型 checkpoint 是 results/logfrequency_test2_6000_10epoch_sim5_freq1/checkpoint_best.pth。
原模型是 /root/autodl-tmp/model/checkpoint_49.pth。
目前 test3.0 上新模型 MAP 轻微提升，但 IOU 全面下降。
请先阅读 frequency_branch.py、CSCL.py、evaluate_checkpoint_metrics.py 和 result_summary_test3_0.json，帮我诊断频域流为什么没有提升 IOU，并设计下一版改法。
```
