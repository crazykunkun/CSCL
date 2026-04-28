# 频域引导的多模态伪造新闻检测与定位模型系统说明

## 3.2 频域引导的图文一致性学习方法

本节介绍本文实现的频域引导多模态伪造新闻检测与定位模型。该模型以原始 CSCL（Consistency Learning）框架为基础，在 METER 多模态编码器输出的图像 patch token 与图像一致性建模模块之间，引入一个轻量级多源频域流。该频域流从图像中提取 SRM 残差、DCT 高频能量、FFT 高通响应以及 Haar 小波高频成分，并将这些频域线索编码为与 RGB patch token 对齐的 frequency tokens。随后，模型通过带门控的跨注意力残差融合模块，将频域异常信息注入原有图像 token，从而为后续图像篡改区域定位和图像篡改类别识别提供补充信息。

与直接替换原始视觉编码器不同，本文采用“主干保持 + 频域残差增强”的方式。METER 仍然负责提取图文语义特征，频域分支只在图像 token 进入一致性建模之前提供辅助特征。这样可以尽量保留原模型已有的图文理解能力，同时利用频域异常对图像伪造痕迹进行补充刻画。

### 3.2.1 符号定义及问题描述

表 3-1 给出了本方法中主要符号的定义。

| 符号 | 含义 |
|---|---|
| $I$ | 输入图像 |
| $T$ | 输入文本 |
| $B$ | 批量大小 |
| $N$ | 图像 patch 数量，本文中 $N=16\times16=256$ |
| $D$ | token 特征维度，本文中 $D=768$ |
| $X_v\in \mathbb{R}^{B\times N\times D}$ | METER 输出的图像 patch tokens |
| $X_t\in \mathbb{R}^{B\times L\times D}$ | METER 输出的文本 token features |
| $X_{cls}$ | METER 输出的图文融合 CLS 特征 |
| $F\in \mathbb{R}^{B\times 8\times H\times W}$ | 多源频域特征图 |
| $Z_f\in \mathbb{R}^{B\times N\times D}$ | 频域编码器输出的 frequency tokens |
| $S_f\in \mathbb{R}^{B\times N}$ | 频域 patch 异常分数 |
| $C_f\in \mathbb{R}^{B\times N\times 64}$ | 频域一致性特征 |
| $\hat{X}_v$ | 频域增强后的图像 tokens |
| $Y_p\in\{0,1\}^{B\times N}$ | 图像 patch 级篡改标签 |
| $M_v\in\{0,1\}^{B\times N\times N}$ | 图像 patch 一致性矩阵标签 |
| $\hat{b}\in[0,1]^4$ | 预测的图像伪造区域框 |
| $b\in[0,1]^4$ | 真实图像伪造区域框 |

给定一个图文样本 $P=\{I,T\}$，模型需要同时完成五类输出：第一，判断样本是否为真实图文匹配样本；第二，判断图像是否包含 face swap 或 face attribute 类篡改；第三，判断文本是否包含 text swap 或 text attribute 类篡改；第四，定位图像中被篡改的区域框；第五，定位文本中被篡改的 token。本文重点改进的是图像侧特征表达，即通过频域流增强 $X_v$，使后续图像一致性学习和图像定位分支能够获得更多伪造痕迹信息。

从数学角度看，本方法希望学习一个函数：

$$
(\hat{y}_{bin}, \hat{y}_{multi}, \hat{b}, \hat{y}_{tok}) = f(I,T)
$$

其中 $\hat{y}_{bin}$ 表示真伪二分类结果，$\hat{y}_{multi}$ 表示多标签篡改类型预测，$\hat{b}$ 表示图像篡改区域预测框，$\hat{y}_{tok}$ 表示文本 token 级篡改定位结果。

### 3.2.2 算法流程

本文模型的整体流程如图 3-1 所示。由于本文档不插入实际图片，采用文字排列方式给出架构图。

```text
图 3-1 频域引导 CSCL 模型整体流程

                ┌────────────────────┐
Image ─────────▶│ METER Image Encoder │────▶ RGB Image Patch Tokens ─────┐
                └────────────────────┘                                  │
                                                                          ▼
Image ─────────▶ SRM / DCT / FFT / Wavelet ─▶ Frequency Encoder ─▶ Frequency Tokens
                                                                          │
                                                                          ▼
Text ──────────▶ METER Text Encoder ───────▶ Text Token Features     Frequency Guided Fusion
                                                                          │
                                                                          ▼
                                                            Fused Image Patch Tokens
                                                                          │
                                                                          ▼
                                             Image Intra-modal Consistency Modeling
                                                                          │
                                                                          ▼
                                             Image Extra-modal Consistency Modeling
                                                                          │
                                                        ┌─────────────────┴───────────────┐
                                                        ▼                                 ▼
                                                   BBox Head                   Image Multi-label Head
                                                        │                                 │
                                                        ▼                                 ▼
                                            Forged Region Box              face_swap / face_attribute

Text Token Features ─▶ Text Intra-modal Consistency ─▶ Text Extra-modal Consistency
                                                        │                 │
                                                        ▼                 ▼
                                             Text Multi-label Head   Token Localization Head
                                                        │                 │
                                                        ▼                 ▼
                                      text_swap / text_attribute   forged text tokens

METER CLS Feature ─▶ Fusion Head ─▶ Binary Classifier ─▶ Real/Fake
```

模型的训练流程可以分为六个步骤：

1. 输入图像 $I$ 和文本 $T$，通过 METER 编码器获得图像 patch tokens、文本 token features 和图文融合 CLS 特征。
2. 将图像 $I$ 送入多源频域特征提取模块，得到 8 通道频域图 $F$。
3. 将频域图送入轻量 CNN 编码器，得到 frequency tokens、frequency patch scores 和 frequency consistency features。
4. 使用 Frequency Guided Fusion 将 frequency tokens 注入 METER 图像 patch tokens，得到增强后的图像 tokens $\hat{X}_v$。
5. 将 $\hat{X}_v$ 和文本 token features 分别送入图像侧、文本侧的一致性建模模块，得到图像聚合特征和文本聚合特征。
6. 使用不同任务头输出真伪分类、多标签分类、图像框定位和文本 token 定位结果，并通过多任务损失联合训练。

### 3.2.3 多源频域特征提取

图像伪造往往会破坏原图在噪声残差、压缩频率、边缘方向高频和全局频谱上的自然分布。单一高频特征只能覆盖其中一部分伪造痕迹，因此本文采用四类互补频域特征：SRM 残差、DCT 高频能量、FFT 高通响应和 Haar 小波高频响应。

#### （1）灰度转换

频域分支首先将 RGB 图像转换为灰度图，降低颜色通道带来的冗余，使后续算子集中关注亮度纹理和局部噪声结构。代码中采用固定加权方式：

$$
G = 0.299I_R + 0.587I_G + 0.114I_B
$$

其中 $G$ 为灰度图，$I_R,I_G,I_B$ 分别表示 RGB 三个通道。

#### （2）SRM 残差提取

SRM（Spatial Rich Model）残差用于提取图像中的局部噪声异常。本文实现中使用 3 个固定的 $5\times5$ 高通卷积核，不参与训练。给定灰度图 $G$，SRM 残差为：

$$
F_{srm}^{k}=|G*K_k|,\quad k=1,2,3
$$

其中 $K_k$ 表示第 $k$ 个固定 SRM 卷积核，$*$ 表示卷积操作。SRM 特征最终得到 3 通道残差图。

该部分对应代码中的 `FixedSRMConv`。其特点是参数量为 0，提取结果稳定，主要用于捕获局部拼接、生成纹理和图像编辑带来的残差异常。

#### （3）DCT 高频能量提取

DCT 特征用于捕获块级频率异常。本文将灰度图划分为 $8\times8$ 图像块，对每个块执行二维 DCT 变换。对于一个图像块 $P$，其 DCT 系数为：

$$
D(u,v)=\alpha(u)\alpha(v)\sum_{x=0}^{7}\sum_{y=0}^{7}P(x,y)
\cos\frac{(2x+1)u\pi}{16}\cos\frac{(2y+1)v\pi}{16}
$$

其中 $u,v$ 为频率索引。本文选择满足 $u+v\geq 4$ 的高频系数，并计算均方能量：

$$
F_{dct}=\sqrt{\frac{1}{|\Omega|}\sum_{(u,v)\in\Omega}D(u,v)^2+\epsilon}
$$

其中 $\Omega=\{(u,v)|u+v\geq4\}$。该特征对应代码中的 `DCTHighFrequencyMap`，输出 1 通道 DCT 高频能量图。

DCT 特征对 JPEG 压缩、块效应和局部篡改造成的频率分布变化较敏感，适合处理新闻图像、网页图像等常见压缩场景。

#### （4）FFT 高通幅度提取

FFT 特征用于从全局频谱角度捕获异常。给定灰度图 $G$，先去除均值，再进行二维傅里叶变换：

$$
\mathcal{F}=FFT2(G-\bar{G})
$$

随后构造高通掩码 $H$：

$$
H(f_x,f_y)=
\begin{cases}
1,&\sqrt{f_x^2+f_y^2}\geq \tau\\
0,&\text{otherwise}
\end{cases}
$$

其中 $\tau$ 为高通截止频率，代码中默认为 0.35。最后通过逆变换得到空间域高频响应：

$$
F_{fft}=|IFFT2(\mathcal{F}\cdot H)|
$$

该部分对应 `FFTAmplitudeHighPass`，输出 1 通道高通响应图。它能提供全局频率视角，对重采样、生成图像和融合伪影具有一定敏感性。

#### （5）Haar 小波高频提取

Haar 小波用于提取局部方向高频。本文使用 4 个固定 $2\times2$ Haar 滤波器，其中 LL 为低频分量，LH、HL、HH 为高频分量。本文只保留高频分量：

$$
F_{wav}=\{|G*W_{LH}|, |G*W_{HL}|, |G*W_{HH}|\}
$$

其中 $W_{LH},W_{HL},W_{HH}$ 分别对应水平、垂直和对角方向高频响应。该部分对应 `HaarWaveletHighPass`，输出 3 通道高频图。

#### （6）频域特征拼接与标准化

四类频域特征的空间尺寸并不完全一致，因此模型首先将 DCT、FFT、Wavelet 输出插值到原图大小，然后与 SRM 残差图拼接：

$$
F=Concat(F_{srm}, F_{dct}, F_{fft}, F_{wav})
$$

由于通道数分别为 3、1、1、3，最终得到 8 通道频域图：

$$
F\in\mathbb{R}^{B\times8\times H\times W}
$$

为了避免不同频域算子的数值尺度差异影响训练，模型对拼接后的频域图做逐样本标准化：

$$
\hat{F}=\frac{F-\mu(F)}{\sigma(F)+\epsilon}
$$

### 3.2.4 频域编码器

频域编码器用于将 8 通道频域图编码为与 METER 图像 patch tokens 对齐的 token 表示。其结构如图 3-2 所示。

```text
图 3-2 频域编码器结构

8-channel Frequency Maps
        │
        ▼
Conv Stem: Conv 3×3 + BatchNorm + GELU
        │
        ▼
ConvNeXt Block × 2
        │
        ▼
Down Block: Conv 3×3 + BatchNorm + GELU + ConvNeXt Block
        │
        ▼
Adaptive Average Pooling to 16×16
        │
        ├────────▶ Token Projection Head ─────▶ Frequency Tokens Z_f
        │
        ├────────▶ Score Head ───────────────▶ Frequency Patch Scores S_f
        │
        └────────▶ Consistency Projection ───▶ Frequency Consistency Features C_f
```

频域编码器由 `MultiSourceFrequencyEncoder` 实现。设标准化后的频域图为 $\hat{F}$，编码过程可以表示为：

$$
U=E_f(\hat{F})
$$

其中 $E_f$ 由卷积层、BatchNorm、GELU 和 ConvNeXtBlock 组成。随后通过自适应平均池化将空间尺寸统一为 $16\times16$：

$$
U_{16}=Pool(U)\in\mathbb{R}^{B\times C\times16\times16}
$$

频域编码器有三个输出头：

1. token projection head：将频域特征投影到 $D=768$ 维，得到 frequency tokens：

$$
Z_f=Proj_{tok}(U_{16})\in\mathbb{R}^{B\times256\times768}
$$

2. score head：输出每个 patch 的频域异常概率：

$$
S_f=\sigma(Proj_{score}(U_{16}))\in\mathbb{R}^{B\times256}
$$

3. consistency projection head：输出频域一致性特征：

$$
C_f=Proj_{con}(U_{16})\in\mathbb{R}^{B\times256\times64}
$$

其中 $Z_f$ 用于与 RGB token 融合，$S_f$ 和 $C_f$ 用于计算频域辅助损失。

### 3.2.5 频域引导融合模块

频域引导融合模块由 `FrequencyGuidedFusion` 实现，位于 METER 图像 patch tokens 和图像 intra-modal consistency 模块之间。其作用是将频域 tokens 以可控方式注入 RGB image tokens。

```text
图 3-3 频域引导融合模块

RGB Image Patch Tokens X_v ───────┐
                                  │ Query
                                  ▼
                           Cross Attention
                                  ▲
                                  │ Key / Value
Frequency Tokens Z_f ─────────────┘

RGB Tokens X_v + Frequency Tokens Z_f ─▶ Gate Network ─▶ gate

Attention Output + gate + learnable alpha
        │
        ▼
Fused Tokens: X_hat_v = X_v + tanh(alpha) × gate × AttentionOutput
```

具体地，首先对 RGB tokens 和 frequency tokens 分别做 LayerNorm：

$$
Q=LN_v(X_v),\quad K=LN_f(Z_f),\quad V=LN_f(Z_f)
$$

然后使用多头注意力计算频域上下文：

$$
A=MultiHeadAttention(Q,K,V)
$$

为了避免频域噪声直接破坏原有语义 token，模型引入门控网络：

$$
g=\sigma(MLP([X_v;Z_f]))
$$

其中 $[X_v;Z_f]$ 表示在特征维度拼接 RGB token 和 frequency token。最后采用残差形式融合：

$$
\hat{X}_v=X_v+\tanh(\alpha)\cdot g\cdot A
$$

其中 $\alpha$ 是可学习标量，初始化为 0。这样做的好处是，训练初期 $\tanh(\alpha)\approx0$，频域分支几乎不影响原模型；随着训练进行，模型可以逐渐学习是否以及如何使用频域信息。

### 3.2.6 图像一致性建模

频域融合后得到的图像 tokens $\hat{X}_v$ 会进入原 CSCL 图像侧一致性建模模块。该部分主要包含图像 intra-modal consistency 和 image-text extra-modal consistency 两级建模。

#### （1）图像 intra-modal consistency

图像 intra-modal consistency 由 `Intra_Modal_Modeling` 实现。该模块首先通过 `Self_Interaction` 对图像 tokens 做自交互编码，然后使用 `consist_encoder` 将每个 patch token 映射到 64 维一致性空间：

$$
H_v=SelfInteraction(\hat{X}_v)
$$

$$
R_v=MLP_{con}(H_v)\in\mathbb{R}^{B\times256\times64}
$$

随后计算 patch-patch 余弦相似度矩阵：

$$
\hat{M}_v(i,j)=\frac{1}{2}\left(\frac{R_v^i\cdot R_v^j}{\|R_v^i\|\|R_v^j\|}+1\right)
$$

该矩阵反映任意两个图像 patch 在一致性空间中的相似程度。训练时，它与由真实 bbox 生成的图像一致性矩阵 $M_v$ 计算 BCE 损失，形成 `Loss_img_matrix`。

随后模块根据相似度矩阵分别选择 top-k 相似 patch 和 top-k 不相似 patch，并通过两个 MultiheadAttention 聚合相关与不相关上下文，以增强每个 patch 的一致性表达。

#### （2）多尺度 patch 标签生成

当前实现中的 `get_sscore_label` 使用多尺度方式生成图像 patch 标签，默认尺度为 $16\times16$ 和 $8\times8$。对于每个尺度，模型根据真实伪造框计算 patch 是否落入伪造区域，然后将较低尺度的标签扩展到 $16\times16$。最终一致性矩阵和 patch 标签由多尺度结果平均得到：

$$
M_v=\frac{1}{K}\sum_{k=1}^{K}M_v^{(k)},\quad
Y_p=\frac{1}{K}\sum_{k=1}^{K}Y_p^{(k)}
$$

这样可以同时利用细粒度 $16\times16$ 信息和更粗粒度 $8\times8$ 区域结构，使监督信号对局部边界扰动更平滑。

#### （3）图像 extra-modal consistency

图像 extra-modal consistency 由 `Extra_Modal_Modeling` 实现。该模块将图像 tokens、图像 CLS 特征和文本 token features 结合起来，用文本上下文辅助图像区域定位。

首先，文本 token features 被编码为跨模态上下文，模型使用一个可学习 CLS token 作为 query，对文本 tokens 做 attention 聚合：

$$
G_t=Attn(q_{cross}, X_t, X_t)
$$

然后计算每个图像 patch 与文本聚合特征之间的相似度：

$$
s_v(i)=\frac{1}{2}\left(\frac{r_v^i\cdot r_t}{\|r_v^i\|\|r_t\|}+1\right)
$$

该分数用于判断每个图像 patch 与文本语义是否一致，并与 patch label 计算 `Loss_img_score`。

同时，模块还使用相似 patch 和不相似 patch 对全局图像聚合特征进行两次 attention 更新，最终得到图像聚合特征 $A_v$。该特征用于图像 bbox 定位和图像篡改类型分类。

### 3.2.7 文本一致性建模

文本分支保持原 CSCL 结构。METER 输出的文本 token features 首先进入 `text_intra_model`，计算 token-token 一致性矩阵。文本标签由 `fake_text_pos` 生成，真实被篡改 token 标记为 1，正常 token 标记为 0，padding 位置标记为无效。

文本 intra-modal consistency 的计算与图像侧类似：先将文本 token 映射到一致性空间，再计算 token-token 相似度矩阵：

$$
\hat{M}_t(i,j)=\frac{1}{2}\left(\frac{R_t^i\cdot R_t^j}{\|R_t^i\|\|R_t^j\|}+1\right)
$$

该矩阵与文本一致性标签 $M_t$ 计算 `Loss_text_matrix`。

随后文本 tokens 进入 `text_extra_model`，与图像 tokens 交互，得到文本聚合特征 $A_t$ 和文本 token 相似分数 $s_t$。其中 $A_t$ 用于文本篡改类型分类，$s_t$ 用于生成 token 级定位结果，并与 token 标签计算 `Loss_text_score`。

### 3.2.8 任务头与输出

模型包含四类任务头。

#### （1）真伪二分类头

METER 输出的图文融合 CLS 特征经过 `fusion_head` 和 `itm_head` 得到二分类 logits：

$$
\hat{y}_{bin}=ITMHead(FusionHead(X_{cls}))
$$

该分支用于判断图文是否为真实匹配样本，对应损失为 `loss_BIC`。

#### （2）图像 bbox 定位头

图像 extra-modal consistency 输出的图像聚合特征 $A_v$ 经过 `bbox_head` 得到归一化坐标：

$$
\hat{b}=\sigma(MLP_{bbox}(A_v))
$$

其中 $\hat{b}=[c_x,c_y,w,h]$。训练时使用 L1 损失和 GIoU 损失共同约束：

$$
L_{bbox}=\|\hat{b}-b\|_1
$$

$$
L_{giou}=1-GIoU(\hat{b},b)
$$

对于纯文本篡改或真实样本，代码会通过 `is_image` 掩码避免这些样本参与 bbox 损失计算。

#### （3）多标签篡改类型分类头

图像聚合特征 $A_v$ 输入 `cls_head_img`，预测图像侧两类标签：face_swap 和 face_attribute。文本聚合特征 $A_t$ 输入 `cls_head_text`，预测文本侧两类标签：text_swap 和 text_attribute。多标签损失为：

$$
L_{MLC}=BCE(\hat{y}_{img},y_{img})+BCE(\hat{y}_{text},y_{text})
$$

#### （4）文本 token 定位头

文本 extra-modal consistency 输出的 token 相似分数 $s_t$ 会被转换为二分类 logits。代码中以 0.5 作为固定阈值：

$$
logit_{real}=s_t-0.5
$$

$$
logit_{fake}=0.5-s_t
$$

最终通过 argmax 判断每个 token 是否为篡改 token。

### 3.2.9 频域辅助损失

频域流不仅通过 tokens 融合影响主干，还引入了专门的频域辅助损失 `Loss_freq`。该损失由两部分组成：

$$
L_{freq}=L_{freq}^{patch}+L_{freq}^{matrix}
$$

#### （1）频域 patch 分类损失

频域编码器输出的 $S_f$ 表示每个 patch 的频域异常概率。它与 patch 标签 $Y_p$ 计算 BCE 损失：

$$
L_{bce}(S_f,Y_p)=-Y_p\log S_f-(1-Y_p)\log(1-S_f)
$$

由于图像中篡改区域通常只占一部分，正负 patch 数量不平衡，代码中使用动态类别权重：

$$
r=\frac{1}{BN}\sum Y_p
$$

$$
w^+=1-r,\quad w^-=r
$$

最终：

$$
L_{freq}^{patch}=\frac{1}{BN}\sum_i w_iL_{bce}(S_f^i,Y_p^i)
$$

#### （2）频域一致性矩阵损失

频域一致性特征 $C_f$ 先进行 L2 归一化，然后计算 patch-patch 余弦相似矩阵：

$$
\hat{M}_f(i,j)=\frac{1}{2}\left(\frac{C_f^i\cdot C_f^j}{\|C_f^i\|\|C_f^j\|}+1\right)
$$

该矩阵与图像一致性矩阵标签 $M_v$ 计算加权 BCE：

$$
L_{freq}^{matrix}=WBCE(\hat{M}_f,M_v)
$$

因此，频域损失同时约束两个目标：一是判断单个 patch 是否属于篡改区域，二是学习 patch 之间的频域一致性结构。

### 3.2.10 总体训练目标

模型总损失为多个任务损失的加权和：

$$
L=\lambda_{BIC}L_{BIC}+\lambda_{bbox}L_{bbox}+\lambda_{giou}L_{giou}+\lambda_{MLC}L_{MLC}+\lambda_{sim}L_{sim}+\lambda_{freq}L_{freq}
$$

其中：

$$
L_{sim}=L_{img\_score}+L_{img\_matrix}+L_{text\_score}+L_{text\_matrix}
$$

最后一次训练配置中使用的权重为：

| 损失项 | 权重 |
|---|---:|
| `loss_BIC_wgt` | 0.2 |
| `loss_bbox_wgt` | 1 |
| `loss_giou_wgt` | 1 |
| `loss_MLC_wgt` | 0.2 |
| `Loss_sim_wgt` | 5 |
| `Loss_freq_wgt` | 1 |

训练时，METER 主干被冻结，主要训练频域融合模块、图像 intra-modal consistency 模块、图像 extra-modal consistency 模块和 bbox head。这样可以降低训练开销，并避免大规模更新 METER 导致原有图文语义能力被破坏。

### 3.2.11 推理流程

推理阶段不计算损失，只保留前向传播路径。流程如下：

```text
图 3-4 推理流程

Image + Text
    │
    ▼
METER Multimodal Encoder
    │
    ├── CLS Feature ─▶ Binary Classifier ─▶ Real/Fake
    │
    ├── Text Tokens ─▶ Text Consistency Modules ─▶ Text Type + Token Localization
    │
    └── Image Tokens ─┐
                      ▼
Image ─▶ Frequency Encoder ─▶ Frequency Guided Fusion
                      │
                      ▼
          Image Consistency Modules
                      │
                      ├── BBox Head ─▶ Forged Region Box
                      └── Image Type Head ─▶ Image Tampering Type
```

输出包括：

1. `logits_real_fake`：真伪二分类结果；
2. `logits_multicls`：四类篡改类型多标签预测；
3. `output_coord`：图像伪造区域框；
4. `logits_tok`：文本 token 级篡改定位结果。

### 3.2.12 算法伪代码

```text
算法 3-1 频域引导 CSCL 前向传播与训练

输入：图像 I，文本 T，图像框标签 b，文本篡改位置 p_text，类别标签 y
输出：总损失 L

1:  使用 tokenizer 处理文本 T，得到 text_ids 和 text_masks
2:  将 I 和 T 输入 METER，得到 X_v, X_t, X_cls
3:  将 I 转换为灰度图 G
4:  使用 SRM 提取残差特征 F_srm
5:  使用 DCT 提取块级高频能量 F_dct
6:  使用 FFT 提取全局高通响应 F_fft
7:  使用 Haar 小波提取方向高频特征 F_wav
8:  将四类频域特征 resize、concat、normalize，得到 F
9:  将 F 输入频域编码器，得到 Z_f, S_f, C_f
10: 使用 Z_f 和 X_v 进行跨注意力门控残差融合，得到 X_hat_v
11: 根据 b 生成图像 patch 标签 Y_p 和一致性矩阵 M_v
12: 根据 p_text 生成文本 token 标签和文本一致性矩阵 M_t
13: 将 X_hat_v 输入图像 intra-modal consistency，得到图像矩阵预测
14: 将 X_t 输入文本 intra-modal consistency，得到文本矩阵预测
15: 进行图像 extra-modal consistency，得到图像聚合特征 A_v 和图像 patch 分数
16: 进行文本 extra-modal consistency，得到文本聚合特征 A_t 和文本 token 分数
17: 通过 bbox head 预测图像框 b_hat
18: 通过 image/text multi-label heads 预测篡改类型
19: 通过 binary classifier 预测真伪标签
20: 计算 L_BIC, L_bbox, L_giou, L_MLC, L_sim, L_freq
21: 按配置权重加权求和得到 L
22: 反向传播并更新可训练参数
```

### 3.2.13 参数量与实现特点

频域流新增参数量约为 4.82M，具体如下：

| 模块 | 参数量 |
|---|---:|
| `FrequencyGuidedFusion` 总计 | 4,818,755 |
| `MultiSourceFrequencyEncoder` | 1,270,593 |
| `Cross Attention` | 2,362,368 |
| `Gate` | 1,182,721 |
| `rgb_norm` | 1,536 |
| `freq_norm` | 1,536 |
| `alpha` | 1 |

频域编码器内部参数如下：

| 模块 | 参数量 |
|---|---:|
| `SRM` | 0 |
| `DCT` | 0 |
| `FFT` | 0 |
| `Wavelet` | 0 |
| `stem` | 286,336 |
| `down` | 286,208 |
| `token_proj` | 689,664 |
| `score_head` | 129 |
| `consist_proj` | 8,256 |

可以看到，SRM、DCT、FFT 和 Haar Wavelet 都是固定算子，不增加可训练参数。频域分支的参数主要来自 CNN 编码器、cross attention 和 gate 网络。该设计使模型能够在较小参数增量下引入频域先验。

### 3.2.14 本节小结

本节介绍了频域引导 CSCL 模型的系统结构和各算法模块实现。该方法在原 CSCL 图文一致性学习框架的基础上，引入多源频域特征提取和频域引导融合模块。具体而言，模型从 SRM、DCT、FFT 和 Haar Wavelet 四种角度提取图像高频异常，再通过轻量 CNN 编码为 frequency tokens，并利用 cross attention、gate 和 residual fusion 将频域信息注入 METER 图像 tokens。随后，增强后的图像 tokens 进入原有图像一致性建模模块，与文本特征共同完成真伪检测、篡改类型识别、图像区域定位和文本 token 定位。

从实现角度看，本方法保留了 METER 主干和 CSCL 原有任务结构，只在图像 token 流中增加频域增强模块，并通过 `Loss_freq` 对频域 patch 异常和频域一致性矩阵进行辅助监督。该设计具有结构清晰、可插拔、参数增量较小的特点，适合用于研究频域异常信息对多模态伪造新闻检测与定位任务的影响。
