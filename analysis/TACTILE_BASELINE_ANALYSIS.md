# Bench2Dex 触觉 baseline 代码分析 — 用于设计我们自己的算法

分析对象（`~/bench2dex/Bench2Dex/policy/` 下三个触觉 fork，**均无发布权重**）：

| fork | 文件 | 架构 | 触觉融合位置 |
|---|---|---|---|
| `ACT-Tactile` | `act/detr/models/detr_vae.py:79-88` | 独立触觉 CNN backbone + **1 个池化 token** | 拼进 ACT encoder 序列 |
| `GR00T_n15_Tactile` | `src/gr00t/model/action_head/tactile_token_encoder.py` | 共享 CNN + site/spatial embedding + **1 个池化 token** | 拼进 VLM token 序列（pre-DiT） |
| `GR00T_n15_Tactile_Cross` | 同一 encoder + `flow_matching_action_head.py:343-361` | 每 site 2×2 网格 = 4 token × N site | 作为 **K/V** 与动作 token 做 cross-attention（pre/post-DiT） |

---

## 一、三个 fork 共有的**架构级缺陷**（我们最重要的差异点）

### D1 ★★★ `nn.Embedding(num_sites)` 把表征绑死在"位点数量 + 位点顺序"上

```python
# GR00T_n15_Tactile_Cross/src/gr00t/model/action_head/tactile_token_encoder.py:40
self.sensor_embedding = nn.Embedding(self.num_sites, output_dim)
# ...:47  硬校验，位数不同直接抛异常
if num_sites != self.num_sites:
    raise ValueError(f"tactile has {num_sites} sites, expected {self.num_sites}")

# ACT-Tactile/act/detr/models/detr_vae.py:84  同样的缺陷
self.tactile_site_embedding = nn.Embedding(self.tactile_cfg["num_sites"], hidden_dim)
```

而 `num_sites` 来自**训练数据集的 schema**（`scripts/gr00t_finetune.py:440`：`len(train_dataset.tactile_site_names)`）。

**实测后果**（我读了全部 12 具身的 `robot/tactile/meta/site_names`）：

| 具身 | 位数 | 命名 | 问题 |
|---|---|---|---|
| Sharpa | 10 | `right_thumb_elastomer, right_index_elastomer, …` | 顺序 **thumb 在前** |
| **LEAP** | **8** | `right_index_2, right_middle_2, right_ring_2, right_thumb_2` | **index 在前**、无 pinky |
| **Allegro** | **8** | `right_thumb_3, right_index_3, right_middle_3, right_ring_3` | **无 pinky**（4 指手） |
| Ability | 10 | `right_index_q2, right_middle_q2, …` | index 在前 |
| Shadow | 10 | `right_FFJ1, right_MFJ1, right_RFJ1, right_LFJ1` | **缩写里没有手指名** |
| Wuji / DexHand021 | 10 | `right_thumb_J4, …` | 后缀不同 |
| Orca | 10 | `right_thumb_dip, right_index_pip, …` | 后缀是关节名 |
| Revo2 | 10 | `right_thumb_touch, …` | — |
| Schunk | 10 | `right_Thumb_Flexion, right_Index_Finger_Distal, …` | 大小写/长度不同 |
| RH56DFX/RH5DG2 | 10 | `right_thumb_pad, …` | — |

**三重障碍**：
1. **位数不同（8 vs 10）→ 直接抛异常**，模型在跨手型上不是"效果差"，是**跑不起来**
2. **顺序不同 → 同一个 index 在不同手上是不同手指**（LEAP index 0 = 食指，Sharpa index 0 = 拇指）→ 即使位数相同，embedding 语义也是错的
3. **后缀是传感器专有的**（`elastomer`/`J4`/`dip`/`pip`/`touch`/`pad`/`FFJ1`）→ 名字本身不可直接复用

**唯一共享的语义**：**手侧（left/right）× 手指角色（thumb/index/middle/ring/pinky）** —— 这个 key 空间在 12 只手上是统一的，只是每只手覆盖其中 8 或 10 个槽位。

> **→ 我们的设计**：把 `Embedding(num_sites)` 换成**固定 10 槽的语义描述子**（side × finger role）+ **存在掩码** + **集合式（置换不变）聚合**。这样 8 位点和 10 位点的手共用同一张表，缺槽用 mask 处理。**这是让跨手型从"不可能"变成"可训练"的最小改动。**

### D2 ★★★ 触觉被压成极小的瓶颈

```python
# GR00T_n15_Tactile_Cross …/tactile_token_encoder.py:36
nn.AdaptiveAvgPool2d((grid_size, grid_size)),   # 默认 grid_size=2
```
- **`AdaptiveAvgPool2d((2,2))`**：240×240 的触觉图 → 2×2。而 TacMap 的接触是**稀疏局部**的（几个 contact 单元），**平均池化会把接触抹掉**。他们自己的评审只写了"可能弱化稀疏接触"，没点出"平均池化对稀疏信号是错误算子"。
- `GR00T_n15_Tactile` / `ACT-Tactile` 更进一步：**整个手池化成 1 个 token**（`tactile_pool_query` 学习的 query 做 attention pool，`detr_vae.py:86-87`）→ 10 个位点 × 240×240 压成 1 个向量。

> **→ 我们的设计**：稀疏接触要用 **max-pool / 接触支撑集保留**，或**按位点保留 token**（不做跨位点池化），让注意力自己做聚合。

### D3 ★★★ 只有**当前单帧**，没有接触时间结构

三个 fork 都只读当前帧（`GR00T_n15_Tactile_Cross/TACTILE_DESIGN_REVIEW.md` 自认"只读当前触觉帧……无法估计接触变化趋势"，标为**高优先级**）。

> **→ 我们的设计**：接触相位本身是**时间量**（跃迁、停滞），必须显式建模时间。这也正是我们"停滞检测"能力的入口。

### D4 ★★ 触觉**只能加性扰动动作特征**，不能影响决策

```python
# flow_matching_action_head.py:353-361
tactile_delta, _ = self.tactile_cross_attn(query=norm(action_features),
                                          key=norm(tactile_tokens), value=tactile_tokens)
tactile_delta = tactile_delta * tactile_keep_mask
return action_features + tactile_delta          # 只是残差相加
```

`out_proj` 零初始化 → 初始为恒等（作者刻意为之，让训练从纯视觉行为起步）。

**结构性后果**：触觉是 **K/V**，动作是 **Q** → 触觉**只能细化动作生成**，**无法**
- 改变视觉/语言表示（任务规划）
- 改变生成的是**哪一种**动作模式（只影响条件，不影响模式选择）

> **→ 我们的设计**：让触觉参与**决策/选择**（相位路由、候选动作块验证），而不只是动作残差。

### D5 ★★ 触觉分支**只有动作 flow-matching 一个梯度来源**

作者自认："只有 action flow-matching loss……**需要触觉干预测试确认网络真的依赖触觉**"，且"零初始化输出投影……首个 backward 中上游触觉 CNN 的梯度可为零"。

**这是"零初始化残差 + 无辅助监督"的固有问题**：触觉必须靠动作损失**间接发现**自己有用。

> **→ 我们的设计**：给触觉分支**直接监督**（仿真 stage 谓词）→ 触觉头不依赖动作损失就能学到有意义的东西，且**可单独验证它是否真的用了触觉**（不依赖干预实验）。

### D6 ★ `V` 未归一化而 `Q/K` 归一化

```python
query=self.tactile_query_norm(action_features),   # LayerNorm
key=self.tactile_key_norm(tactile_tokens),        # LayerNorm
value=tactile_tokens,                             # ← 未归一化
```
非对称缩放：残差幅度直接由 value 的量纲决定。这是**稳定性隐患**，不是设计选择。

---

## 二、他们**自认**的缺陷（`TACTILE_DESIGN_REVIEW.md`，2026-09-08）

| 缺陷 | 影响 | 他们标的优先级 |
|---|---|---|
| 反馈间隔 ~0.8 s 仿真时间（16 步后才重规划） | 排队动作不受新观测影响 | — |
| `pre_dit` 的 Q 是**带噪动作**，未融合视觉/语言 | 动作特征层面的残差 ≠ 物理动作修正 | — |
| 只读当前帧 | 无法估计接触趋势 | 🔴 高 |
| 240×240 → 2×2 | 弱化稀疏局部接触 | 🟡 中 |
| 信号是 CPD 滤波法向距离 + 掩码 | **"不能据此宣称测量了法向力、剪切力或摩擦"** | 🔴 高 |
| 只有 flow-matching 梯度 | 需干预测试证明真的用了触觉 | 🔴 高 |
| 固定 0.3 整模态 dropout | 最优值未知 | 🟡 中 |

**并明确写着：尚未训练新模型或运行 Isaac Sim 闭环实验。**
也承认**找不到可核实的 "TouchRefine" 原始论文**（设计依据不牢）。

**他们的下一步计划**：短段重规划 → 因果触觉历史 `[t-3..t]` + 小 temporal encoder → 更多接触阶段/错位恢复演示 → 接触状态预测辅助损失 → 慢-快结构。
**注意：没有一条用 stage 谓词监督、没有相位路由、没有跨手型规范化、没有不变性分解。**

---

## 三、他们**没有**占的设计空间（我们的机会）

| # | 空白 | 证据 |
|---|---|---|
| **S1** | **跨手型规范化**：语义槽（side × finger role）+ 存在掩码 + 置换不变聚合 | D1：现有实现位数不同直接抛异常；调研 E2 ~85% 空 |
| **S2** | **用仿真 stage 谓词直接监督触觉** | D5 + 调研 E1（触觉 × 阶段 = 空交集）。他们靠动作损失间接学；他们的计划里也没有 |
| **S3** | **触觉相位驱动决策/路由**，而非动作残差 | D4 + 调研 E3（触觉相位路由 = 空） |
| **S4** | **时间上的"停滞检测"**（该跃迁却没跃迁）与恢复 | D3 + 我们的失败签名（GR00T/73：LSCR 0.71、零个零进展、却仅 8/50 成功） |
| **S5** | **稀疏接触的正确算子**（max / 支撑集保留） | D2 |
| **S6** | **可验证触觉真的被使用**的方法学 | D5：他们要靠干预实验；我们有直接监督的探针 |

---

## 四、由此导出的我们的算法设计（与他们的差异）

| 维度 | 他们 | 我们 |
|---|---|---|
| 位点编码 | `Embedding(num_sites)`，位置索引 | **语义槽（side × finger role）+ 存在掩码 + 集合聚合** |
| 空间算子 | 平均池化到 2×2 / 全局池化成 1 token | **稀疏保持**（max/支撑集） |
| 时间 | 单帧（他们要做 `[t-3..t]`） | **显式相位/进度时间结构** |
| 监督 | 只有动作 flow-matching | **+ 仿真 stage 谓词直接监督相位头** |
| 触觉作用点 | 动作特征残差（K/V） | **决策/路由 + 停滞检测与恢复** |
| 跨手型 | 架构上不可能（抛异常） | **同一模型，held-out 手型 zero-shot** |
| 证明触觉被使用 | 需干预实验 | **相位头可单独评测** |

**一句话**：他们做的是"**把触觉塞进动作分支**"（fusion）；我们做的是"**让触觉说出任务走到哪一步，并据此做决策**"（state estimation + decision），而且**表征按手指语义而非位点序号组织**，所以能跨手。

**与已发表工作的差异**（调研）：BRIDGE/TRACT/StageACT 是**视觉**门控；CAAT/Kamijo 门控**模态**；TacWAM/TouchWorld 预测**触觉本身**而非阶段；ME-Dex 跨**数据集**但不做 held-out 手；**"阶段监督 × 触觉 × 跨手型"这个组合在两个独立调研里都是空的**。
