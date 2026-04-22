# SimpleVLA-RL 核心流程详解

## 一、项目定位

将 **GRPO（Group Relative Policy Optimization）** 应用到 **VLA（Vision-Language-Action）** 机器人模型上的强化学习框架。核心思路：用少量轨迹做 SFT 冷启动，然后用 RL 大幅提升模型的长程规划能力。

论文效果：LIBERO-Long 上从 17.3 提升到 91.7（+430%）。

---

## 二、整体架构与数据流

训练循环在 `verl/trainer/ppo/ray_trainer.py:fit()`（第 469-717 行），每一轮迭代：

```
数据加载 → Rollout（与环境交互）→ 验证（算分）→ 过滤 → 奖励计算 → 优势计算 → Actor更新 → 验证评估 → 保存checkpoint
```

---

## 三、核心文件

| 文件 | 作用 |
|------|------|
| `verl/trainer/main_ppo.py` | 入口：初始化 Ray、创建 RewardManager、启动 Trainer |
| `verl/trainer/ppo/ray_trainer.py` | 训练循环：Rollout → Filter → Advantage → Actor Update |
| `verl/trainer/ppo/core_algos.py` | RL 算法：GRPO、GAE、PPO policy loss、KL 控制器 |
| `verl/workers/rollout/rob_rollout.py` | Rollout：VLA 与机器人环境交互收集轨迹 |
| `verl/workers/actor/dp_rob.py` | Actor：VLA 特化的 PPO 策略更新 |
| `verl/workers/fsdp_workers.py` | FSDP Worker：Hybrid Engine（Actor+Rollout 共享模型） |
| `verl/utils/dataset/rob_dataset.py` | 数据集：LIBERO_Dataset、Robotwin_Dataset |
| `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py` | VLA 模型：OpenVLA-OFT 的前向和推理 |

---

## 四、关键设计决策

| 设计选择 | 具体做法 | 原因 |
|---------|---------|------|
| **RL算法** | GRPO（非标准PPO） | 不需要 critic/value function，组内归一化即可 |
| **奖励** | 二值 0/1（成功/失败） | 最简奖励工程，避免手工设计 |
| **KL系数** | 0.0（无参考策略约束） | 允许模型自由偏离 SFT 策略 |
| **裁剪范围** | 非对称 [0.2, 0.28] | 鼓励更大的策略更新 |
| **温度** | 1.6（高温度） | 鼓励探索多样轨迹 |
| **精度过滤** | [0.1, 0.9] | 只保留有学习信号的组 |
| **Action表示** | 离散256 bins | VLA 原生 token 化，无需重训 action head |
| **Hybrid Engine** | Actor 和 Rollout 共享 FSDP 模型 | 节省显存，无需单独推理引擎 |

---

## 五、完整流程：轨迹 → 奖励 → GRPO → 模型更新

假设：LIBERO 任务，`batch_size=4`（4个不同任务），`n_samples=8`（每个任务采样8条轨迹），`action_token_len=7`，`action_chunks_len=8`。

### 阶段 1：Rollout 生成轨迹

入口：`ray_trainer.py` 第 540-553 行

```python
# 每个 prompt 复制 n_samples 份
batch_lst = [[newbatch[i:i+1] for _ in range(n_samples)] for i in range(len(newbatch))]
# batch_lst: 4个prompt × 8份 = 32个样本

gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts=gen_batch)
```

`generate_sequences` 内部（`fsdp_workers.py:506`）调用 `rollout.generate_sequences()`，即 `rob_rollout.py` 的实现。

Rollout 过程：
- 32 个环境并行运行（4任务 × 8采样）
- 每个环境逐步执行：观测 → VLA推理 → 执行动作 → 新观测
- 最终输出 `gen_batch_output`，其中 `responses` shape 为 `(32, T, 56)`（T是实际执行步数）

**同时**（`fsdp_workers.py:515-524`），rollout 结束后立即用**当前策略**计算 `old_log_probs`：

```python
old_log_probs = self.actor.compute_log_prob(data=output)
output.batch['old_log_probs'] = old_log_probs
```

`old_log_probs` shape: `(32, T × 56)` — 每个 token 位置的对数概率。

### 阶段 2：验证 + 精度过滤

**① 验证算分**（`ray_trainer.py:561`）：

```python
scores_tensor, reward_metrics, ... = self.reward_fn.verify(roll_batch)
```

`verify()` 在 `main_ppo.py:38-56`，直接读 `data.batch['complete']`（bool），生成 `acc` 标量：

```python
data.batch['acc'] = torch.tensor([1.0, 0.0, 1.0, 0.0, ...])  # shape (32,)
```

**② 精度过滤**（`ray_trainer.py:573-578`）：

```python
filtered_roll_batch = self.filter(roll_batch.batch['acc'].unsqueeze(1), roll_batch, n_samples)
```

`filter()` 在 `ray_trainer.py:755-816`：

```python
# 把 32 条轨迹按 n_samples=8 分组，计算每组精度
reward_matrix = reward_tensor.sum(-1).reshape(-1, 8)  # (4, 8)
acc_tensor = torch.mean(reward_matrix, dim=-1)        # (4,)  例如 [0.5, 0.125, 0.0, 1.0]

# 只保留精度在 [0.1, 0.9] 之间的组
acc_mask = (acc_tensor >= 0.1) & (acc_tensor <= 0.9)  # [True, True, False, False]
# 第3组全失败(0.0)、第4组全成功(1.0)被过滤掉
```

过滤后可能只剩 16 条轨迹（2组 × 8采样）。

**为什么过滤？** 全成功（acc=1.0）或全失败（acc=0.0）的组没有梯度信号——GRPO 的优势来自组内的**差异**，如果组内全部一样，归一化后 advantage 全是 0。

### 阶段 3：计算奖励

调用 `RobRewardManager.__call__()`（`main_ppo.py:58-99`）：

```python
# Step 1: 创建全零 tensor，和 responses 同形
verifier_reward = torch.zeros_like(data.batch['responses'])  # (16, T, 56)
verifier_reward = verifier_reward.reshape((16, -1))           # (16, T×56)

# Step 2: 计算 valid_response_length
valid_response_length = finish_step * action_token_len  # 例如 [210, 280, ...]

# Step 3: 在最后一个有效 token 位置放奖励
for i in range(16):
    verifier_reward[i, valid_response_length[i]-1] += verifier_score[i]
    # verifier_score[i] 是 1.0 或 0.0

# Step 4: 放入 reward_tensor_dict
reward_tensor_dict['gt_scores'] = verifier_reward  # (16, T×56)，几乎全0，只有末尾有0或1
reward_tensor_dict['all'] = verifier_reward * reward_coef  # reward_coef=1.0
```

此时 `batch.batch['token_level_scores']` = `(16, T×56)` 的稀疏奖励。

**为什么奖励只放在最后一个 token？** 机器人任务只有最终成功/失败的信号，没有中间步骤的对错标注。这是 outcome-based（结果型）稀疏奖励的标准做法。

### 阶段 4：GRPO 优势计算

核心代码：`core_algos.py:173-216` 的 `compute_grpo_outcome_advantage()`。

**Step 1: 每条轨迹求总奖励**

```python
scores = token_level_rewards.sum(dim=-1)  # (16,)
# 因为奖励只在末尾有一个值，sum 就是那个值
# scores = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, ...]
```

**Step 2: 按 uid 分组归一化**

```python
# uid 在 ray_trainer.py:537 生成，同一个 prompt 的 8 个采样共享同一个 uid
# index = [uid_0, uid_0, ..., uid_0, uid_1, uid_1, ..., uid_1, ...]

for i in range(bsz):
    id2score[index[i]].append(scores[i])

# 例如 uid_0 对应的 8 个 scores: [1, 0, 1, 1, 0, 0, 1, 0]
# mean = 0.5, std ≈ 0.53

for i in range(bsz):
    scores[i] = (scores[i] - mean) / (std + epsilon)
# 归一化后: [0.94, -0.94, 0.94, 0.94, -0.94, -0.94, 0.94, -0.94]
```

**Step 3: 扩展到每个 token 位置**

```python
scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
# (16,) → (16, T×56)  每个 token 位置得到相同的 advantage 值
```

**输出**：`advantages` shape `(16, T×56)`，每个 token 位置的值 = 该轨迹归一化后的 advantage。

**直觉理解**：同一 prompt 的 8 个采样中，成功的轨迹获得正 advantage，失败的获得负 advantage。模型会倾向于增加成功轨迹的概率、降低失败轨迹的概率。

### 阶段 5：Actor 策略更新

入口：`ray_trainer.py:662-665` → `dp_rob.py:update_policy()`（第 401-531 行）

**Step 1: 准备数据**

```python
select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values',
               'old_log_probs', 'advantages', 'finish_step']
batch = data.select(batch_keys=select_keys).batch
```

**Step 2: 按 traj_mini_batch_size 分段**（节省显存）

```python
# traj_len = T, 假设 traj_mini_batch_size = 8
# 把轨迹分成 T/8 段，逐段前向+反向
for i in range(0, traj_len, traj_split_num):
```

**Step 3: 前向传播，计算新的 log_prob**

```python
# 把 (batch_size, traj_len, ...) reshape 成 (batch_size × traj_len, ...)
input_ids = input_ids.reshape((batch_size * traj_len,) + ...)
responses = responses.reshape((batch_size * traj_len,) + ...)

# VLA 模型前向
logits = self.actor_module(input_ids=..., attention_mask=..., pixel_values=..., proprio=...)

# 取出 action token 的 logits（词表末尾 256 个位置）
logits = logits[..., -256-64:-64]  # (batch_size×traj_len, seq_len, 256)
responses = responses - (vocab_size - 256)  # 映射到 [0, 255]

# 计算 log_prob 和 entropy
log_prob = logprobs_from_logits(logits, responses)
```

**Step 4: 计算 PPO loss**（`core_algos.py:293-301`）

```python
ratio = exp(new_log_prob - old_log_prob)              # 重要性采样比率
pg_losses  = -advantages × ratio                      # 未裁剪的 loss
pg_losses2 = -advantages × clip(ratio, 0.8, 1.28)    # 裁剪后的 loss
pg_loss = mean(max(pg_losses, pg_losses2))             # 取较大值（保守更新）
```

**Step 5: 反向传播 + 优化器更新**

```python
loss = pg_loss / gradient_accumulation
loss.backward()

# 所有 micro_batch 累积完后
grad_norm = self._optimizer_step()  # 梯度裁剪 + optimizer.step()
```

---

## 六、完整数据流图

```
Rollout (4任务×8采样=32条轨迹)
  │
  │ responses: (32, T, 56)     ← VLA 生成的 action token IDs
  │ old_log_probs: (32, T×56)  ← rollout 后立即用当前策略计算
  │ complete: (32,)             ← 任务是否成功
  ▼
Verify + Filter
  │ acc: (32,) → 分组过滤 → 保留 16 条
  │ 全成功/全失败的组被丢弃（无学习信号）
  ▼
Compute Reward
  │ token_level_scores: (16, T×56)  ← 稀疏奖励，末尾放 0 或 1
  ▼
GRPO Advantage
  │ scores = sum → (16,)  ← 每条轨迹的总奖励
  │ 按 uid 分组 (每组8个) → 组内归一化 (减均值/除标准差)
  │ advantages: (16, T×56)  ← 每个 token 位置相同的 advantage
  ▼
Actor Update (PPO)
  │ new_log_prob = VLA 前向传播 → (16, T×56)
  │ ratio = exp(new_log_prob - old_log_prob)
  │ loss = -advantages × clip(ratio, 0.8, 1.28)
  │ loss.backward() → optimizer.step()
  ▼
  模型权重更新完成
```

---

## 七、Tensor 形状速查表

| 变量 | 形状 | 说明 |
|------|------|------|
| `responses` | `(batch_size, T, 56)` | T=轨迹步数, 56=action_token_len×action_dim (LIBERO) |
| `old_log_probs` | `(batch_size, T×56)` | 每个 token 位置的对数概率 |
| `complete` | `(batch_size,)` | 每条轨迹是否成功 |
| `acc` | `(batch_size,)` | complete 的 float 版本 |
| `token_level_scores` | `(batch_size, T×56)` | 稀疏奖励，几乎全0 |
| `advantages` | `(batch_size, T×56)` | GRPO 归一化后的优势 |
| `finish_step` | `(batch_size,)` | 每条轨迹实际执行的步数 |
| `valid_response_length` | `(batch_size,)` | finish_step × action_token_len |
| `response_mask` | `(batch_size, T×56)` | 只有 finish_step 之前的位置为 True |

---

## 八、常见问题

### Q: 为什么 GRPO 需要先 rollout 再 replay 更新？rollout 直接记录 logits 不行吗？为什么每次只更新一个?(assert self.config.ppo_micro_batch_size == 1)

没有那么大显存吧，必须搞成这种rollout and replay的方式。但是确实保留了rollout时的log_prob，然后因为更新模型参数了，所以每次更新前都要重新forward一次计算新的log_prob，来算importance sampling ratio。


<!-- #### 关键：PPO 是 off-policy 算法，需要 importance sampling ratio

PPO 的 loss 公式是：

```
ratio = exp(new_log_prob - old_log_prob)    # π_θ_new / π_θ_old
loss  = -advantages × clip(ratio, 0.8, 1.28)
```

这里 **`old_log_prob`** 是 rollout 时用旧策略 π_θ₀ 计算的，**`new_log_prob`** 是更新时用当前策略 π_θ_new 计算的。**两次 forward 的策略参数不同**，所以不能复用。

#### 时间线

```
时间 ──────────────────────────────────────────────────►

[1] Rollout 阶段
    用策略 π_θ₀ 生成轨迹，同时记录 old_log_probs
    old_log_probs = log π_θ₀(action | obs)

[2] Actor Update 阶段（可能多轮迭代）
    ┌─ 第1次 backward ─┐
    │ new_log_probs = log π_θ₁(action | obs)   ← θ₁ ≈ θ₀（刚更新）
    │ ratio = exp(new - old)
    │ loss.backward() → θ₁
    │
    ├─ 第2次 backward ─┤
    │ new_log_probs = log π_θ₂(action | obs)   ← θ₂ ≠ θ₁（又更新了）
    │ ratio = exp(new - old)
    │ loss.backward() → θ₂
    │
    └─ ... ────────────┘
```

每次 `optimizer.step()` 之后，模型参数 θ 就变了，所以 **必须重新 forward** 才能得到当前策略下的 log_prob。

#### 如果只 forward 一次会怎样？

假设 rollout 时记录了 logits，然后直接复用：

```
old_logits = logits from π_θ₀  (rollout 时记录)
```

在 update 阶段，如果只做一次 backward + optimizer.step()，那确实可以只 forward 一次。但问题是：

**PPO 通常在一个 batch 上做多轮更新**（`ppo_mini_batch_size` 次），每轮更新后 θ 都变了，下一轮的 `new_log_prob` 必须用更新后的 θ 重新计算。

#### 为什么不直接记录 logits 而只记录 log_prob？

`dp_rob.py` rollout 后记录的是 `old_log_probs`（标量），不是 logits（向量）。因为：

1. **logits 很大**：`(batch_size, seq_len, vocab_size)`，vocab_size=32000，显存吃不消
2. **log_prob 够用**：PPO loss 只需要 `ratio = exp(new_log_prob - old_log_prob)`，两个标量就够了

#### 和 REINFORCE 的区别

如果用纯 REINFORCE（没有 importance sampling ratio），确实只需要 rollout 一次：

```
# REINFORCE: 直接用 advantage × log_prob 的梯度
loss = -advantages × log_prob   # 不需要 old_log_prob
```

但 PPO 引入了 **clipped surrogate objective**，核心就是用 `ratio` 来约束更新幅度，所以必须保留旧策略的 log_prob 做对比。这也是 PPO 比 REINFORCE 更稳定的原因——它不会让策略一步走太远。

#### 总结

| 算法 | 需要几次 forward？ | 为什么？ |
|------|-------------------|---------|
| REINFORCE | 1 次（rollout） | 直接用 log_prob 的梯度，不需要对比 |
| PPO/GRPO | 2 次（rollout + update） | 需要 `ratio = π_new / π_old`，每次参数更新后 π_new 都变了 | -->
