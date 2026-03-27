# 奖励是如何计算的

## 🔄 反馈获取流程

### 1. **奖励计算入口** (`kernel_reward.py:compute_kernel_reward_batch`)
```python
def compute_kernel_reward_batch(solution_strs, ground_truths, entry_points, **kwargs):
    # 提取生成的 CUDA 代码
    kernel_code = extract_kernel_code(solution_str)
    
    # 调用 KernelRewardClient 批量计算奖励
    results = client.compute_batch_rewards(tasks)
```

### 2. **与 KernelServer 通信** (`reward_client.py:KernelRewardClient`)

#### 提交任务
```python
# POST /evaluate
resp = client.post(f"{server_url}/evaluate", json=task_data)
# task_data 包含:
#   - reference_code: 参考实现
#   - kernel_code: 生成的 CUDA 代码
#   - entry_point: 入口点
#   - num_perf_trials: 性能测试次数
#   - num_correct_trials: 正确性测试次数
#   - enable_profiling: 是否启用性能分析
```

#### 轮询结果
```python
# GET /status/{task_id}
while time.time() - start_ts < client_timeout:
    s = client.get(f"{server_url}/status/{task_id}")
    status = data.get("status")  # completed/failed/timeout
    
    if status == "completed":
        # GET /results/{task_id}
        r = client.get(f"{server_url}/results/{task_id}")
        result = r.json()
```

### 3. **反馈内容** (从 KernelServer 返回)

```json
{
  "status": "completed",
  "compiled": true,
  "correctness": true,
  "decoy_kernel": false,
  "speedup": 2.5,
  "reference_runtime": 0.045,
  "kernel_runtime": 0.018,
  "metadata": {
    "device": "0",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "num_correct_trials": "5/5",
    "profiling": {
      "top_10_kernels": [...],
      "total_cuda_time_us": 53.4,
      "kernel_count": 10
    }
  },
  "error_message": null
}
```

### 4. **奖励计算** (`reward_client.py`)

#### 方式1: `calculate_reward_like_kernel`
```python
if not compiled:
    reward = -0.5  # 编译失败惩罚
elif not correctness:
    reward = -0.3  # 正确性失败惩罚
else:
    # 基于加速比的阶梯奖励
    if speedup >= 3.0: reward = 1.0
    elif speedup >= 2.0: reward = 0.8
    elif speedup >= 1.5: reward = 0.6
    elif speedup >= 1.2: reward = 0.4
    elif speedup >= 1.0: reward = 0.2
    else: reward = -0.1  # 性能退化惩罚
```

#### 方式2: `calculate_reward_weighted`
```python
# 加权奖励 = 正确性权重 * 正确 + 性能权重 * 加速比
reward = init_correct_weight * correctness + init_performance_weight * is_speedup_positive

# 可选: 添加 coverage 奖励
if correctness and coverage_reward.enable:
    reward += coverage_reward.weight * coverage
    # coverage = custom_kernel_time / total_kernel_time
```

### 5. **多轮训练中的反馈传递** (`vllm_async_engine.py:~1750`)

```python
# 1. 调用 reward_fn 计算奖励
env_result = reward_fn(
    model_response,  # 生成的 CUDA 代码
    ground_truth,     # 参考实现
    entry_point,      # 入口点
    uuid,
    return_full_state=True,
)

# 2. 提取环境状态（包含详细反馈）
env_state = env_result["env_state"]
tool_response_json = json.dumps(env_state, indent=2)

# 3. 使用模板格式化反馈
tool_response = current_prompt_template.format(feedback=tool_response_json)
# 模板内容:
"""
Now you have received the server feedback for your last implementation.
Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}
"""

# 4. 将反馈添加到对话历史
messages.append({"role": "user", "content": tool_response})
```

### 6. **反馈信息包含**

#### 基础信息
- ✅ **编译状态**: 是否成功编译
- ✅ **正确性**: 是否通过正确性测试 (5/5)
- ✅ **加速比**: kernel_runtime / reference_runtime
- ⚠️ **Decoy Kernel**: 是否检测到欺骗性内核

#### 性能分析
- 📊 **Top 10 Kernels**: 最耗时的10个内核
- ⏱️ **CUDA Time**: 总 CUDA 执行时间
- 📈 **Kernel Count**: 内核调用次数
- 💾 **Memory Stats**: 内存分配统计

#### 错误信息
- ❌ **编译错误**: nvcc 编译失败信息
- ❌ **运行时错误**: CUDA 执行错误
- ❌ **超时信息**: 任务超时详情

## 🎯 关键特性

1. **异步批量处理**: 使用 Ray 并发处理多个任务
2. **速率限制**: Token bucket 限制请求频率
3. **超时控制**: 两级超时（服务端 + 客户端）
4. **Decoy 检测**: 自动识别并惩罚欺骗性内核
5. **Coverage 奖励**: 鼓励使用自定义内核而非 PyTorch 算子
6. **详细 Profiling**: 提供内核级别的性能分析

这个反馈机制确保模型能够：
- 知道代码是否编译成功
- 了解性能瓶颈在哪里
- 获得具体的优化建议
- 在多轮对话中持续改进