# 1. SOTA模型
## 1.1. 大模型的基础能力
由 Gemini-3-Pro 领跑（覆盖率87.5%，HQI 85.1），紧随其后的是 GPT-5.4-Pro、Claude-4.6 等顶尖闭源模型。

Synthesis-in-the-Loop Evaluation of LLMs for RTLGeneration: Quality,Reliability,and FailureModes

https://arxiv.org/pdf/2603.11287

![./assets/image-20260313170255343.png](./assets/image-20260313170255343.png)

![alt text](image-1.png)

- 现有评估的局限性： 目前对大语言模型（LLM）生成硬件描述语言（如Verilog）的评估主要停留在功能正确性（即通过仿真测试bench）。然而，硬件设计不仅要求功能正确，还必须满足可综合性（能转化为门电路）和实现质量（面积、时序效率）。
- 差距： 许多能通过仿真的代码在实际综合时会失败，或者生成的硬件效率极低（面积或延迟比专家设计差数倍）。现有的基准测试忽略了这一关键差距。
作者提出了一套新的评估流程，包含三个连续关卡：
1. 语法有效性： 使用 Icarus Verilog 解析。
2. 可综合性： 使用 Yosys 工具配合 Nangate 45nm 工艺库进行综合，确保代码能转化为网表。
3. 功能正确性： 通过测试bench仿真。
引入新指标：硬件质量指数 (HQI)
- 这是一个 0-100 分的评分系统。
- 只有通过了上述所有关卡的设计才会得分。
- 得分基于生成设计与专家参考设计（Golden Reference）在综合后面积、延迟和警告数量上的对比。100分表示与专家设计持平。
根据 Global HQI（最佳表现），32个模型清晰地分为三个梯队：
- 第一梯队 (Tier 1, HQI ≥ 71)： 共13个模型。由 Gemini-3-Pro 领跑（覆盖率87.5%，HQI 85.1），紧随其后的是 GPT-5.4-Pro、Claude-4.6 等顶尖闭源模型。
- 第二梯队 (Tier 2, HQI 53–68)： 共11个模型。包括 GPT-4o、Gemini-2.5-Pro 以及最强的开源模型（如 DeepSeek-V3.2, Qwen3.5）。开源模型与顶尖闭源模型仍有约15-20分的差距。
- 第三梯队 (Tier 3, HQI < 53)： 共8个模型。包括一些基础版模型（如 GPT-5 base）和较小的开源模型。