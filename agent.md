
---

## 总体思路

新版本的 `assistant_agent1027.py` 是按照手稿中的四个阶段（Phase 1–4）以及三个核心模块（**Planner / Memory / Executor**）重新强化的：

> “智能体具备自治规划、上下文记忆、模块化执行、自我校正和多阶段管线控制能力”
> ——手稿 Phase 2 Agent Construction 与 Phase 3 Modular Reasoning 对应

---

## 总体架构调整

| 模块                     | 新增 / 改进                                                                                       | 目的                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Planner（规划器）**       | 新增 LLM 自动规划 + 回退规则规划                                                                          | 让智能体能理解自然语言任务描述（例如 “denoise and correct this transcript”）并自动生成 JSON 格式执行计划；若模型不可用则自动切换到规则规划。 |
| **Memory（记忆模块）**       | 引入短期记忆（每次执行保存中间输出）+ 长期记忆（保存 domain 配置、知识等）                                                    | 对应手稿中 “Memory preserves domain configurations and contextual outputs”，支持上下文复用与长期知识嵌入。        |
| **Executor（执行器）**      | 新增执行验证、自动修复、再规划机制                                                                             | 对每一步执行结果进行自动检测（如行数一致性、空输出等），若异常则尝试修复或触发 replanning，实现闭环自治。                                   |
| **Tool Registry（工具层）** | 统一注册 pipeline 工具函数（denoise / correction / speaker / segmentation / preprocess / sct_pipeline） | 支持自适应模块调用，符合 “modular orchestration” 设计。                                                     |

---

## 二、Planner 改动详解

| 改动                                   | 说明                                                   | 对应手稿内容                                                                 |
| ------------------------------------ | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| ✅ 支持 GPT-5-nano 自动规划调用               | 允许通过大模型生成任务执行 JSON 计划。                               | “Planner interprets tasks and decomposes them into subtasks.”          |
| ✅ 增强解析容错（不崩溃）                        | 自动提取 JSON 数组，解析失败则降级到规则规划。                           | “Fallback to rule-based planner.”                                      |
| ✅ 支持引用符号 `{ref:input}`、`{ref:step1}` | 在自动生成的计划中，可以引用前一步输出。                                 | “Planner interprets dependencies and constructs executable blueprint.” |
| ✅ 自定义 fallback 规则                    | 自动判断任务描述关键词（denoise、correct、segment 等），生成合理的默认 plan。 | 保证无 LLM 也能运行。                                                          |

---

## 三、Memory 模块增强

| 改动                                   | 说明                              | 对应手稿内容                                                                 |
| ------------------------------------ | ------------------------------- | ---------------------------------------------------------------------- |
| ✅ 新增 `short_term` 与 `long_term` 区分   | 前者保存步骤输出，后者保存知识与 domain prompt。 | “Preserves both short-term and long-term memory for contextual reuse.” |
| ✅ 自动写入 input、domain prompt、knowledge | 启动时读取 pipeline 的配置写入长期记忆。       | “Integrates domain configurations and medical knowledge into memory.”  |
| ✅ Agent 执行中动态更新记忆                    | 每个步骤结果都会进入 short_term，供后续引用。    | “Contextual recall for adaptive reuse.”                                |

---

## 四、Executor / Agent 核心逻辑强化

| 功能                     | 新增 / 优化                                                       | 实现方式                                    |
| ---------------------- | ------------------------------------------------------------- | --------------------------------------- |
| ✅ **动态参数解析**           | 支持 `{ref:...}` 形式自动取上一步输出或输入文本                                | 用 `_resolve_arg()` 实现引用替换               |
| ✅ **执行验证（Validation）** | 执行完工具后自动检测：<br>‣ 行数是否一致（去噪/纠正/角色识别必须保持行数）<br>‣ 输出非空（分段工具）     | `_validate_step()`                      |
| ✅ **自动修复（Repair）**     | 如果验证失败：<br>‣ 调用 pipeline 的分块处理重试<br>‣ 若仍失败，触发 fallback 计划再试   | `_attempt_repair()`                     |
| ✅ **自我修正（Replanning）** | 若修复仍失败，Planner 会根据错误生成新的 fallback 计划并重新执行部分步骤                 | `_run_step()` 中自动触发                     |
| ✅ **统一输出结构**           | 每个执行结果保存到 JSON 文件，包含 step 名称、工具名、结果、是否修复等信息                   | 便于后续溯源与调试                               |
| ✅ **日志与报告输出**          | 自动保存到 `agent_execution_steps.json` 和 `agent_final_output.txt` | Phase 4 “SCT generation and validation” |

---

## 五、工具层（Tool Registry）重构

| 工具名                | 调用函数                                      | 功能 / 描述     |
| ------------------ | ----------------------------------------- | ----------- |
| `preprocess`       | `pipeline.preprocess_ruct()`              | 批量切分与匿名化    |
| `denoise`          | `pipeline._remove_noise()`                | 去噪（保持行数）    |
| `correction`       | `pipeline._correct_content()`             | 内容纠正（结合医疗知识） |
| `speaker_identify` | `pipeline._identify_speakers()`           | 说话人识别（基于语义线索） |
| `segmentation`     | `pipeline.apply_segmentation()`           | 主题/情绪/时间分段  |
| `sct_pipeline`     | 自动调用 pipeline 完整处理（包括加载知识与 domain prompt） | 用于端到端 SCT 生成 |

> 🧩 所有工具均统一注册，Executor 可以通过 JSON plan 自适应调用。

---

## 六、CLI（命令行接口）更新

| 功能                       | 改进点                            |
| ------------------------ | ------------------------------ |
| ✅ 增加参数 `--planner-model` | 指定 Planner 调用的 LLM 模型          |
| ✅ 自动配置医疗领域 domain        | 根据 `--domain` 参数自动选取领域知识       |
| ✅ 处理结果持久化                | 输出 JSON 日志与最终文本                |
| ✅ 错误保护                   | 确保即使某步失败也能记录 traceback 不中断整个程序 |

---

## 七、整体运行逻辑总结

> 新智能体运行流程如下（完全符合手稿描述）：

1. **加载配置与输入** → 读取 domain prompt 与 knowledge → 写入 Memory
2. **Planner 生成计划** → 解析任务自然语言 → 输出 JSON 步骤表
3. **Executor 顺序执行步骤** → 每步调用对应工具模块
4. **每步验证与修复** → 自动检测异常 → 失败则 replanning
5. **最终输出结果** → 汇总日志、保存结构化 SCT

---

## 八、功能实现对照手稿摘要

| 手稿功能描述                                                      | 是否已在新版实现 |
| ----------------------------------------------------------- | -------- |
| 自主任务规划与分解（Planner）                                          | ✅        |
| 上下文记忆与知识复用（Memory）                                          | ✅        |
| 自适应模块调用（Executor + Tools）                                   | ✅        |
| 自动验证与再规划机制                                                  | ✅        |
| 支持 noise removal / correction / speaker / segmentation 四大模块 | ✅        |
| 支持端到端 SCT 生成                                                | ✅        |
| 支持 domain prompt 与 knowledge 嵌入                             | ✅        |
| 支持工具扩展与模块注册                                                 | ✅        |
| 输出结构化日志与结果验证                                                | ✅        |

---

