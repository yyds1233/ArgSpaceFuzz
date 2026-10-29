# 并发 Oracle 模块：总体设计与接口方案

> 版本：设计草案 v1.0  
> 适用项目：ArgSpaceFuzz / ArgSpaceFuzz-TF  
> 模块名称建议：`concurrent_oracle`  
> 核心原则：与 Screen 完全解耦，直接消费 `fuzz_api_one` 生成的 Harness。

## 1. 设计目标与基本原则

新增模块作为独立测试流水线，不接管 Screen 的执行筛选任务，也不依赖 Screen 的内部调度状态。现有工程继续负责 PSR、`fuzz_api_one` 的 Harness 生成与校验，以及 Screen 原有筛选流程。新模块负责 API 适用性分析、并发 Harness 改造、执行、Oracle 判定和问题复现。

设计原则：

- **与 Screen 完全解耦：** 只通过 Harness 产物及元信息衔接，不调用或修改 Screen 内部筛选器。
- **复用原始 Harness：** 优先改造 `fuzz_api_one` 已生成的 Harness，不从零重写所有测试逻辑。
- **适用性分析前置：** 先分析 API 的测试价值、候选场景和 Oracle 可判定性，再消耗代码生成与执行预算。
- **LLM 与 Oracle 分工：** LLM 负责分析和代码转换；确定性校验器与 Oracle 引擎负责验证和判定。
- **合同驱动判定：** 并发异常不自动等于 Bug；需要考虑 API 支持范围、线程安全约定、同步要求和可验证的正确性性质。
- **可扩展、可复现：** PyTorch 与 TensorFlow 共用整体流程和数据结构，各自维护适配器、API 合同和执行配置。

## 2. 总体架构

```text
ArgSpaceFuzz / ArgSpaceFuzz-TF
  ├── PSR / API 参数空间
  ├── fuzz_api_one
  ├── 原始 Harness 生成与校验
  └── screen（保持不变，独立执行筛选）
              |
              | Harness Artifact / 元信息
              v
concurrent_oracle（独立流水线）
  1. Harness 加载与分析
  2. API 并发测试适用性分析
     ├── 规则预筛选
     ├── LLM 语义分析
     └── Oracle 可判定性检查
  3. 并发测试计划生成
  4. LLM 并发 Harness 转换
  5. Harness 校验与串行基线
  6. 并发执行与观测
  7. Oracle 引擎与结果分类
  8. 重复复现、失败缩减与回归
  9. 标准化报告
```

### 2.1 模块职责

| 模块  | 职责  |
| --- | --- |
| `harness_loader` | 读取 Harness、PSR 和元信息 |
| `harness_analyzer` | 提取 API 调用、输入构造、输出检查和共享对象线索 |
| `api_suitability_analyzer` | 分析测试价值、候选场景与信息缺口 |
| `contract_registry` | 维护 API 合同、支持范围、同步约束和 Oracle 适用条件 |
| `scenario_generator` | 生成线程配置、共享模式、同步方式和测试组合 |
| `llm_transformer` | 将原始 Harness 改造成计划指定的并发 Harness |
| `harness_validator` | 检查语法、导入、串行基线、线程收尾和结果关联 |
| `runner` | 隔离执行、超时控制、异常和进程状态采集 |
| `oracle_engine` | 调度适用 Oracle 并统一判定 |
| `reducer` | 缩减输入、线程数、任务组合和触发条件 |
| `reporter` | 输出标准结果、日志和复现信息 |

## 3. API 并发测试适用性分析

### 3.1 为什么需要这个阶段

不能只让 LLM 输出“适合/不适合”两个选项。同一个 API 可能不适合共享 Tensor 的 Lost Update 测试，却适合测试内部并行安全性、异常恢复或上下文隔离。

核心问题是：

> 对当前 API 和 Harness，哪些并发场景值得测试？有哪些 Oracle 可以使用？需要哪些合同或环境信息？生成并发 Harness 是否具有可验证的测试价值？

### 3.2 分析维度

1. **并发风险特征：** 原地修改与共享可变 Tensor、全局状态、线程上下文、自动微分上下文、tracing、临时 monkey-patch、内部并行计算、非法输入校验、异常清理逻辑。
2. **Oracle 可判定性：** 是否有数学性质、可信的串并行关系、输入不变性、异常契约、内存安全观测或上下文恢复要求。
3. **API 合同与支持范围：** 文档是否说明线程安全和同步要求；当前场景是否受支持；合同未知时保留候选，不自行假设线程安全契约。
4. **Harness 可改造性：** 原始 Harness 是否可执行、能否识别 API 调用和结果处理，是否依赖复杂外部状态，以及能否以有限改动生成并发测试。

### 3.3 适用性决策

| 决策  | 含义  | 后续处理 |
| --- | --- | --- |
| `READY` | 有明确场景和可用 Oracle | 进入测试计划生成 |
| `CONDITIONAL` | 有价值，但依赖合同确认或额外配置 | 补充条件后再生成 |
| `LOW_PRIORITY` | 当前价值较低或成本较高 | 暂缓，可抽样复查 |
| `INSUFFICIENT_INFO` | Harness、文档或合同信息不足 | 补充信息后重新分析 |

不建议让 LLM 永久输出 `UNSUITABLE` 并彻底排除 API。适用性依赖具体场景；若某个场景明确不受支持，应在测试计划层标记 `UNSUPPORTED_SCENARIO`。

### 3.4 LLM 与规则引擎分工

| 工作  | 主要执行者 |
| --- | --- |
| 读取 PSR、Harness 结构和基础元信息 | 规则引擎 / 静态分析 |
| 识别潜在共享状态、上下文和清理逻辑 | LLM 分析，规则辅助 |
| 提出候选并发场景和 Oracle | LLM |
| 检查 Oracle 是否有明确可验证的判定条件 | 合同检查器 |
| 决定执行顺序和预算 | 调度器 |
| 判定是否违反测试性质 | Oracle 引擎 |
| 确认缺陷和根因 | 复现证据、源码分析及必要的人工确认 |

LLM 输出应包含理由与证据线索；能定位源码或文档时保留文件路径、符号名或文档位置。无证据的判断标注为推测。LLM 的置信度只是分析自评，不是漏洞概率。

适用性分析器本身也需要评估：从 `LOW_PRIORITY` 或 `INSUFFICIENT_INFO` 集合抽样执行，检查漏报，并统计各决策类别的执行成本、可执行率和候选违规产出。

## 4. 并发测试场景模型

建议将一次测试表示为：

\[
\mathrm{TestCase} = (X, C, S, O)
\]

| 字段  | 含义  | 示例  |
| --- | --- | --- |
| `X` | 输入与参数 | Tensor、shape、repeats、参数约束 |
| `C` | 并发执行配置 | 外部线程数、内部线程配置、屏障、任务顺序 |
| `S` | 共享状态模型 | 独立对象、共享只读对象、共享可变 Tensor、上下文 |
| `O` | Oracle 计划 | 安全拒绝、语义一致、状态保持、上下文隔离 |

### 4.1 区分外部并发与内部并行

- `N`：外部调用线程数，即多少个 Python 线程同时调用 API。
- `T`：单次 API 调用的内部计算线程配置。

| 配置  | N   | T   | 目标  |
| --- | --- | --- | --- |
| C00 | 1   | 1   | 串行基线 |
| C01 | 1   | 4   | 探索单次算子的内部并行问题 |
| C10 | 4   | 1   | 探索多个调用之间的干扰 |
| C11 | 4   | 4   | 探索内外两层并发组合 |

线程数可扩展为 2、4、8、16，但应由适用性分析和预算决定。`repeat_interleave` 一类内部并行路径需要内部线程配置；外部多线程不能替代内部并行测试。

## 5. Oracle 体系

| Oracle | 名称  | 主要判定内容 | 代表场景 |
| --- | --- | --- | --- |
| O1  | 非法输入安全拒绝 | 非法参数是否被正确拒绝；执行中是否发生内存错误 | `repeat_interleave` |
| O2  | 串并行语义一致 | 在合同允许比较的配置下，结果是否满足预期关系 | `Tensor.add_` 等候选 |
| O3  | 调用独立性 | 一个任务是否受到其他独立任务的非预期干扰 | 独立调用、并发计算 |
| O4  | 只读状态保持 | 输入、参数或声明为只读的状态是否意外改变 | 共享只读对象 |
| O5  | 异常隔离与恢复 | 一个任务的异常是否污染其他任务或后续调用 | 异常与正常任务混合 |
| O6  | 线程上下文隔离 | 有明确线程局部语义的上下文是否相互影响、能否恢复 | `no_grad` 等上下文 |
| O7  | tracing 状态与生命周期 | tracing 是否出现状态交叉、清理失败或后续任务污染 | `symbolic_trace`、`make_fx` |

### 5.1 判定边界

- **O1：** 根据明确参数约束判断非法输入；不能把任意异常都当作安全拒绝。内存错误应结合崩溃信息或 ASan 等证据确认。
- **O2：** 只有 API 合同允许比较时，才用串并行关系判错。共享原地写入的支持语义需要先核实。
- **O3/O4：** 明确哪些对象应共享、哪些应独立，以及哪些状态必须保持不变。
- **O5：** 区分预期异常、意外异常、崩溃和后续任务污染。
- **O6：** 只对合同明确要求线程隔离或状态恢复的上下文自动判定。
- **O7：** 观察 tracing 异常和清理状态；并发支持范围未确认时，不直接把异常解释为实现缺陷。

每个 API 或测试场景需登记合同状态：

| 状态  | 含义  | 判定策略 |
| --- | --- | --- |
| `SUPPORTED` | 并发用法有明确支持依据 | 可根据已确认性质判错 |
| `UNSUPPORTED` | 文档或维护者明确说明不支持 | 记录使用限制，不自动判为实现缺陷 |
| `UNKNOWN` | 尚未确认线程安全约定 | 保留候选，进一步核查 |

## 6. 从原始 Harness 到并发候选的完整流程

1. **读取与分析：** 加载原始 Harness、PSR（如有）及元信息，提取目标 API、输入构造、调用、返回值处理、断言和共享对象线索。
2. **适用性分析：** 规则预筛选 + LLM 语义分析 + 合同检查，输出候选场景、Oracle 和信息缺口。
3. **生成测试计划：** 确定并发模式、共享状态、线程配置、Oracle 组合、预算和超时。
4. **LLM 并发转换：** 以原始 Harness、API 元信息、测试计划、Oracle 规范和固定模板为输入，优先复用原测试逻辑。
5. **校验生成产物：** 检查语法、依赖、线程收尾、异常收集、结果关联和 Oracle 一致性；先执行串行基线，必要时有限修复。
6. **并发执行与观测：** 采集返回值、异常、超时、进程退出状态和必要的内存检测报告。
7. **Oracle 判定：** 区分通过、违反明确合同、证据不足和不适用。
8. **复现、缩减与回归：** 重复运行并缩减输入、线程数和任务组合，生成独立 PoC，并在修复版本上回归。

## 7. 工程目录结构

在两个仓库中分别新建同名目录 `concurrent_oracle/`。整体架构和数据格式尽量一致，PyTorch 与 TensorFlow 各自维护适配器、合同和运行配置。

```text
ArgSpaceFuzz/
├── build_yaml/                     # 现有：PSR / API 参数空间
├── fuzz_api_one/                   # 现有：Harness 生成与校验
├── screen/                         # 现有：执行筛选，保持不变
└── concurrent_oracle/              # 新增：独立并发测试模块
    ├── __init__.py
    ├── cli.py                      # 独立命令行入口
    ├── pipeline.py                 # 编排完整流程
    ├── config.py                   # 配置和预算
    ├── schemas.py                  # 统一数据模型
    ├── harness_loader.py           # 读取原始 Harness / PSR / 元信息
    ├── harness_analyzer.py         # 提取 API 调用与状态线索
    ├── api_suitability_analyzer.py # API 并发测试适用性分析
    ├── suitability_rules.py        # 规则预筛选
    ├── llm_suitability.py          # LLM 适用性分析
    ├── contract_registry.py        # API 合同与支持范围
    ├── scenario_generator.py       # 并发场景与测试计划
    ├── llm_transformer.py          # LLM 并发 Harness 转换
    ├── prompt_templates/
    │   ├── suitability.md          # 适用性分析提示词
    │   └── transform_harness.md    # 并发转换提示词
    ├── harness_validator.py        # 语法、依赖、串行基线检查
    ├── runner.py                   # 隔离执行、超时和观测
    ├── observation.py              # 标准化执行观测
    ├── oracle_engine.py            # Oracle 注册、调度与判定
    ├── oracles/
    │   ├── __init__.py
    │   ├── safety.py               # O1：非法输入安全拒绝
    │   ├── consistency.py          # O2：串并行语义一致
    │   ├── independence.py         # O3：调用独立性
    │   ├── state.py                # O4：只读状态保持
    │   ├── recovery.py             # O5：异常隔离与恢复
    │   ├── context.py              # O6：线程上下文隔离
    │   └── tracing.py              # O7：tracing 状态与生命周期
    ├── reducer.py                  # 失败用例缩减
    ├── reproducer.py               # 重复复现与复现率统计
    ├── reporter.py                 # 标准化报告
    ├── adapters/
    │   ├── __init__.py
    │   ├── base.py                 # 框架适配器抽象接口
    │   ├── pytorch.py              # PyTorch 适配
    │   └── tensorflow.py           # TensorFlow 适配
    ├── contracts/
    │   ├── pytorch/                # PyTorch API 合同
    │   └── tensorflow/             # TensorFlow API 合同
    ├── templates/
    │   └── concurrent_harness.py   # 固定并发执行模板
    ├── tests/
    │   ├── test_schemas.py
    │   ├── test_suitability.py
    │   ├── test_harness_validator.py
    │   ├── test_oracle_engine.py
    │   └── test_pipeline_smoke.py
    └── outputs/                    # 运行产物目录，建议加入 .gitignore
        ├── artifacts/
        ├── plans/
        ├── generated_harnesses/
        ├── logs/
        ├── oracle_results/
        ├── reproducers/
        └── reports/
```

目录实施原则：

- 这是目标结构，不要求第一版一次性创建所有文件。
- Screen 保持原职责；新模块不导入其内部筛选器。
- `adapters/` 处理框架差异；`oracles/` 处理测试性质，两者分离。
- `contracts/` 保存有依据的合同，并记录来源与状态。
- `outputs/` 保存运行产物，建议忽略大量日志、生成代码和二进制崩溃文件。
- 先实现数据模型、适用性分析、转换、执行和判定主链路，再逐步增加其他 Oracle。

## 8. 核心数据接口

阶段间建议使用 JSON 持久化，Python 内部用 `dataclass` 或 Pydantic 校验。以下是逻辑契约，具体字段可按仓库现有产物调整。

### 8.1 `HarnessArtifact`

```json
{
  "artifact_id": "torch_add_001",
  "framework": "pytorch",
  "api_name": "torch.Tensor.add_",
  "harness_path": "path/to/original_harness.py",
  "psr_path": "path/to/api_psr.yaml",
  "metadata_path": null
}
```

- `harness_path` 必填；PSR 和元信息可选。
- 原始 Harness 只读，转换结果写入新目录。
- 支持单文件和批量 Harness 清单。

### 8.2 `SuitabilityReport`

```json
{
  "artifact_id": "torch_add_001",
  "decision": "READY",
  "risk_categories": [
    "shared_mutable_state",
    "concurrent_result_consistency"
  ],
  "candidate_scenarios": [
    "shared_tensor_updates",
    "independent_calls"
  ],
  "candidate_oracles": ["O2", "O3"],
  "contract_status": "UNKNOWN",
  "required_evidence": ["API concurrency semantics"],
  "rationale": "The harness exercises an in-place operation. Shared-state concurrency is worth investigating, but expected behavior must be established before reporting a violation.",
  "confidence": 0.82
}
```

`decision` 使用 `READY`、`CONDITIONAL`、`LOW_PRIORITY`、`INSUFFICIENT_INFO`。`confidence` 是 LLM 分析自评，不代表漏洞概率。

### 8.3 `ConcurrentTestPlan`

```json
{
  "plan_id": "plan_001",
  "artifact_id": "torch_add_001",
  "scenario": "shared_tensor_updates",
  "execution": {
    "external_threads": [2, 4, 8],
    "internal_threads": [1],
    "trials": 50,
    "use_lock_control": true
  },
  "state_model": "shared_mutable_tensor",
  "oracles": ["O2", "O3"],
  "expected_relation": "requires_contract_validation",
  "timeout_seconds": 30
}
```

`expected_relation` 必须来自合同检查或已定义规则；无法建立可信判定条件时，可以探索性执行，但只能产生候选结果。

### 8.4 `ConcurrentHarnessArtifact`

```json
{
  "artifact_id": "torch_add_001",
  "source_harness": "path/to/original_harness.py",
  "concurrent_harness": "path/to/concurrent_harness.py",
  "plan_id": "plan_001",
  "generation_status": "GENERATED",
  "validation_status": "PENDING",
  "generation_attempt": 1,
  "oracle_bindings": ["O2", "O3"]
}
```

必须保留原始 Harness 和 `plan_id`，以追踪生成过程和复现路径。

### 8.5 `OracleResult`

```json
{
  "plan_id": "plan_001",
  "scenario": "shared_tensor_updates",
  "execution_status": "COMPLETED",
  "observations": {
    "calls": 400,
    "exceptions": 0,
    "wrong_outputs": 0,
    "timeouts": 0
  },
  "oracle_results": [
    {
      "oracle_id": "O2",
      "status": "INCONCLUSIVE",
      "reason": "Concurrent behavior has not yet been established as violating the API contract."
    }
  ],
  "reproduction_rate": null,
  "failure_artifacts": []
}
```

执行状态与 Oracle 状态必须分开。Oracle 状态统一为：

- `PASS`：满足已确认的测试性质；
- `VIOLATION`：观察到违反明确合同的行为；
- `INCONCLUSIVE`：有异常或差异，但证据不足；
- `NOT_APPLICABLE`：当前场景不适用。

## 9. Python 内部接口

| 接口  | 输入  | 输出  | 职责  |
| --- | --- | --- | --- |
| `load_harness(path)` | Harness 路径 | `HarnessArtifact` | 加载和验证原始产物 |
| `analyze_harness(artifact)` | Artifact | 分析结果 | 提取 API、输入、调用和状态线索 |
| `analyze_suitability(artifact)` | Artifact + 分析结果 | `SuitabilityReport` | 规则预筛选和 LLM 适用性分析 |
| `build_test_plan(report)` | SuitabilityReport | `ConcurrentTestPlan` | 生成候选测试计划 |
| `transform_harness(artifact, plan)` | Artifact + Plan | `ConcurrentHarnessArtifact` | LLM 并发改造 |
| `validate_harness(artifact, plan)` | 并发 Harness + Plan | 验证报告 | 语法、依赖、串行基线和结构检查 |
| `run_concurrent_test(artifact, plan)` | 已验证 Harness + Plan | 执行观测 | 执行和采集结果 |
| `evaluate_oracles(observation, plan)` | 观测 + Plan | `OracleResult` | 调用适用 Oracle 并统一判定 |
| `reduce_failure(result)` | 候选结果 | 缩减结果 | 缩减输入、线程数和任务组合 |
| `write_report(result)` | OracleResult | 报告路径 | 保存报告和证据 |

边界要求：

- `analyze_suitability()` 不执行测试。
- `transform_harness()` 不确认缺陷。
- `evaluate_oracles()` 不修改 Harness。
- `runner` 负责执行与观测，不替代合同判定。
- `reducer` 不得丢弃原始失败日志和配置。

## 10. 独立 CLI

```bash
# 分析一个已有 Harness 的并发测试适用性
python -m concurrent_oracle.cli analyze \
  --harness path/to/original_harness.py

# 根据分析结果生成测试计划和并发 Harness
python -m concurrent_oracle.cli generate \
  --artifact path/to/artifact.json

# 执行生成的并发 Harness
python -m concurrent_oracle.cli run \
  --plan path/to/test_plan.json

# 对候选问题重复复现和缩减
python -m concurrent_oracle.cli reproduce \
  --result path/to/oracle_result.json
```

这是建议的命令形式，实际实现时需根据仓库 Python 包结构确定导入路径。可以额外提供端到端 `pipeline` 命令，但每个阶段仍应能独立执行和调试。

## 11. 第一版实施顺序

### P0：最小闭环

1. 建立 `concurrent_oracle/` 和独立 CLI。
2. 定义 `HarnessArtifact`、`SuitabilityReport`、`ConcurrentTestPlan`。
3. 实现原始 Harness 加载和基础分析。
4. 实现规则预筛选与 LLM 适用性分析。
5. 实现 LLM 并发 Harness 转换与有限修复。
6. 实现代码校验、串行基线、外部多线程执行。
7. 实现统一观测与 Oracle 状态。
8. 先接入 O1、O2、O3，打通端到端链路。

### P1：扩展上下文与复现

1. 增加 O5、O6、O7。
2. 接入 `forward_ad`、`symbolic_trace` 和 `make_fx` 场景。
3. 实现重复复现、失败缩减和标准报告。
4. 抽样执行 `LOW_PRIORITY` 候选，评估适用性分析漏报。

### P2：跨框架与系统评估

1. 将数据模型和流程接入 TensorFlow。
2. 为两个框架维护独立适配器和 API 合同。
3. 评估历史缺陷检出、复现率、生成成功率、候选确认率、误报和执行成本。
4. 根据实验结果优化适用性分析、场景生成与 Oracle 组合。

## 12. 评估指标

| 指标  | 说明  |
| --- | --- |
| Harness 转换成功率 | 生成的并发 Harness 中通过校验的比例 |
| 串行基线通过率 | 生成产物能正常执行原始语义的比例 |
| 并发场景覆盖数 | 实际执行的 API × 场景 × Oracle 组合数 |
| 候选违规数量 | Oracle 标记为 `VIOLATION` 或需进一步确认的候选 |
| 候选确认率 | 经复现、合同检查和根因分析后确认的问题比例 |
| 重复复现率 | 相同配置重复执行时再次触发的比例 |
| 适用性漏报率 | 被低优先级处理但抽样后发现有价值问题的比例 |
| 平均执行成本 | 每个 API 或场景的时间与资源消耗 |
| 回归检出率 | 历史受影响版本与修复版本的预期差异是否被识别 |

所有指标应明确分母和统计范围，不能把最终进入执行集合的 Harness 数量直接当作生成成功率。

## 13. 最终设计决策

1. 并发 Oracle 是独立于 Screen 的第二条测试流水线。
2. 输入是 `fuzz_api_one` 生成的 Harness 及可选元信息。
3. 在 Harness 转换前新增 API 并发测试适用性分析。
4. 适用性分析采用规则与 LLM 协作，输出场景、Oracle、合同状态和信息缺口。
5. LLM 负责分析与代码转换；确定性校验器、合同检查器和 Oracle 引擎负责验证与判定。
6. 采用统一的测试计划、执行观测和 Oracle 结果数据结构。
7. 先完成最小端到端闭环，再扩展 Oracle、PyTorch/TensorFlow 适配和失败缩减。

**下一步建议：** 先实现目录骨架、三个前置数据模型、独立 CLI 和适用性分析接口，再选择一个已有 Harness 做端到端试验。待输入、分析、计划、转换、执行和判定闭环稳定后，再逐步接入其他 Oracle。