# concurrent_oracle 分阶段验证方案

> 对应 `concurrent_oracle/guidance.md` v1.0。
> 验证对象：新模块的实现正确性 + 方法有效性（能否在可接受误报/成本下产出可确认的并发候选）。
> 输入语料：`/root/fuzz_output_all/` 的 269 个 `llm.*.py` Harness；可选 PSR；正/负控制。
> 原则：不修改 Screen；原始 Harness 只读，复制到 `concurrent_oracle/tests/fixtures/` 后再改造。

---

## 0. 验证原则

1. **两条线分开**
   - **模块正确性**：schema、加载、分析、计划、转换、校验、执行、Oracle、复现是否按接口工作。
   - **方法有效性**：适用性分析是否漏报、转换是否成功、Oracle 是否能检出正控、是否对负控误报、复现率与成本。
2. **执行状态与 Oracle 状态分开**：`COMPLETED/TIMEOUT/CRASH/ERROR` 不等于 `VIOLATION`；`VIOLATION` 必须有合同或明确测试性质支撑。
3. **正控 + 负控**：每个 Oracle 至少一个“应检出”的正控和一个“不应误报”的负控；否则无法证明 Oracle 有效。
4. **分母明确**：转换成功率的分母是“尝试转换的 Harness”，不是最终进入执行集合的数量；适用性漏报率的分母是抽样集合。
5. **固定 case**：原始 Harness 是 Atheris 字节驱动；并发测试必须固定输入/case，C00 与 C10/C01/C11 使用同一 `X`，否则结果不可比。
6. **不把 LLM 置信度当漏洞概率**；LLM 只产出分析/代码，判定由确定性校验器和 Oracle 引擎完成。

---

## 1. 测试夹具与场景矩阵

### 1.1 从 `fuzz_output_all` 复制 Harness

```bash
mkdir -p concurrent_oracle/tests/fixtures/harnesses
for api in \
  torch.repeat_interleave torch.arange torch.sparse_coo_tensor \
  torch.nn.quantized.functional.conv2d torch.tensordot \
  torch.addmm torch.cat torch.gather torch.nn.utils.clip_grad_value_ \
  torch.nn.init.normal_ torch.matmul torch.bmm \
  torch.nn.functional.conv2d torch.nn.linear \
  torch.sort torch.max torch.argsort torch.index_select torch.topk \
  torch.set_grad_enabled torch.random.fork_rng \
  torch.autograd.grad torch.autograd.backward \
  torch.jit.trace torch.jit.load torch.jit.save
do
  cp "fuzz_output_all/llm.$api.py" concurrent_oracle/tests/fixtures/harnesses/
done
```

原始文件只读；转换产物写到 `outputs/generated_harnesses/`。

### 1.2 场景 × Harness 矩阵

| 场景/Oracle | Harness | 触发点 | 验证用途 |
|---|---|---|---|
| O1 非法输入安全拒绝 | `torch.repeat_interleave` | `repeats`/`output_size` 非法 | 预期异常 vs 内存错误 |
| O1 | `torch.arange` | 已知 FPE crash（历史 run） | crash 归因，不把任意异常当拒绝 |
| O1 | `torch.sparse_coo_tensor` | 非法 indices | 非法构造/崩溃 |
| O1 | `torch.nn.quantized.functional.conv2d` | 历史 crash | 原生故障与安全拒绝区分 |
| O2 串并行语义一致 | `torch.addmm` | `out=` 共享输出（源码 181/193 行） | 共享可变 tensor |
| O2 | `torch.cat` | `out=` 共享输出（136/148 行） | 共享输出 |
| O2 | `torch.gather` | `out=` 共享输出（135/150 行） | 共享输出 |
| O2/O4 | `torch.nn.utils.clip_grad_value_` | 原地修改 parameters | 共享参数状态 |
| O2/O4 | `torch.nn.init.normal_` | 原地初始化 tensor | 原地写；随机性需固定 seed |
| O3 调用独立性 | `torch.matmul` / `torch.bmm` | 独立输入 | 独立调用互不干扰 |
| O3/C01/C11 | `torch.nn.functional.conv2d` | 内部并行 | 单次调用内部多线程 |
| O3 | `torch.nn.linear` / `torch.addmm` | 独立调用 | 多线程独立计算 |
| O4 只读状态保持 | `torch.sort` / `torch.max` / `torch.argsort` | 输入只读 | 输入是否被意外修改 |
| O4 | `torch.index_select` / `torch.topk` | 索引/输出 | 参数与输入状态 |
| O5 异常隔离 | `torch.gather` / `torch.cat` / `torch.index_select` | 非法 index/shape | 一个任务异常是否污染其他任务 |
| O5 | `torch.arange` | FPE crash | 崩溃与后续恢复 |
| O6 上下文隔离 | `torch.set_grad_enabled` | 显式 grad 上下文 | 线程局部上下文 |
| O6 | `torch.random.fork_rng` | RNG 上下文 | RNG 状态隔离/恢复 |
| O6 | `torch.autograd.grad` / `backward` | 自动微分上下文 | 梯度上下文 |
| O7 tracing 生命周期 | `torch.jit.trace` | tracing | 两个 trace 并行是否交叉 |
| O7 | `torch.jit.load` / `torch.jit.save` | 加载/保存 | 生命周期与清理 |

### 1.3 正控 / 负控（必须创建）

| 控制 | 文件 | 预期 | 用途 |
|---|---|---|---|
| O2 正控 | `tests/fixtures/seeded/lost_update.py` | 两个线程在 barrier 后用 Python 级 read-modify-write（如 `x.copy_(x + 1)`）无锁更新，最终值小于 2N → VIOLATION（合同 SUPPORTED 时） | 证明 O2 能检出丢失更新 |
| O3 正控 | `seeded/cross_talk.py` | 一个线程修改共享全局 tensor，另一个独立调用结果被污染 → VIOLATION | 证明 O3 能检出干扰 |
| O4 正控 | `seeded/readonly_mutate.py` | 被声明只读的输入被写 → VIOLATION | 证明 O4 能检出只读状态变化 |
| O5 正控 | `seeded/exception_pollution.py` | 一个线程抛异常后另一个线程结果/后续调用被污染 → VIOLATION | 证明 O5 能检出异常隔离失败 |
| O6 正控 | `seeded/context_leak.py` | 一个线程 `set_grad_enabled(False)` 泄漏到另一个线程 → VIOLATION | 证明 O6 能检出上下文泄漏 |
| O7 正控 | `seeded/trace_cross.py` | 两个 `jit.trace` 并行导致图/清理交叉 → VIOLATION | 证明 O7 能检出 tracing 生命周期问题 |
| O3 负控 | `seeded/independent_safe.py` | 独立只读 matmul → PASS | 检查误报 |
| O4 负控 | `seeded/readonly_safe.py` | `sort` 共享只读输入 → PASS | 检查误报 |
| O6 负控 | `seeded/context_safe.py` | 每线程独立 grad 模式 → PASS | 检查误报 |

正控是“证明 Oracle 能亮”的测试信号，不代表真实 PyTorch 缺陷；真实候选必须经合同检查和人工确认。

注意：正控必须设计成**确定性可触发**。O2 用 barrier + Python 级 read-modify-write，避免 `Tensor.add_` 在 C++ 层过于原子而无法制造丢失更新；O6/O7 的正控使用 synthetic shared state，因为 PyTorch 的 grad mode / RNG / tracing 本身可能是线程局部的。若正控执行后是 `PASS`，先检查控制设计（是否真的共享了状态、是否真的有交错），不要把“Oracle 没报”直接当成模块失败。

### 1.4 PSR / 固定 case

- 可选 PSR：从 `yaml_*` / `yaml_supplement` 的 `06_final_yaml` 找对应 API 的 `*MULTIRANK.yaml`；找不到时 `psr_path=null`，验证模块在无 PSR 时仍可工作。
- 固定 case：原始 Harness 的 `TestOneInput(data)` 用 `FuzzedDataProvider`；转换时提供两种模式：
  1. **minimal wrapper**：固定 `data` 字节串，多线程调用 `TestOneInput(data)`；适合 O3/O5/O6/O7 的全局/上下文测试。
  2. **refactored case**：抽取 `build_case()` / `run_case()`，让多个线程共享同一组 Tensor/参数；O2/O4 必须用这个模式。
- 记录固定 seed/case；C00 与并发配置必须使用同一 `X`。

---

## 2. V0：静态与合同验证（无 LLM、无执行）

**目标**：数据模型、加载器、分析器、校验器、Oracle 注册表、CLI 解析可用。

步骤：

1. 运行 `pytest concurrent_oracle/tests/test_schemas.py test_harness_validator.py test_oracle_engine.py`。
2. 对 `HarnessArtifact`、`SuitabilityReport`、`ConcurrentTestPlan`、`OracleResult` 做 JSON round-trip。
3. 用 10 个 fixture Harness 测 `load_harness`：banner `# target_api=`、路径、可选 PSR。
4. 用 5 个故意错误 Harness 测 validator：语法错误、缺 import、无线程 join、无异常收集、Oracle 未绑定。
5. 测 Oracle 注册表：O1–O7 的 ID、状态枚举、`NOT_APPLICABLE` 路径。

**验收**：schema 100% round-trip；loader 10/10；analyzer API 提取 ≥9/10；validator 5/5 捕获；无 LLM 调用。

**产物**：`outputs/validation/V0/pytest.xml`、`V0_report.md`。

---

## 3. V1：Harness 加载与基础分析（P0）

**目标**：从原始 Harness 提取 API、输入构造、目标调用、返回值处理、共享对象线索。

步骤：

1. 对 1.1 的 15 个 Harness 批量 `analyze`。
2. 人工标注：目标 API、调用点、输入构造方式、是否有 `out=`/原地写、是否使用全局/上下文。
3. 对比分析器输出；记录漏报/误报。
4. 检查可选 PSR 缺失时的行为。

**验收**：15 个中 ≥14 个 API/输入构造正确；0 崩溃；分析 JSON 可读。

**指标**：API 提取准确率、调用点数、共享状态线索 recall、单 Harness 分析耗时。

---

## 4. V2：适用性分析（规则 + LLM）（P0）

**目标**：决策、候选场景、候选 Oracle、合同状态、信息缺口。

准备：15 个 Harness 的人工标签，覆盖 `READY` / `CONDITIONAL` / `LOW_PRIORITY` / `INSUFFICIENT_INFO` ≥3 类。

步骤：

1. **规则基线**：只跑 `suitability_rules.py`，记录决策/候选 Oracle。
2. **LLM 分析**：每个 Harness 跑 3 次（temperature=0，固定模板），记录 decision、candidate_scenarios、candidate_oracles、contract_status、required_evidence、confidence。
3. **合同检查**：确认没有永久 `UNSUITABLE`；不支持的场景标记 `UNSUPPORTED_SCENARIO`。
4. **稳定性**：3 次 decision 与 Oracle 集合的重合率。
5. **抽样复查**：从 `LOW_PRIORITY`/`INSUFFICIENT_INFO` 抽 5 个，人工判断是否漏掉有价值场景。

**验收**：

- 决策覆盖 ≥3 类；无永久 `UNSUITABLE`。
- `READY` 的 Oracle 集合与人工标签重合 ≥80%。
- 3 次决策稳定率 ≥80%；不稳定案例有记录。
- 抽样 5 个低优先级中，漏报（本应 READY）≤1 个。
- 每条输出都有 rationale/evidence；无证据判断标为推测。

**指标**：决策分布、Oracle precision/recall（人工）、稳定性、单 Harness token/时间。

---

## 5. V3：测试计划生成（P0）

**目标**：每个 READY/CONDITIONAL 生成可执行、可判定的 `ConcurrentTestPlan`。

步骤：

1. 对 V2 的 READY/CONDITIONAL 生成计划。
2. 检查 `scenario`、`execution.external_threads/internal_threads/trials`、`state_model`、`oracles`、`expected_relation`、`timeout_seconds`。
3. 强制包含 C00 串行基线；O2/O4 必须 C00。
4. 无合同/无 `expected_relation` 的计划只能标记为探索性，不能绑定 `VIOLATION`。
5. 检查预算：每 Harness 计划数上限、线程矩阵上限。

**验收**：

- 100% 计划 schema 合法。
- 每个计划至少 1 个 Oracle；O2 不得在 `contract_status=UNKNOWN` 时直接判 `VIOLATION`。
- 不支持场景有 `UNSUPPORTED_SCENARIO`，不是 `VIOLATION`。
- 每个 READY Harness 至少 1 个 C00 + 1 个并发配置。

**产物**：`outputs/plans/*.json`、`V3_report.md`。

---

## 6. V4：LLM 转换、校验与串行基线（P0）

**目标**：把原始 Harness 改造成可并发执行、可观测、可关联结果的 Harness，并先通过 C00。

Harness 集（5 个 READY）：`addmm`、`cat`、`gather`、`repeat_interleave`、`conv2d`。

步骤：

1. `transform_harness(artifact, plan)`：注入线程配置、共享对象、屏障、异常收集、结果关联。
2. 静态校验：
   - 语法/import；
   - 无 `atheris.Setup/Fuzz` 在 worker 内；
   - 有固定 `build_case`/`run_case`（refactored）或固定 `data`（minimal）；
   - 线程 `join`、超时、异常收集；
   - Oracle binding 与 plan 一致；
   - 不依赖本地 helper。
3. C00 串行基线：执行原语义，比较输出/状态与原始 Harness 的固定 case 结果。
4. 有限修复：校验/基线失败最多 2 轮，记录 `generation_attempt`。

**验收**：

- 首次转换静态校验通过 ≥4/5；修复后 ≥5/5。
- C00 串行基线通过 ≥4/5；失败分类为转换/环境/输入。
- 无线程泄漏、无未捕获异常丢失、无本地依赖。
- 转换产物保留原始 Harness 路径与 `plan_id`。

**指标**：首次转换成功率、最终成功率、平均修复轮次、串行基线通过率、单 Harness 转换耗时/token。

---

## 7. V5：并发执行 + O1/O2/O3 最小闭环（P0）

**目标**：执行 C00/C01/C10/C11，完成观测、Oracle 判定和正/负控验证。

计划集：

- O1：`repeat_interleave` 非法 repeats；`arange` crash 对照。
- O2：`addmm`/`cat`/`gather` 的 `out=` 共享；`seeded/lost_update.py` 正控。
- O3：`matmul`/`bmm`/`conv2d` 独立调用；`seeded/independent_safe.py` 负控。

步骤：

1. 每个计划执行 C00、C10（N=2/4/8）、C01（T=4）、C11（4×4），trials 20–50。
2. 采集：调用次数、返回值、异常类型/次数、timeout、进程 rc、crash artifact、共享对象快照。
3. Oracle 引擎判定 O1/O2/O3；状态 `PASS/VIOLATION/INCONCLUSIVE/NOT_APPLICABLE`。
4. 正控必须出现 `VIOLATION`（或 `INCONCLUSIVE` + 明确证据）；负控不得出现 `VIOLATION`。
5. 合同 `UNKNOWN` 的 O2 只能 `INCONCLUSIVE`；`SUPPORTED` 才可 `VIOLATION`。
6. 记录执行成本（wall/cpu/峰值 RSS）。

**验收**：

- 观测字段 100% 完整；无未处理 crash/hang；进程清理干净。
- 正控检出 ≥1；负控误报 0。
- O2 的 `VIOLATION` 均绑定 `SUPPORTED` 合同；`UNKNOWN` 全为 `INCONCLUSIVE`。
- O1 不把任意异常当安全拒绝；crash 有 artifact/ASan 证据。

**指标**：Oracle 状态分布、候选违规数、确认数（人工/合同）、误报率、场景覆盖数、平均成本。

---

## 8. V6：复现与缩减（P1）

**目标**：候选结果可重复、可缩减、可独立运行。

步骤：

1. 对 V5 的 `VIOLATION`/`INCONCLUSIVE` 候选重复 20–50 次，记录复现率。
2. 缩减：线程数（8→4→2）、trials、输入尺寸、共享对象数量、屏障位置。
3. 生成独立 PoC；保留原始日志/配置。
4. 若有修复版本，做回归（原版本触发、修复版本不触发）。

**验收**：复现率可计算；缩减后仍触发 ≥1 次；PoC 独立运行；原始证据未丢失；回归结果明确或标记 N/A。

**指标**：复现率、缩减比例、PoC 运行时间、回归检出。

---

## 9. V7：O5/O6/O7 扩展（P1）

### O5 异常隔离与恢复

- Harness：`gather` 非法 index、`cat` 非法 shape、`index_select` 越界、`arange` crash。
- 场景：一个线程抛预期/意外异常，另一个线程执行正常任务；随后再执行一次正常调用。
- 正控：`seeded/exception_pollution.py`（污染其他任务）。
- 验收：预期异常不污染；意外异常/崩溃有隔离证据；后续调用恢复；状态分类正确。

### O6 上下文隔离

- Harness：`set_grad_enabled`、`random.fork_rng`、`autograd.grad/backward`。
- 场景：线程 A `set_grad_enabled(False)`/fork RNG，线程 B 保持 True/独立 RNG。
- 正控：`seeded/context_leak.py`；负控：`seeded/context_safe.py`。
- 验收：`torch.is_grad_enabled()`/RNG 状态按线程隔离；上下文退出后恢复；只对合同明确要求隔离的场景判 `VIOLATION`。

### O7 tracing 生命周期

- Harness：`torch.jit.trace`、`torch.jit.load/save`。
- 场景：两个线程分别 trace 不同函数/模块；检查 traced graph、异常、清理、后续 trace。
- 正控：`seeded/trace_cross.py`；负控：两个独立 trace。
- 验收：不把 tracing 异常直接当实现缺陷；有明确状态交叉/清理失败才 `VIOLATION`；否则 `INCONCLUSIVE`。

---

## 10. V8：规模化与鲁棒性（P2）

**目标**：50 个 Harness 的端到端稳定性、成本、指标。

步骤：

1. 从 269 个 Harness 分层抽 50：覆盖 6 个 Oracle、不同 API 族、READY/CONDITIONAL/LOW。
2. 跑完整链路：load → analyze → suitability → plan → transform → validate → C00 → concurrent → oracle → reproduce。
3. 重复 2–3 次（不同 seed），记录稳定性。
4. 抽样 `LOW_PRIORITY`/`INSUFFICIENT_INFO` 评估漏报。
5. 汇总指标（见第 12 节），与 V5 小样本对比。

**验收**：

- 链路完成率 ≥90%；无未处理 hang/crash/thread leak。
- 转换成功率、串行基线通过率、Oracle 正控检出、负控误报在可接受范围。
- 候选违规有抽样确认；复现率/成本报告完整。
- 所有指标分母明确。

---

## 11. V9：TensorFlow 适配（P2）

- 复用同一套 schema/CLI；新增 `adapters/tensorflow.py`、`contracts/tensorflow/`。
- 用 TF Harness 5–10 个做 smoke：load/analyze/transform/C00/O1/O3。
- 验收：JSON schema 与 PyTorch 一致；TF 适配器测试通过；无 PyTorch 假设泄漏。
- 不在 P0/P1 阶段混入 TF，避免跨框架问题掩盖主链路问题。

---

## 12. 指标定义与目标

| 指标 | 定义 | 分母 | 目标 |
|---|---|---|---|
| Harness 转换成功率 | 通过静态校验的并发 Harness | 尝试转换的 Harness | 首次 ≥80%，修复后 ≥95% |
| 串行基线通过率 | C00 执行成功且输出符合原语义 | 生成的并发 Harness | ≥90% |
| 并发场景覆盖数 | 实际执行的 API × 场景 × Oracle | — | P0 ≥30 组合；P1 ≥80 |
| 候选违规数量 | `VIOLATION` + 有证据的 `INCONCLUSIVE` | 执行计划 | 报告，不设“越多越好” |
| 候选确认率 | 经合同/人工确认的候选 | 候选违规 | 报告；P0 目标 ≥1 个确认或明确 N/A |
| 重复复现率 | 重复执行再次触发 | 重复次数 | 报告；正控应高 |
| 适用性漏报率 | 抽样低优先级中发现有价值场景 | 抽样低优先级数 | ≤20% |
| 平均执行成本 | wall/CPU/token per API/场景 | 执行数 | 报告 |
| 回归检出率 | 修复版本与受影响版本差异被识别 | 有修复版本的案例 | 报告 |

**禁止**：把“最终进入执行的 Harness 数 / 全部 Harness 数”当作生成成功率；所有比率必须写明分母。

---

## 13. 阶段门禁与停止条件

| 门禁 | 条件 | 不通过的处理 |
|---|---|---|
| G0 | V0 全部单测通过 | 停止，不进入 V1 |
| G1 | V1 ≥14/15 API/输入正确 | 修 analyzer/loader |
| G2 | V2 决策稳定 ≥80%，无永久 UNSUITABLE | 修 prompt/规则；不进入 V3 |
| G3 | V3 100% schema，O2 合同门禁生效 | 修 plan 生成 |
| G4 | V4 首次 ≥4/5，串行基线 ≥4/5 | 修 transform/validator；不进入 V5 |
| G5 | V5 正控检出、负控 0 误报、观测完整 | 修 runner/oracle；不进入 V6 |
| G6 | V6 复现率可计算、PoC 可运行 | 修 reproducer/reducer |
| G7 | V7 每个 Oracle 有正控检出 + 负控通过 | 修对应 oracle |
| G8 | V8 完成率 ≥90%，无 hang/leak | 修鲁棒性/预算 |

**立即停止条件**：未捕获的进程 hang、线程泄漏、C00 串行基线 <60%、负控出现 `VIOLATION`、Oracle 在 `UNKNOWN` 合同下直接判 `VIOLATION`、原始日志被 reducer 覆盖。

---

## 14. 产物与命令（接口验收）

按 guidance §10 的 CLI 形态，每个阶段可独立执行：

```bash
# V1/V2
python -m concurrent_oracle.cli analyze --harness tests/fixtures/harnesses/llm.torch.addmm.py

# V3/V4
python -m concurrent_oracle.cli generate --artifact outputs/artifacts/<id>.json

# V5
python -m concurrent_oracle.cli run --plan outputs/plans/<plan>.json

# V6
python -m concurrent_oracle.cli reproduce --result outputs/oracle_results/<result>.json
```

阶段产物：

```text
outputs/
├── validation/V0..V8/          # 每阶段报告
├── artifacts/                  # HarnessArtifact
├── plans/                      # ConcurrentTestPlan
├── generated_harnesses/        # 并发 Harness
├── logs/                       # 执行日志
├── oracle_results/             # OracleResult
├── reproducers/                # 缩减后的 PoC
└── reports/                    # 汇总
```

---

## 15. 风险与对策

| 风险 | 对策 |
|---|---|
| LLM 非确定性 | temperature=0；每个 Harness 3 次；schema 校验；修复上限 2 轮；记录模型/版本 |
| 线程调度不确定性 | 屏障、固定 trials、重复执行；报告复现率而非单次结论 |
| 误报 | 正/负控；合同状态；`UNKNOWN` 只出 `INCONCLUSIVE`；人工确认 |
| 转换漂移 | C00 串行基线；与原 Harness 固定 case 比较输出 |
| 线程过订阅 | CPU 线程 T=4 上限；`OMP_NUM_THREADS=1`；记录 peak RSS |
| 覆盖率构建缺 LAPACK | 避免 `torch.linalg`；用 CPU 可执行算子 |
| 原始 Harness 无限循环/大输入 | runner 独立 timeout；固定 case 尺寸上限；crash artifact |
| 原始 Harness 吞异常 | validator 要求并发 Harness 记录异常，不允许统一 `except: return` |
| 结果不可关联 | 每线程结果带 thread_id/case_id/plan_id；Oracle 结果绑定 plan_id |

---

## 16. 建议执行顺序（最短闭环）

1. V0（0.5 天，无 LLM）
2. V1（0.5–1 天，15 Harness）
3. V2（2–3 天，15×3 次分析）
4. V3（1 天）
5. V4（3–5 天，5 Harness）
6. V5（3–5 天，O1/O2/O3）
7. V6（2 天）
8. V7（3–4 天，O5/O6/O7）
9. V8（5 天，50 Harness）
10. V9（TF，后续）

先打通 V0–V5 的最小闭环，再扩展 O5/O6/O7 和规模；不要在 O1/O2/O3 尚未稳定时同时引入 TF 或全部 Oracle。
