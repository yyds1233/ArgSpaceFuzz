# concurrent_oracle

独立于 Screen 的并发测试流水线。输入是 `fuzz_api_one` 生成的 Atheris
Harness，输出是并发测试计划、生成的并发 Harness、执行观测和 Oracle 判定。

## P0 状态

- 已实现 P0：数据模型、Harness 加载/静态分析、规则适用性分析、可选 LLM
  适用性分析、测试计划生成、固定 case 捕获、模板化并发 Harness 生成、
  静态校验、串行基线 + 外部多线程执行、O1/O2/O3、重复复现和报告。
- 已实现 P1 基础版：O5 混合 case 异常隔离、O6 grad-mode 上下文探针、
  O7 `jit.trace` 生命周期观测、C01/C11 内部并行配置。
- P2 预留：TensorFlow adapter/contracts；O6 的 `fork_rng`、O7 的
  `symbolic_trace`/`make_fx`、完整 reducer 仍待扩展。

### P1 已验证

```text
O5 torch.cat（invalid + valid 混合） -> PASS（invalid 抛 RuntimeError，valid 正常，recovery 成功）
O6 torch.set_grad_enabled（True/False 两 case） -> PASS（线程局部，main 保持 True）
O6 torch.random.fork_rng -> INCONCLUSIVE（并发 fork/restore 出现 RNG 状态差异；合同 UNKNOWN）
O7 torch.jit.trace -> INCONCLUSIVE（合同 UNKNOWN，证据已记录）
O7 torch.fx.symbolic_trace -> INCONCLUSIVE（并发 trace 出现异常/回收失败；合同 UNKNOWN）
O7 make_fx -> INCONCLUSIVE（并发 trace 成功；合同 UNKNOWN）
C01/C11 torch.matmul -> baseline/C10/C01/C11 四组观测，O3 PASS
reducer seeded O2 -> trials 3→1 仍 VIOLATION（配置缩小成功）
```

慢速 P1 测试：

```bash
RUN_CONCURRENT_SMOKE=1 python3 -m unittest concurrent_oracle.tests.test_pipeline_smoke -v
# 10 tests: matmul O3, O5 cat, O6 grad, O6 fork_rng, O7 jit/symbolic/make_fx,
#           reducer, seeded O2, analyze
```

### P0 已验证

```text
单元测试：33 项通过（9 项慢速 e2e 默认跳过）
慢速 e2e：RUN_CONCURRENT_SMOKE=1 -> 10 项全部通过
  （matmul O3 + C01/C11、O5 cat、O6 grad/fork_rng、O7 jit/symbolic/make_fx、
    reducer、seeded O2 正控）

torch.matmul:
  O2 UNKNOWN       -> INCONCLUSIVE
  O3 SUPPORTED     -> PASS
  O4 SUPPORTED     -> PASS

torch.addmm:
  O2 UNKNOWN       -> INCONCLUSIVE（共享 out= 语义未确认）

torch.repeat_interleave:
  O1 SUPPORTED     -> PASS（synthetic invalid case: RuntimeError）

seeded lost_update（O2 正控，合同 SUPPORTED）:
  O2               -> VIOLATION（并发最终值 3 vs 串行基线 6）
```

### 简化与限制

- `run_plan` 只执行 `external_threads` 的最大值；C01/C11 会额外执行，
  串行基线按 `trials × N` 运行，保证共享状态比较的总操作数一致。
- O1 目前只对 `repeat_interleave`、`arange`、`cat`、`gather` 有显式
  synthetic 非法 case；其他 API 捕获的是合法 case，O1 会返回
  `NOT_APPLICABLE`。
- 真实 `out=` 共享写入默认合同为 `UNKNOWN`，所以 O2 是 `INCONCLUSIVE`；
  只有 seeded 正控使用 `SUPPORTED` 合同并触发 `VIOLATION`。
- O5 的 invalid case 同样只覆盖上述 4 个 API；valid case 从原始 Harness
  捕获，recovery 使用 valid case。
- O6 已实现 `set_grad_enabled`（grad mode）和 `fork_rng`（RNG restore）探针。
  `fork_rng` 的并发合同默认 `UNKNOWN`，因此即使观察到 restore 差异也只给
  `INCONCLUSIVE`，不直接判缺陷。
- O7 已实现 `jit.trace`、`symbolic_trace`、`make_fx` 的 basic lifecycle
  观测；双线程图结构交叉检查和清理失败根因分析尚未实现。
- LLM 适用性分析是可选路径；`llm_transformer.py` 已有接口，但默认
  使用确定性模板，不把 LLM 输出当作已验证实现。
- reducer 已实现线程数/trials/internal threads 的配置搜索；尚未缩减
  输入 tensor、共享对象数量或 barrier 位置。

## 快速使用

```bash
cd /root

# 1) 只分析，不执行
python3 -m concurrent_oracle.cli analyze \
  --harness concurrent_oracle/tests/fixtures/harnesses/llm.torch.addmm.py

# 2) 生成计划 + 捕获 case + 生成/校验 Harness（不执行）
python3 -m concurrent_oracle.cli generate \
  --harness concurrent_oracle/tests/fixtures/harnesses/llm.torch.addmm.py \
  --outputs-dir /tmp/co_generate

# 3) 完整闭环
python3 -m concurrent_oracle.cli pipeline \
  --harness concurrent_oracle/tests/fixtures/harnesses/llm.torch.matmul.py \
  --outputs-dir /tmp/co_pipeline \
  --threads 2,4 --internal-threads 1,4 --trials 10 --timeout 60

# 4) 运行一个计划
python3 -m concurrent_oracle.cli run \
  --plan /tmp/co_pipeline/plans/<plan>.json \
  --generated-harness /tmp/co_pipeline/generated_harnesses/<file>.py

# 5) 重复复现
python3 -m concurrent_oracle.cli reproduce \
  --plan /tmp/co_pipeline/plans/<plan>.json \
  --generated-harness /tmp/co_pipeline/generated_harnesses/<file>.py \
  --repeats 5

# 6) P1 targeted oracles
python3 -m concurrent_oracle.cli pipeline \
  --harness concurrent_oracle/tests/fixtures/harnesses/llm.torch.cat.py \
  --outputs-dir /tmp/co_o5 --oracle O5 --max-plans 1 --threads 2,4 --trials 2

python3 -m concurrent_oracle.cli pipeline \
  --harness concurrent_oracle/tests/fixtures/harnesses/llm.torch.set_grad_enabled.py \
  --outputs-dir /tmp/co_o6 --oracle O6 --max-plans 1 --threads 2,4 --trials 2

python3 -m concurrent_oracle.cli pipeline \
  --harness concurrent_oracle/tests/fixtures/harnesses/llm.torch.jit.trace.py \
  --outputs-dir /tmp/co_o7 --oracle O7 --max-plans 1 --threads 2 --trials 1

# 7) seeded O2 positive control
python3 -m concurrent_oracle.cli pipeline \
  --harness concurrent_oracle/tests/fixtures/seeded/lost_update_harness.py \
  --contract-dir concurrent_oracle/tests/fixtures/seeded/contracts \
  --outputs-dir /tmp/co_seed --max-plans 1 --threads 2 --trials 3

# 8) reduce a candidate configuration
python3 -m concurrent_oracle.cli reduce \
  --plan /tmp/co_seed/plans/<plan>.json \
  --generated-harness /tmp/co_seed/generated_harnesses/<file>.py \
  --contract-dir concurrent_oracle/tests/fixtures/seeded/contracts \
  --outputs-dir /tmp/co_reduce --oracle O2 --trigger-status VIOLATION
```

`--use-llm` 打开 LLM 适用性分析；默认关闭，规则路径可离线运行。

## 设计约束

- 原始 Harness 只读；复制到 `tests/fixtures/harnesses/` 使用。
- 不导入 Screen。
- 执行状态（COMPLETED/TIMEOUT/CRASH/ERROR）与 Oracle 状态
  （PASS/VIOLATION/INCONCLUSIVE/NOT_APPLICABLE）分离。
- 合同 `UNKNOWN` 时，O2/O6/O7 只能给 `INCONCLUSIVE`，不能直接判
  `VIOLATION`。
- `O1` 使用显式非法输入的 synthetic case；捕获到的合法 case 不会自动
  变成非法 case。
- 生成的并发 Harness 不含 Atheris fuzz loop；每个线程的异常必须收集。
- `outputs/` 是运行产物，已用 `.gitignore` 忽略。

## 验证

```bash
python3 -m unittest discover -s concurrent_oracle/tests -v
RUN_CONCURRENT_SMOKE=1 python3 -m unittest concurrent_oracle.tests.test_pipeline_smoke -v
```

完整的分阶段验证方案见 `VALIDATION_PLAN.md`，设计说明见 `guidance.md`。
