You are converting a PyTorch Atheris harness into a **concurrent test harness**.

Inputs:
- Original harness source.
- Captured fixed case (args/kwargs) or instructions to use a deterministic case.
- ConcurrentTestPlan (scenario, threads, shared state, oracles).

Hard requirements:
1. Return one executable Python file. No atheris fuzz loop.
2. Keep the original target API and the original input construction semantics.
3. Use the same fixed `X` for serial baseline and concurrent runs.
4. Create `threading.Thread` workers with a barrier; join with timeout.
5. Collect every exception per thread (type, message, traceback); never swallow it.
6. Record per-thread results (or digests) so Oracle checks can compare them.
7. For shared-state plans, share only the objects named in `share_keys`;
   otherwise deep-copy tensors.
8. Set `torch.set_num_threads(internal_threads)`.
9. Keep the file self-contained (stdlib + torch only); no local project imports.
10. Do not claim a bug or a thread-safety contract in comments.

If the original harness cannot be converted without changing its semantics,
return a minimal wrapper that uses the captured case and explains the
limitation in a comment; the caller will validate the serial baseline.
