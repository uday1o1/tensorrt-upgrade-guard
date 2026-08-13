# Reports

Generated qualification reports remain outside Git until a reviewer verifies their source commit, environment lock, artifact hashes, model license, and claim scope.

The remote runner writes its working evidence under `.upgrade-guard/cuda-pm`.
Large engines, timing caches, profiler reports, and raw repeated outputs stay there or in an approved release asset store.

Only small aggregate tables and reviewed narratives belong under `reports/published`.
