# PR #121 validation

Validation of the MLU support update after resolving conflicts with upstream MUSA support.

- CUDA / NVIDIA H20: 62 passed, 39 skipped; utilities/tooling: 103 passed.
- MLU / two devices: 90 passed, 10 skipped; utilities/tooling: 103 passed.
- Both wheel builds, installations, dependency checks and installed-wheel lifecycle checks passed.
- CUDA testing used a newly created H20 instance and a new virtual environment without system-site packages.

The archives contain logs, JUnit XML, environment records, source snapshots and reproduction scripts. The CUDA archive also retains the initial download failure and subsequent successful recovery logs. CUDA wheel smoke validation uses explicit in-region pool cleanup.

See `provenance.json` for the exact revision and environments and `SHA256SUMS.json` for archive checksums. MUSA, ROCm and XPU hardware validation was not run.
