# PR #124 validation

| Platform | Tested commit | Full regression | Packaging/setup/release tooling |
| --- | --- | --- | --- |
| CUDA / A100 | dc1b676 | 64 passed, 41 skipped | 84 passed |
| MLU, two idle devices | 46e294a | 93 passed, 11 skipped | 84 passed |

Both commits have identical source trees; the amend only corrected commit identity metadata. All 24 MLU-specific cases passed. The previous three MLU OOM failures passed in the full rerun with free device memory.

Use torch-memory-saver-mlu-46e294a.zip for the latest complete MLU report. The old MLU archive is retained as historical evidence. Each ZIP includes complete logs, JUnit XML, commands, exit statuses, environment details, scripts and source snapshot. SHA256SUMS records archive checksums.
