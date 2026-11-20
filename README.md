# PR #124 validation

Validation of `dc1b67683d3eed5edbd4b39c3dc81b8530377ef1` on 2026-11-20.

| Platform | Full regression | Packaging/setup/release tooling |
| --- | --- | --- |
| CUDA, one A100 | 64 passed, 41 skipped | 84 passed |
| MLU, two visible devices on a shared host | 90 passed, 11 skipped, 3 failed (out of device memory) | 84 passed |

MLU's 24 backend-specific cases all passed, but its full suite did not pass: the two graph cases and training-engine case exhausted the remaining device memory. No tests were modified or excluded.

Each ZIP includes complete logs, JUnit XML, commands, exit codes, environment information, scripts, and the exact source snapshot. Read the README inside each archive for coverage and initial-attempt details. SHA256SUMS records archive checksums.
