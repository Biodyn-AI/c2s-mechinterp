# Historical audit source provenance

`phase8_independent_audit_checks.py.txt` preserves the exact source bytes of the earlier metadata and synthetic audit executed from `experiments/results/audits/`. Its SHA256 is `6226bb855efd94e1819b16c3bcdd27ae5d0c0f36c61c7eefde728b1b0f9b2aa0`.

This is a provenance record, not an executable public entry point. It derives dataset and producer paths from that original directory layout, executes at import time and writes an audit report in that layout. Renaming or importing this copy does not make it portable. The original executed copy and its report remain in the numerical archive.

Use `revision/audit_phase8_saved.py` for the portable, explicit-input saved-output audit and `revision/test_audit_phase8_saved.py` for its synthetic fixtures. The later audit checks saved cell alignment, self-exclusion, full BH families and finite tied-score AP expectations. It does not reproduce every historical metadata check automatically.
