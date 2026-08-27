# Pending specs

Tests in this directory describe modules that do not exist yet. They are excluded
from collection (see `../conftest.py`) so the pytest report reflects real coverage
instead of a phantom skip.

- `test_flow_php_repo_dataset.py` — contract for `dataloader.flow_php_repo_dataset`.
  Move it back up one level once that module lands.
