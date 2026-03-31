# TODO

## Burn migration remaining items

- **Conv fuse()**: The BatchNorm-into-Conv fusion for inference was removed. Burn's `BatchNorm` uses `RunningState` which doesn't expose its internal values via public API. Burn already handles train/eval mode automatically, so this is a performance optimization only — not a correctness issue.
