"""Stand-in for OneTrainer's scripts/train.py. Reads the run config it is handed
and behaves per its "fake_mode" key: ok | fail | oom | hang. Emits progress lines
shaped like OneTrainer's tqdm output (which the real trainer writes to stderr;
TrainerProcess merges channels, so stdout here is equivalent)."""
import json
import os
import sys
import time

def main():
    # Test hook: prove an extra environment variable reached the child.
    if os.environ.get("KF_ECHO_ENV"):
        print("env:" + os.environ["KF_ECHO_ENV"], flush=True)
    cfg_path = sys.argv[sys.argv.index("--config-path") + 1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mode = cfg.get("fake_mode", "ok")
    if mode == "hang":
        time.sleep(300)
        sys.exit(3)
    if mode == "slow":
        print("step:  25%|##        | 1/4 [00:00<00:00,  9.90it/s, loss=0.30]", flush=True)
        time.sleep(2)
        print("step: 100%|##########| 4/4 [00:02<00:00,  9.90it/s, loss=0.20, smooth loss=0.21]", flush=True)
        sys.exit(0)
    print("epoch:   0%|          | 0/2 [00:00<?, ?it/s]", flush=True)
    print("step:   0%|          | 0/4 [00:00<?, ?it/s]", flush=True)
    print("step:  50%|#####     | 2/4 [00:00<00:00,  9.90it/s, loss=0.25, smooth loss=0.24]", flush=True)
    if mode == "oom":
        print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB", flush=True)
        sys.exit(1)
    if mode == "oom-truncated":
        sys.stdout.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB")
        sys.stdout.flush()
        sys.exit(1)
    if mode == "fail":
        print("RuntimeError: something broke", flush=True)
        sys.exit(1)
    print("step: 100%|##########| 4/4 [00:01<00:00,  9.90it/s, loss=0.20, smooth loss=0.21]", flush=True)
    print("epoch: 100%|##########| 2/2 [00:01<00:00,  1.90it/s]", flush=True)
    out = cfg.get("output_model_destination", "")
    if out:
        with open(out, "wb") as f:
            f.write(b"fake-safetensors")
    sys.exit(0)

if __name__ == "__main__":
    main()
