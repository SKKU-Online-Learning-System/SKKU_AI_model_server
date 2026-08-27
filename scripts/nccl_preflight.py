from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    value = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(
            f"NCCL all_reduce returned {value.item()}, expected {expected}"
        )

    dist.barrier()
    if rank == 0:
        print(
            f"NCCL preflight OK: world_size={world_size} "
            f"sum={value.item():.0f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
