"""reduce, broadcast and all_reduce through commux, at any world size and root.

Run:
    torchrun --nnodes=1 --nproc-per-node=4 tests/test_collectives.py

For every root: reduce (SUM) leaves the sum of all inputs on the root and every
other rank's input unchanged, and broadcast gives every rank the root's tensor.
all_reduce (SUM) gives every rank the sum. World sizes that are not a power of
two and roots other than 0 cover any relabelling of ranks inside a collective.
The inputs are small integers, so a sum is exact in any order. Each rank counts
its failures and they are agreed with a final all_reduce, so no rank leaves a
collective early.
"""
import sys

import torch
import torch.distributed as dist
import commux

commux.register()
dist.init_process_group(backend="ucx", init_method="env://")
rank = dist.get_rank()
world = dist.get_world_size()
failures = []


def mine(r):
    return torch.arange(5, dtype=torch.float64) + 10.0 * (r + 1)


def check(name, got, want):
    if not torch.equal(got, want):
        failures.append(f"{name}: rank {rank} holds {got.tolist()}, expected {want.tolist()}")


total = sum(mine(r) for r in range(world))
for root in range(world):
    t = mine(rank)
    dist.reduce(t, dst=root, op=dist.ReduceOp.SUM)
    check(f"reduce root={root}", t, total if rank == root else mine(rank))

    t = mine(rank) if rank == root else torch.zeros(5, dtype=torch.float64)
    dist.broadcast(t, src=root)
    check(f"broadcast root={root}", t, mine(root))

t = mine(rank)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
check("all_reduce", t, total)

for msg in failures:
    print(f"FAIL {msg}", flush=True)
count = torch.tensor([float(len(failures))])
dist.all_reduce(count, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"world {world}: {int(count.item())} failure(s) over {world} roots"
          + ("" if count.item() else "; ALL commux collective TESTS PASSED"), flush=True)
dist.destroy_process_group()
sys.exit(1 if count.item() else 0)
