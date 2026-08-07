# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark for the MP block-transfer kernel's gather stage.

Times ``lmc_ops.multi_layer_block_kv_transfer`` (D2H: paged blocks ->
contiguous GPU staging objects) across a matrix of KV-cache shapes and
per-launch object counts. Both endpoints live on the GPU, isolating the
gather kernel from DMA and orchestration.

Usage::

    python benchmarks/mp_transfer_microbench.py [--iters 50] [--nl 32]
"""

# Standard
import argparse

# Third Party
import torch

# First Party
import lmcache.c_ops as lmc_ops

# Matches the store path: one LMCache chunk per memory object.
TOKENS_PER_OBJECT = 256
PAGED_BLOCK_SIZE = 16
BLOCKS_PER_OBJECT = TOKENS_PER_OBJECT // PAGED_BLOCK_SIZE
WARMUP_ITERS = 5


class ShapeCase:
    """One KV-cache geometry to benchmark.

    Args:
        name: Row label, e.g. ``"MLA"``.
        engine_kv_format: ``lmc_ops.EngineKVFormat`` value describing the
            paged-buffer layout.
        kv_size: K/V axis length (2 split, 1 fused).
        nh: Number of KV heads per device.
        hs: Head (or latent) size in elements.
    """

    def __init__(
        self,
        name: str,
        engine_kv_format: "lmc_ops.EngineKVFormat",
        kv_size: int,
        nh: int,
        hs: int,
        nb: int,
    ) -> None:
        self.name = name
        self.engine_kv_format = engine_kv_format
        self.kv_size = kv_size
        self.nh = nh
        self.hs = hs
        self.nb = nb


def build_cases() -> list[ShapeCase]:
    """Return the benchmark shape matrix.

    All cases use bf16. ``thread_dim_x`` saturates at 32 for every listed
    ``hs``, so per-block warp count is ``kv_size * nh`` -- the quantity
    under study.
    """
    fmt = lmc_ops.EngineKVFormat
    return [
        ShapeCase(
            "MHA (nh=32)", fmt.NL_X_TWO_NB_BS_NH_HS, kv_size=2, nh=32, hs=128, nb=1024
        ),
        ShapeCase(
            "GQA TP4 (nh=2)", fmt.NL_X_TWO_NB_BS_NH_HS, kv_size=2, nh=2, hs=128, nb=8192
        ),
        ShapeCase(
            "GQA TP8 (nh=1)", fmt.NL_X_TWO_NB_BS_NH_HS, kv_size=2, nh=1, hs=128, nb=8192
        ),
        ShapeCase(
            "MLA (nh=1, hs=576)", fmt.NL_X_NB_BS_HS, kv_size=1, nh=1, hs=576, nb=8192
        ),
    ]


def make_shape_desc(case: ShapeCase, nl: int) -> "lmc_ops.PageBufferShapeDesc":
    """Build the kernel shape descriptor for *case* with *nl* layers."""
    desc = lmc_ops.PageBufferShapeDesc()
    desc.kv_size = case.kv_size
    desc.nl = nl
    desc.nb = case.nb
    desc.bs = PAGED_BLOCK_SIZE
    desc.nh = case.nh
    desc.hs = case.hs
    desc.element_size = 2  # bf16
    if hasattr(desc, "block_stride_elems"):  # absent in older builds
        desc.block_stride_elems = 0
    return desc


def make_paged_tensors(
    case: ShapeCase, nl: int, device: torch.device
) -> list[torch.Tensor]:
    """Allocate per-layer paged buffers matching *case*'s layout."""
    fmt = lmc_ops.EngineKVFormat
    if case.engine_kv_format == fmt.NL_X_TWO_NB_BS_NH_HS:
        shape = [2, case.nb, PAGED_BLOCK_SIZE, case.nh, case.hs]
    elif case.engine_kv_format == fmt.NL_X_NB_BS_HS:
        shape = [case.nb, PAGED_BLOCK_SIZE, case.hs]
    else:
        raise ValueError(f"Unsupported format: {case.engine_kv_format}")
    return [torch.randn(shape, dtype=torch.bfloat16, device=device) for _ in range(nl)]


def run_case(
    case: ShapeCase,
    nl: int,
    num_objects: int,
    iters: int,
    device: torch.device,
) -> float:
    """Time the gather kernel for one (shape, num_objects) cell.

    Returns:
        Gather throughput in GB/s, averaged over *iters* launches.
    """
    paged = make_paged_tensors(case, nl, device)
    paged_ptrs = torch.tensor(
        [t.data_ptr() for t in paged], dtype=torch.int64, device=device
    )
    hidden = case.nh * case.hs
    # Rotate over independent (objects, block_ids) sets so successive
    # launches touch fresh memory; otherwise the per-iteration working set
    # stays L2-resident and the result measures cache, not DRAM, bandwidth.
    num_sets = 4
    total_blocks = num_objects * BLOCKS_PER_OBJECT
    object_sets = []
    block_id_sets = []
    for _ in range(num_sets):
        objects = [
            torch.zeros(
                [case.kv_size, nl, TOKENS_PER_OBJECT, hidden],
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in range(num_objects)
        ]
        object_sets.append(objects)
        block_id_sets.append(
            torch.randperm(case.nb, device=device)[:total_blocks].to(torch.int64)
        )
    object_ptr_sets = [[t.data_ptr() for t in objs] for objs in object_sets]

    shape_desc = make_shape_desc(case, nl)
    nbytes_per_launch = sum(t.numel() * t.element_size() for t in object_sets[0])

    def launch(i: int) -> None:
        lmc_ops.multi_layer_block_kv_transfer(
            paged_ptrs,
            object_ptr_sets[i % num_sets],
            block_id_sets[i % num_sets],
            device,
            lmc_ops.TransferDirection.D2H,
            shape_desc,
            TOKENS_PER_OBJECT,
            case.engine_kv_format,
            0,
        )

    for i in range(WARMUP_ITERS):
        launch(i)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        launch(i)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_s = start.elapsed_time(end) / 1e3
    return nbytes_per_launch * iters / elapsed_s / 1e9


def main() -> None:
    """Run the full matrix and print a GB/s table with batch speedups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50, help="Timed launches/cell.")
    parser.add_argument("--nl", type=int, default=32, help="Layers per launch.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = torch.device("cuda:0")
    torch.cuda.init()

    batches = [1, 2, 4]
    print(f"gather-stage throughput, GB/s (nl={args.nl}, iters={args.iters})")
    header = f"{'shape':<22}" + "".join(f"batch={b:<8}" for b in batches) + "x(4/1)"
    print(header)
    print("-" * len(header))
    for case in build_cases():
        results = []
        for b in batches:
            results.append(run_case(case, args.nl, b, args.iters, device))
            torch.cuda.empty_cache()
        speedup = results[-1] / results[0] if results[0] > 0 else 0.0
        row = f"{case.name:<22}" + "".join(f"{r:<14.1f}" for r in results)
        print(f"{row}{speedup:.2f}")


if __name__ == "__main__":
    main()
