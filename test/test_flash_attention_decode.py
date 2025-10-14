import torch
import triton
from native_sparse_attention.ops import flash_attention_decode
from native_sparse_attention.ops import torch_attention_decode

if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 76
    max_length = 8192
    seqlens = torch.arange(batch_size, dtype=torch.int32).npu() * 128 + 1
    seqlens[seqlens > max_length] = max_length
    seqlens = seqlens[torch.randn_like(seqlens, dtype=torch.float32).argsort(-1)]
    q = (
        torch.empty(batch_size, 32, 128, device="npu")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    k = (
        torch.empty(batch_size, max_length, 4, 128, device="npu")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    v = (
        torch.empty(batch_size, max_length, 4, 128, device="npu")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    import time
    start_torch = time.time()
    o1 = torch_attention_decode(q, k, v, seqlens)
    end_torch = time.time() 
    print(f"torch time {(end_torch - start_torch) * 1000} ms")  

    start_triton = time.time()
    o2 = flash_attention_decode(q, k, v, seqlens)
    end_triton = time.time() 
    print(f"triton time {(end_triton - start_triton) * 1000} ms")  

    print(torch.allclose(o1, o2, atol=1e-2, rtol=1e-2))

@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_dim","max_length"],
            x_vals=[
                   (64,4096),
                   (64,8192),
                   (128,8192),
                   ],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=[
                "Torch",
                "Triton",
            ],
            styles=[("green", "-"), ("green", "--")],
            ylabel="ms",
            plot_name="** flash_attetion_decode **",
            args={},
        )
    )
def benchmark(head_dim, max_length,provider):
        batch_size = 76
        seqlens = torch.arange(batch_size, dtype=torch.int32).npu() * 128 + 1
        seqlens[seqlens > max_length] = max_length
        seqlens = seqlens[torch.randn_like(seqlens, dtype=torch.float32).argsort(-1)]
        q = (
            torch.empty(batch_size, 32, head_dim, device="npu")
            .uniform_(-1, 1)
            .to(torch.bfloat16)
        )
        k = (
            torch.empty(batch_size, max_length, 4, head_dim, device="npu")
            .uniform_(-1, 1)
            .to(torch.bfloat16)
        )
        v = (
            torch.empty(batch_size, max_length, 4, head_dim, device="npu")
            .uniform_(-1, 1)
            .to(torch.bfloat16)
        )
        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_attention_decode(q, k, v, seqlens),
                quantiles=quantiles,
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: flash_attention_decode(q, k, v, seqlens),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

benchmark.run(show_plots=True, print_data=True)