# Lab 05: Memory Coalescing + Workgroup Tuning on ROCm / HIP

SAXPY benchmark (`y = a*x + y`) on AMD GPU (gfx1100) using HIP.

## How to run
```bash
hipcc saxpy_bench.cpp -o saxpy_bench
./saxpy_bench
hipcc stride.cpp -o stride
./stride
```
## Results
# Block Size Tuning (coalesced)
| Block Size | Bandwidth |
| --- | --- |
| 64 | 614.7 GB/s |
| 128 | 670.3 GB/s |
| 256 | 669.9 GB/s |
| 512 | 669.3 GB/s |
| 1024 | 669.9 GB/s |

# Stride/ Coalescing (block = 256)

| Stride | Bandwidth |
| --- | --- |
| 1 (coalesced) | 704.0 GB/s |
| 4 | 179.6 GB/s |
| 16 | 48.3 GB/s |

## Discussion

Block size had a small effect. Performance rose from 614 GB/s at block=64 to ∼670 GB/s at block=128, then plateaued.
Increasing block size beyond 128 gave no further benefit, indicating the kernel is memory-bandwidth bound, not thread-count bound.
Stride had a massive effect. Increasing stride from 1 to 4 dropped bandwidth from 704 GB/s to 180 GB/s (3.9x slower),
and stride=16 dropped it to 48 GB/s (14.6x slower). 
This confirms that non-coalesced accesses waste memory bandwidth because each 128-byte cacheline fetch is only partially used.
Coalesced access is therefore critical for SAXPY performance on ROCm/HIP.

## What I Learned

- SAXPY is memory-bound, not compute-bound — performance is limited by how fast we can move data, not by math.
- Coalesced access is critical. When threads access consecutive addresses (stride = 1), the GPU uses full cachelines and hits ~704 GB/s.
- Strided access wastes bandwidth. Stride = 4 dropped to ~180 GB/s and stride = 16 dropped to ~48 GB/s because most of each 128-byte cacheline fetch goes unused.
- Block size matters only up to a point. Going from 64 to 128 improved bandwidth from ~615 GB/s to ~670 GB/s, but 256, 512, and 1024 gave no further gain once the GPU was saturated.
- In HIP/ROCm, the same kernel can be 14x slower just by changing the access pattern, even with the same amount of work.

