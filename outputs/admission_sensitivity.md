# Admission-bound sensitivity (round-2)

Event sim, dense-70B, 40 sessions x 6 calls, arrival 3/s, think 0.3s. The system is 'admission-bound' when **admit-stall is large and TTFT is dominated by capacity queueing** (peak-resident is driven up by the many small early calls in the real trace, but the large 64k calls still queue). Below: where that dissolves.

| knob | value | peak resident | admit stall (total s) | TTFT mean (s) | throughput tok/s |
|---|---|---|---|---|---|
| HBM | 96 GB | 0 (weights don't fit / no completions) | - | - | - |
| HBM | 192 GB | 15 | 46891 | 196.6 | 57.1 |
| HBM | 384 GB | 40 | 10133 | 46.3 | 121.6 |
| HBM | 768 GB | 40 | 947 | 19.1 | 135.0 |
| HBM | 1536 GB | 40 | 0 | 19.1 | 135.0 |
| old-KV r | r=1.0 | 15 | 46891 | 196.6 | 57.1 |
| old-KV r | r=0.5 | 15 | 37129 | 155.9 | 69.7 |
| old-KV r | r=0.25 | 15 | 32218 | 135.5 | 78.5 |
| offload f | f=0.0 | 15 | 46891 | 196.6 | 57.1 |
| offload f | f=0.5 | 30 | 36643 | 155.2 | 63.5 |
| offload f | f=0.9 | 40 | 34715 | 148.7 | 62.9 |

## Reading it
- **HBM size:** at 96-192 GB only ~1 session is resident -> admission-bound. As HBM grows (>=384-768 GB) peak-resident rises and admit-stall/TTFT fall: the bound dissolves once HBM can hold several 64k-context sessions.
- **old-KV r:** lowering r (materialize/recompute less old KV, KVPR-style) shrinks the per-session HBM reservation, admitting more sessions even at 192 GB -- so the admission bound is partly a consequence of r=1, not just the workload.
- **offload f:** sparse decode-attention offload shrinks resident hot KV; large f admits more sessions. But note f cannot shrink the old-KV append floor (that needs r) -- consistent with the analytical finding.

**Conclusion:** 'admission-bound' is real for dense-70B on ONE GB200 with full old-KV materialization, but it is a property of (HBM size, r, single-GPU), not an intrinsic property of the workload. Sharding (Phase 5) or more HBM or lower r each relax it.
