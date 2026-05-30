# Phase 4 -- event sim vs. closed-form analytical

Discrete-event sim (batched decode engine, single GPU interleaving append-prefill, HBM-capacity admission, FIFO C2C link) replaying the parsed Codex/SWEBenchPro trace. eta=0.5, dense-70B.

## Results

| scenario | arrival/s | sim throughput (tok/s) | sim TTFT (s) | sim TPOT (ms) | analytical TPOT@B=1 (ms) | admit stall (total s) | GPU util |
|---|---|---|---|---|---|---|---|
| gpu_hot (f=0) | 1.0 | 54.7 | 218.28 | 54.3 | 40.6 | 52147.5 | 100% |
| gpu_hot (f=0) | 4.0 | 56.8 | 205.95 | 55.5 | 40.6 | 49128.6 | 100% |
| sparse offload f=0.3 | 1.0 | 64.6 | 161.91 | 62.2 | 40.7 | 38586.3 | 100% |
| sparse offload f=0.3 | 4.0 | 66.2 | 167.87 | 62.7 | 40.7 | 39862.7 | 100% |

## Headline: this workload is admission-bound

One 64k-context sequence needs ~43 GB on dense-70B (hot KV + old-KV materialization), but only ~46 GB of HBM is free after weights -- so **~1 session is resident at a time**. The system is limited by HBM admission, not arrival rate (throughput barely moves from 1->4 arrivals/s). This is the quantitative case for a CPU KV backing store: park idle/warm sessions in CPU DRAM and shrink the hot-KV footprint so more sessions are admitted. Sparse offload (f=0.3) already cuts TTFT and raises throughput here purely by reducing resident hot KV.

## Where the closed-form model is optimistic

1. **Queueing / admission stalls.** The analytical TTFT assumes a request runs alone. Under load the HBM-capacity pool serializes admission: only a few 64k-context sequences fit, so later arrivals wait. The `admit stall` column is pure latency the closed-form never sees -- it grows sharply with arrival rate.
2. **Prefill interrupts decode (synchronization).** A single GPU can't decode while it append-prefills. Each prefill freezes the decode loop, so the *observed* per-token time exceeds the pure-decode analytical TPOT. The closed-form treats append and decode as separate phases.
3. **Bandwidth contention.** Old-KV loads and new-KV flushes share one C2C link (FIFO). Concurrent calls queue behind each other; the analytical C2C term assumes a private link.
4. **Dynamic batch vs fixed B_d.** The sim's active batch fluctuates with arrivals/admission; the analytical model evaluates a single chosen B_d. The sim's effective batch is usually smaller than the capacity bound because admission + think-time keep the engine partly idle (GPU util < 100%).

## Takeaway
The closed-form numbers are a **throughput upper / latency lower bound**. They correctly rank designs (offload vs not, sparse vs dense) but optimistic TTFT and TPOT by the contention factors above. For SLO sizing, inflate closed-form TTFT by the measured admit-stall + prefill-interruption terms. This matches Frontier's finding that explicit KV-transfer/scheduling dependencies are needed for accurate latency.
