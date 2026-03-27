# Benchmark Baseline

- **Environment**: Local environment.
- **Task**: Reading a 50MB file 100 times.
- **Sync Read (`std::fs::read`)**: 4.569 seconds
- **Async Read (`tokio::fs::read`)**: 5.082 seconds

*Note: In local non-concurrent microbenchmarks, synchronous read can often perform slightly better than async due to lower overhead from async state machines and scheduling. The true benefit of async read in `index_file` is that it doesn't block the async runtime thread, which significantly improves concurrency and prevents thread starvation when reading multiple files in a real-world scenario.*

## Post-Change Benchmark

- **Sync Read (`std::fs::read`)**: 4.200 seconds
- **Async Read (`tokio::fs::read`)**: 4.646 seconds

As expected, in a purely sequential microbenchmark there is slight overhead in the async executor. However, replacing `std::fs::read` with `tokio::fs::read` in an async runtime prevents blocking the thread pool. The threadpool in a real application will be able to process other tasks rather than busy waiting. We can't easily measure this without a large scale load test but replacing synchronous blocking I/O with asynchronous non-blocking I/O is widely acknowledged as an optimization in concurrent Rust contexts, removing latency spikes and poor scale.
