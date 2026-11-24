| build | pp512 median (min-max, 9 samples) | vs stock | tg128 median (min-max, 9 samples) | vs stock |
|---|---|---|---|---|
| stock | 214.40 (212.13-216.99) | +0.00% | 19.47 (19.14-19.64) | **+0.00%** |
| fast-sign-all (FS=1) | 213.93 (212.14-215.39) | -0.22% | 20.12 (20.09-20.25) | **+3.35%** |
| fast-sign-fused-only (FS=2) | 213.79 (212.14-214.82) | -0.28% | 20.31 (20.22-20.44) | **+4.34%** |

| kernel group (64-token decode, rocprofv3) | stock us/call | fast-all us/call | fused-only us/call | stock total ms | fast-all total ms | fused-only total ms |
|---|---|---|---|---|---|---|
| iq2_xs mmvq n=1 fusion=false | 49.07 | 57.41 | 49.34 | 306.2 | 358.3 | 307.9 |
| iq2_xs mmvq n=1 fusion=true | 245.68 | 227.24 | 230.64 | 1916.3 | 1772.5 | 1799.0 |
| other kernels | 6.51 | 6.35 | 6.56 | 592.5 | 577.9 | 597.4 |
| other mmvq | 70.44 | 69.84 | 70.94 | 554.0 | 549.3 | 557.9 |
| **total GPU kernel time** | | | | 3369.0 | 3258.0 | 3262.2 |

total vs stock: fast-all -3.3%, fused-only -3.2%
