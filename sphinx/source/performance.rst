Performance
===========

This page compares TorchPfaffian with `pfapack <https://github.com/basnijholt/pfapack>`_, the reference
Pfaffian library, on CPU. pfapack computes one NumPy matrix at a time in double precision, without gradients
or GPU support; the faster of its two implementations, the C one, is used here. Both compute the signed
Pfaffian of the same random skew-symmetric matrices, scaled so that ``|Pf|`` stays near 1, and agree to
within 4e-12 relative difference in every case.

The times are per matrix: for a batch, the time of the whole batch divided by its size. TorchPfaffian takes
the batch in one call; pfapack is called on each matrix. "torchpfaffian + gradient" includes the backward
pass, which pfapack does not offer. The speed-up is the pfapack time divided by the TorchPfaffian time.

Measured with torchpfaffian 0.0.6 (development build), torch 2.15.0+cpu, numpy 2.5.4, pfapack 1.1.1, on Python 3.12.2 on Linux-7.0.0-42-generic-x86_64-with-glibc2.39, x86_64, 16 cores. Times on another machine differ; the script
``benchmarks/library_comparison.py`` reproduces the measurement (see ``benchmarks/README.md``).

One thread
----------

float64:

===  =====  ===========  =============  ========  ========================
m    batch  pfapack (C)  torchpfaffian  speed-up  torchpfaffian + gradient
===  =====  ===========  =============  ========  ========================
8    1      13.6 µs      55.8 µs        0.24x     389.3 µs
8    256    3.9 µs       1.0 µs         3.85x     2.9 µs
32   1      5.9 µs       23.4 µs        0.25x     203.7 µs
32   256    5.6 µs       9.3 µs         0.60x     54.3 µs
128  1      73.8 µs      156.1 µs       0.47x     703.2 µs
128  256    74.0 µs      138.4 µs       0.53x     1.35 ms
256  1      577.0 µs     846.3 µs       0.68x     2.99 ms
256  256    573.8 µs     1.22 ms        0.47x     6.36 ms
===  =====  ===========  =============  ========  ========================

complex128:

===  =====  ===========  =============  ========  ========================
m    batch  pfapack (C)  torchpfaffian  speed-up  torchpfaffian + gradient
===  =====  ===========  =============  ========  ========================
8    1      10.2 µs      39.4 µs        0.26x     242.5 µs
8    256    6.2 µs       1.9 µs         3.30x     5.1 µs
32   1      12.3 µs      34.6 µs        0.36x     272.9 µs
32   256    11.2 µs      15.8 µs        0.71x     134.5 µs
128  1      264.3 µs     456.6 µs       0.58x     1.61 ms
128  256    294.1 µs     479.5 µs       0.61x     3.22 ms
256  1      1.57 ms      3.16 ms        0.50x     10.52 ms
256  256    1.70 ms      3.74 ms        0.45x     17.55 ms
===  =====  ===========  =============  ========  ========================

Eight threads
-------------

The Rust kernel parallelizes over the batch and PyTorch over the linear algebra; pfapack uses one thread.

float64:

===  =====  ===========  =============  ========  ========================
m    batch  pfapack (C)  torchpfaffian  speed-up  torchpfaffian + gradient
===  =====  ===========  =============  ========  ========================
8    1      13.6 µs      57.5 µs        0.24x     338.1 µs
8    256    3.8 µs       1.1 µs         3.52x     6.4 µs
32   1      16.2 µs      37.7 µs        0.43x     215.3 µs
32   256    5.7 µs       3.4 µs         1.64x     16.6 µs
128  1      121.2 µs     130.3 µs       0.93x     552.2 µs
128  256    67.9 µs      48.9 µs        1.39x     369.4 µs
256  1      571.2 µs     902.0 µs       0.63x     2.44 ms
256  256    584.0 µs     527.0 µs       1.11x     2.11 ms
===  =====  ===========  =============  ========  ========================

complex128:

===  =====  ===========  =============  ========  ========================
m    batch  pfapack (C)  torchpfaffian  speed-up  torchpfaffian + gradient
===  =====  ===========  =============  ========  ========================
8    1      13.6 µs      41.2 µs        0.33x     315.1 µs
8    256    8.1 µs       1.2 µs         6.91x     7.4 µs
32   1      15.6 µs      45.6 µs        0.34x     386.4 µs
32   256    15.3 µs      6.9 µs         2.23x     56.1 µs
128  1      673.2 µs     921.5 µs       0.73x     2.40 ms
128  256    348.4 µs     136.2 µs       2.56x     1.33 ms
256  1      3.46 ms      5.46 ms        0.63x     8.96 ms
256  256    1.69 ms      1.24 ms        1.37x     6.17 ms
===  =====  ===========  =============  ========  ========================

Reading the results
-------------------

- On a single matrix, pfapack is faster. TorchPfaffian has a fixed cost of a few tens of microseconds per
  call (input checks, dispatch and the autograd machinery), and its Parlett-Reid kernel is up to about 2x
  slower than pfapack's on one large matrix.
- On a batch of small matrices, TorchPfaffian is faster even on one thread: one call amortizes the fixed
  cost. This is the typical workload of matchgate simulation. On a batch of large matrices (m = 128 and
  more), it is slower on one thread and faster with several, since its kernel runs the matrices in
  parallel while pfapack is called on one matrix at a time.
- The gradient costs a few times the forward pass. It is the main reason to use TorchPfaffian: pfapack has
  none, and it would otherwise need a separate inverse or a finite-difference approximation.
- On a GPU, TorchPfaffian runs the whole computation on the device (not measured here).
