Performance
===========

This page compares TorchPfaffian with three other libraries, on CPU and on a GPU:

- `pfapack <https://github.com/basnijholt/pfapack>`_, the reference Pfaffian library (NumPy, one matrix at a
  time, double precision, no gradients or GPU); its C implementation is used;
- `torchpfaff <https://github.com/Nuclear-Physics-with-Machine-Learning/PyPfaffian>`_, a differentiable
  PyTorch Pfaffian with a Triton kernel on CUDA, for real matrices;
- `lrux <https://github.com/ChenAo-Phys/lrux>`_, JAX low-rank updates of determinants and Pfaffians, whose full
  Pfaffian comes from `fermix <https://github.com/ChenAo-Phys/fermix>`_ (JAX/Pallas kernels on GPU, a generic
  ``jax.numpy`` fallback on CPU), jitted.

Every library computes the signed Pfaffian of the same random skew-symmetric matrices, scaled so that ``|Pf|``
stays near 1, and agrees with pfapack. The times are per matrix: for a batch, the time of the whole batch divided
by its size; pfapack is called on each matrix, the others take the batch in one call. "+ gradient" includes the
backward pass. The fastest library of each row is in bold, and "—" means not measured: torchpfaff has no complex
support and its CUDA kernel is not run above ``m = 64`` (it compiles for minutes), and pfapack has no gradient.
A method whose first call takes more than 10 s is timed from that call, compilation included (torchpfaff on CUDA
at ``m = 64``).

Measured with torchpfaffian 0.0.6 (development build), torch 2.15.0+cu132, numpy 2.5.4, pfapack 1.1.1,
torchpfaff 0.2.0, lrux 0.2.0 (fermix 0.1.0, jax 0.12.0.1), Python 3.12.16 on Linux x86_64 (AMD Ryzen 9 3900X),
one CPU thread, and an NVIDIA GeForce RTX 3070 (CUDA 13.2, Triton 3.9.0). Times on another machine differ;
``benchmarks/library_comparison.py`` reproduces the measurement and ``benchmarks/plot_results.py`` plots it (see
``benchmarks/README.md``).

Overview
--------

Each library at its best method on each input and device, divided by the fastest library there (1x is the best),
against the input size; time on the left, peak memory per call on the right, CPU above and CUDA below. Over all
the inputs, TorchPfaffian is the fastest on 35% of the CPU inputs and 36% of the CUDA ones, at a geometric mean of
1.4x and 1.6x of the fastest; lrux on 49% and 60%, at 1.7x and 1.4x.

.. image:: images/benchmark_overview.png
   :alt: Time and peak memory of each library relative to the fastest one, on CPU and CUDA.
   :width: 100%

Full Pfaffians
--------------

Pfaffian, float64:

===  =====  ================  ==================  ===================  ===============  ================  =========  ===========
m    batch  pfapack (C), CPU  TorchPfaffian, CPU  TorchPfaffian, CUDA  torchpfaff, CPU  torchpfaff, CUDA  lrux, CPU  lrux, CUDA 
===  =====  ================  ==================  ===================  ===============  ================  =========  ===========
8    1      **8.2 µs**        21.1 µs             131.8 µs             571.5 µs         128.8 µs          14.2 µs    228.5 µs   
8    256    7.5 µs            0.8 µs              1.2 µs               5.3 µs           **0.7 µs**        0.7 µs     1.7 µs     
16   1      **9.0 µs**        23.1 µs             132.1 µs             1.12 ms          137.5 µs          17.4 µs    226.4 µs   
16   256    8.3 µs            1.5 µs              2.5 µs               22.9 µs          **1.3 µs**        4.7 µs     1.7 µs     
32   1      **11.1 µs**       27.2 µs             138.7 µs             2.24 ms          232.9 µs          51.9 µs    228.7 µs   
32   256    11.9 µs           6.7 µs              7.1 µs               200.5 µs         3.3 µs            33.7 µs    **1.7 µs** 
64   1      **21.4 µs**       50.0 µs             166.2 µs             4.50 ms          116.56 s          291.8 µs   369.7 µs   
64   256    22.2 µs           30.6 µs             34.3 µs              920.8 µs         —                 480.5 µs   **4.3 µs** 
128  1      **99.8 µs**       203.8 µs            362.8 µs             10.69 ms         —                 2.57 ms    789.4 µs   
128  256    107.2 µs          180.6 µs            62.3 µs              23.42 ms         —                 4.20 ms    **13.5 µs**
256  1      **856.2 µs**      1.15 ms             1.37 ms              33.20 ms         —                 27.82 ms   2.24 ms    
256  256    899.6 µs          1.39 ms             302.5 µs             291.61 ms        —                 44.16 ms   **54.2 µs**
===  =====  ================  ==================  ===================  ===============  ================  =========  ===========

Pfaffian, complex128:

===  =====  ================  ==================  ===================  =========  ===========
m    batch  pfapack (C), CPU  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU  lrux, CUDA 
===  =====  ================  ==================  ===================  =========  ===========
8    1      **14.2 µs**       21.7 µs             192.9 µs             21.1 µs    349.0 µs   
8    256    12.7 µs           **1.3 µs**          2.2 µs               1.4 µs     3.6 µs     
16   1      **15.8 µs**       23.5 µs             190.4 µs             27.8 µs    358.1 µs   
16   256    14.7 µs           4.1 µs              5.3 µs               10.6 µs    **3.5 µs** 
32   1      **22.4 µs**       34.4 µs             207.3 µs             85.6 µs    345.7 µs   
32   256    22.3 µs           17.4 µs             19.6 µs              76.3 µs    **3.6 µs** 
64   1      **69.6 µs**       104.3 µs            280.2 µs             538.4 µs   631.1 µs   
64   256    71.1 µs           96.4 µs             101.8 µs             1.07 ms    **9.6 µs** 
128  1      **444.5 µs**      609.1 µs            846.6 µs             4.33 ms    1.81 ms    
128  256    455.8 µs          706.3 µs            112.7 µs             8.82 ms    **51.5 µs**
256  1      **3.27 ms**       4.98 ms             5.33 ms              51.04 ms   10.46 s    
256  256    3.46 ms           5.51 ms             **745.3 µs**         —          —          
===  =====  ================  ==================  ===================  =========  ===========

Pfaffian + gradient, float64:

===  =====  ==================  ===================  ===============  ================  ============  ============
m    batch  TorchPfaffian, CPU  TorchPfaffian, CUDA  torchpfaff, CPU  torchpfaff, CUDA  lrux, CPU     lrux, CUDA  
===  =====  ==================  ===================  ===============  ================  ============  ============
8    1      203.9 µs            484.8 µs             721.4 µs         495.1 µs          **34.6 µs**   337.2 µs    
8    256    4.0 µs              4.4 µs               6.9 µs           **2.0 µs**        2.3 µs        2.8 µs      
16   1      221.4 µs            497.9 µs             1.27 ms          508.7 µs          **49.6 µs**   328.7 µs    
16   256    10.5 µs             5.9 µs               27.1 µs          **2.6 µs**        11.0 µs       2.8 µs      
32   1      271.5 µs            549.1 µs             2.36 ms          558.7 µs          **107.6 µs**  329.3 µs    
32   256    63.5 µs             11.9 µs              174.5 µs         6.4 µs            68.0 µs       **2.9 µs**  
64   1      541.7 µs            815.8 µs             5.00 ms          1.22 ms           **411.0 µs**  627.3 µs    
64   256    424.1 µs            50.3 µs              1.06 ms          28.8 µs           944.2 µs      **9.1 µs**  
128  1      **1.17 ms**         1.33 ms              11.10 ms         —                 3.26 ms       1.27 ms     
128  256    4.17 ms             204.8 µs             48.16 ms         —                 6.29 ms       **35.0 µs** 
256  1      4.96 ms             **3.35 ms**          36.22 ms         —                 33.05 ms      3.42 ms     
256  256    8.34 ms             991.3 µs             294.93 ms        —                 32.23 ms      **168.3 µs**
===  =====  ==================  ===================  ===============  ================  ============  ============

Pfaffian + gradient, complex128:

===  =====  ==================  ===================  ============  ==========
m    batch  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU     lrux, CUDA
===  =====  ==================  ===================  ============  ==========
8    1      251.3 µs            609.5 µs             **49.3 µs**   10.34 s   
8    256    7.5 µs              5.8 µs               **4.4 µs**    —         
16   1      270.0 µs            589.9 µs             **63.8 µs**   —         
16   256    28.8 µs             **9.3 µs**           22.8 µs       —         
32   1      383.0 µs            716.6 µs             **169.1 µs**  —         
32   256    151.0 µs            **29.5 µs**          154.9 µs      —         
64   1      **859.2 µs**        1.19 ms              899.5 µs      —         
64   256    958.6 µs            **144.3 µs**         1.66 ms       —         
128  1      2.81 ms             **2.58 ms**          6.93 ms       —         
128  256    4.38 ms             **455.4 µs**         11.94 ms      —         
256  1      19.19 ms            **9.91 ms**          67.59 ms      —         
256  256    26.24 ms            **2.64 ms**          61.11 ms      —         
===  =====  ==================  ===================  ============  ==========

Pfaffian ratios under low-rank updates
--------------------------------------

``benchmarks/library_comparison.py --ratios`` times :class:`~torch_pfaffian.low_rank.LowRankPfaffian` and
:class:`~torch_pfaffian.low_rank.PfaffianUpdater` against lrux's ``pf_lru`` and ``pf_lru_delayed``, and against
computing the Pfaffian of the updated matrix with pfapack, in float64 (times per update). "rank 2" is one dense
update ``A - u J u^T``, "row" one row and column replaced, and "row x16" sixteen successive accepted row
replacements, with immediate or delayed (``T = 4``) inverse updates. "ratio + inverse" also updates the inverse,
which the next move needs.

rank 2, ratio:

===  =====  =========================  ==================  ===================  ===========  ==========
n    batch  pfapack (C), new Pfaffian  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU    lrux, CUDA
===  =====  =========================  ==================  ===================  ===========  ==========
32   1      12.0 µs                    17.0 µs             123.8 µs             **10.4 µs**  159.2 µs  
32   64     11.0 µs                    **0.9 µs**          2.0 µs               0.9 µs       2.5 µs    
128  1      101.2 µs                   20.7 µs             128.9 µs             **17.8 µs**  165.3 µs  
128  64     103.6 µs                   4.1 µs              5.1 µs               7.6 µs       **2.9 µs**
512  1      10.10 ms                   66.9 µs             133.3 µs             **53.0 µs**  186.0 µs  
512  64     9.92 ms                    108.8 µs            59.8 µs              64.6 µs      **9.2 µs**
===  =====  =========================  ==================  ===================  ===========  ==========

rank 2, ratio + inverse:

===  =====  =========================  ==================  ===================  ===========  ============
n    batch  pfapack (C), new Pfaffian  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU    lrux, CUDA  
===  =====  =========================  ==================  ===================  ===========  ============
32   1      **12.0 µs**                20.7 µs             235.1 µs             12.2 µs      157.4 µs    
32   64     11.0 µs                    2.4 µs              3.7 µs               **1.3 µs**   2.5 µs      
128  1      101.2 µs                   38.1 µs             237.6 µs             **36.1 µs**  160.0 µs    
128  64     103.6 µs                   26.1 µs             8.4 µs               11.8 µs      **5.0 µs**  
512  1      10.10 ms                   285.1 µs            320.4 µs             384.1 µs     **215.5 µs**
512  64     9.92 ms                    1.32 ms             106.4 µs             779.1 µs     **39.2 µs** 
===  =====  =========================  ==================  ===================  ===========  ============

row, ratio:

===  =====  =========================  ==================  ===================  ==========  ==========
n    batch  pfapack (C), new Pfaffian  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU   lrux, CUDA
===  =====  =========================  ==================  ===================  ==========  ==========
32   1      24.4 µs                    22.0 µs             112.3 µs             **8.9 µs**  166.9 µs  
32   64     11.5 µs                    0.6 µs              1.8 µs               **0.5 µs**  2.6 µs    
128  1      115.7 µs                   22.3 µs             110.5 µs             **9.5 µs**  175.2 µs  
128  64     112.4 µs                   **0.8 µs**          1.7 µs               2.1 µs      3.2 µs    
512  1      9.68 ms                    **25.3 µs**         113.2 µs             40.4 µs     165.9 µs  
512  64     10.19 ms                   2.7 µs              **1.8 µs**           50.4 µs     7.8 µs    
===  =====  =========================  ==================  ===================  ==========  ==========

row, ratio + inverse:

===  =====  =========================  ==================  ===================  ===========  ============
n    batch  pfapack (C), new Pfaffian  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU    lrux, CUDA  
===  =====  =========================  ==================  ===================  ===========  ============
32   1      24.4 µs                    26.7 µs             287.2 µs             **10.8 µs**  167.0 µs    
32   64     11.5 µs                    2.2 µs              4.5 µs               **0.8 µs**   2.6 µs      
128  1      115.7 µs                   **41.3 µs**         283.4 µs             43.0 µs      175.8 µs    
128  64     112.4 µs                   23.8 µs             7.0 µs               6.9 µs       **5.3 µs**  
512  1      9.68 ms                    257.5 µs            307.8 µs             207.9 µs     **194.1 µs**
512  64     10.19 ms                   1.31 ms             49.7 µs              769.9 µs     **39.8 µs** 
===  =====  =========================  ==================  ===================  ===========  ============

row x16, immediate updates:

===  =====  =========================  ==================  ===================  ===========  ============
n    batch  pfapack (C), new Pfaffian  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU    lrux, CUDA  
===  =====  =========================  ==================  ===================  ===========  ============
32   1      23.0 µs                    28.7 µs             276.7 µs             **22.7 µs**  112.2 µs    
32   64     11.5 µs                    2.5 µs              4.2 µs               **1.4 µs**   1.9 µs      
128  1      115.2 µs                   **42.4 µs**         273.1 µs             47.5 µs      119.9 µs    
128  64     105.5 µs                   23.5 µs             4.4 µs               16.6 µs      **2.9 µs**  
512  1      9.92 ms                    257.4 µs            274.1 µs             297.1 µs     **115.1 µs**
512  64     9.91 ms                    1.29 ms             48.0 µs              810.3 µs     **35.6 µs** 
===  =====  =========================  ==================  ===================  ===========  ============

row x16, delayed updates (T=4):

===  =====  =========================  ==================  ===================  =========  ============
n    batch  pfapack (C), new Pfaffian  TorchPfaffian, CPU  TorchPfaffian, CUDA  lrux, CPU  lrux, CUDA  
===  =====  =========================  ==================  ===================  =========  ============
32   1      **23.0 µs**                27.0 µs             449.9 µs             79.4 µs    196.5 µs    
32   64     11.5 µs                    **2.5 µs**          7.2 µs               2.8 µs     3.1 µs      
128  1      115.2 µs                   **41.3 µs**         442.2 µs             94.3 µs    201.8 µs    
128  64     105.5 µs                   34.6 µs             8.6 µs               26.7 µs    **3.2 µs**  
512  1      9.92 ms                    258.6 µs            458.4 µs             317.6 µs   **202.0 µs**
512  64     9.91 ms                    527.9 µs            82.8 µs              811.5 µs   **24.2 µs** 
===  =====  =========================  ==================  ===================  =========  ============

Reading the results
-------------------

- On a single matrix on CPU, pfapack is the fastest: TorchPfaffian has a fixed cost of about 15 µs per call
  (checks, dispatch), and its Rust kernel is up to about 2x slower than pfapack's on one large matrix.
- On a batch of small matrices, TorchPfaffian is faster than pfapack even on one thread, since one call
  amortizes the fixed cost: the typical workload of matchgate simulation.
- On CUDA, small inputs are computed by the Rust kernel on the host, and larger real ones by one Triton kernel
  per batch. TorchPfaffian is the fastest on single matrices from ``m = 16`` (and on every single complex one),
  and lrux on batches, where fermix's kernels use the GPU best (this GeForce card has slow float64
  arithmetic), except complex ``m = 256``.
- The gradient costs a few times the forward pass. It is the main reason to use TorchPfaffian or torchpfaff:
  pfapack has none.
- For one low-rank move on a single matrix on CPU, TorchPfaffian is 1.1 to 2.5x slower than lrux, except a row
  move with its inverse at ``n = 128``, a row ratio at ``n = 512`` and a rank-2 update with its inverse at
  ``n = 512``, where it is faster. On chains of 16 row
  moves it is faster from ``n = 128``, and with delayed updates at every size. lrux is faster on batched inverse
  updates on CPU, and on most inputs on CUDA, where TorchPfaffian is faster on batched row ratios.
