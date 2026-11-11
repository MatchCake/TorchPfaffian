use half::f16;
use num_complex::Complex;
use num_traits::{One, Zero};
use numpy::ndarray::{Array1, Array2, Array3, Axis};
use numpy::{Complex32, Complex64, IntoPyArray, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

const PIVOT_EPSILON: f64 = 1e-30;

/// Batches with at least this many matrices are evaluated in parallel across threads; smaller
/// batches run serially to avoid thread-pool overhead.
const PARALLEL_BATCH_THRESHOLD: usize = 8;

/// Magnitude used to rank pivots and to test for a numerically zero pivot.
///
/// For real scalars this is the absolute value; for complex scalars it is the modulus. The pivot
/// choice and the singularity test only ever compare magnitudes, so the elimination itself stays
/// generic over real and complex precisions while the algebra runs in the native (possibly complex)
/// type.
trait Magnitude {
    fn magnitude(&self) -> f64;
}

impl Magnitude for f16 {
    fn magnitude(&self) -> f64 {
        self.to_f64().abs()
    }
}

impl Magnitude for f32 {
    fn magnitude(&self) -> f64 {
        self.abs() as f64
    }
}

impl Magnitude for f64 {
    fn magnitude(&self) -> f64 {
        self.abs()
    }
}

impl Magnitude for Complex<f32> {
    fn magnitude(&self) -> f64 {
        self.norm() as f64
    }
}

impl Magnitude for Complex<f64> {
    fn magnitude(&self) -> f64 {
        self.norm()
    }
}

/// Unit-modulus phase ``self / |self|`` of a nonzero scalar, staying in the native type.
///
/// For real scalars this is the sign (``+-1``); for complex scalars it is the unit phasor. Used only
/// by the log-domain (slog) kernel, so it is implemented for the four precisions that expose a slog
/// entry point (not ``f16``). Callers guard against a zero magnitude before dividing.
trait SlogScalar: Magnitude {
    fn unit_phase(self) -> Self;
}

impl SlogScalar for f32 {
    fn unit_phase(self) -> f32 {
        self / self.abs()
    }
}

impl SlogScalar for f64 {
    fn unit_phase(self) -> f64 {
        self / self.abs()
    }
}

impl SlogScalar for Complex<f32> {
    fn unit_phase(self) -> Complex<f32> {
        self / self.norm()
    }
}

impl SlogScalar for Complex<f64> {
    fn unit_phase(self) -> Complex<f64> {
        self / self.norm()
    }
}

/// The arithmetic of the Parlett-Reid elimination (every precision, ``f16`` included).
trait PfaffianScalar:
    Copy + Zero + One + Magnitude + Neg<Output = Self> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl<T> PfaffianScalar for T where
    T: Copy + Zero + One + Magnitude + Neg<Output = T> + Add<Output = T> + Sub<Output = T> + Mul<Output = T>
        + Div<Output = T>
{
}

/// The arithmetic of the log-domain elimination.
trait SlogKernelScalar: PfaffianScalar + SlogScalar {}

impl<T: PfaffianScalar + SlogScalar> SlogKernelScalar for T {}

/// Defines ``$name``, which runs ``$inner`` (an ``#[inline(always)]`` kernel) compiled for AVX2 and FMA when the CPU
/// has them, checked at run time (the check is cached by the standard library), and as compiled for the target
/// otherwise. The wheels target baseline x86-64 (SSE2), on which these loops vectorize only two doubles wide.
/// The kernel is called directly from the ``#[target_feature]`` function, so that it is inlined into it and its
/// loops are generated with the features.
macro_rules! with_avx2 {
    (fn $name:ident<$scalar:ident: $bound:path>($($argument:ident: $kind:ty),* $(,)?) -> $result:ty => $inner:ident) => {
        fn $name<$scalar: $bound>($($argument: $kind),*) -> $result {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                fn accelerated<$scalar: $bound>($($argument: $kind),*) -> $result {
                    $inner($($argument),*)
                }
                if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                    // SAFETY: the CPU supports the enabled features, checked just above.
                    return unsafe { accelerated($($argument),*) };
                }
            }
            $inner($($argument),*)
        }
    };
}

/// Signed Pfaffian of a single skew-symmetric matrix held as a flat row-major buffer.
///
/// Generic over the scalar type so the same algorithm serves ``f32``/``f64`` and their complex
/// counterparts: the arithmetic runs in the native type while pivoting compares magnitudes. Working
/// on a flat ``Vec`` (indexed ``data[row * dimension + col]``) keeps the hot rank-2 Schur update over
/// a contiguous row slice, which the compiler can auto-vectorize. The caller owns the buffer, so this
/// runs under ``py.allow_threads`` with no borrow of Python-managed data. A pivot column whose
/// magnitude falls below ``PIVOT_EPSILON`` yields ``0`` (the historical linear-domain threshold; the
/// slog kernel uses an exact-zero test instead).
#[inline(always)]
fn pfaffian_from_flat<T>(mut data: Vec<T>, dimension: usize) -> T
where
    T: Copy
        + Zero
        + One
        + Magnitude
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    if dimension % 2 == 1 {
        return T::zero();
    }
    if dimension == 0 {
        return T::one();
    }
    let mut sign = T::one();
    let mut column = 0usize;
    while column + 2 < dimension {
        // Partial pivoting: largest magnitude data[row, column] for row > column + 1.
        let mut pivot_row = column + 2;
        let mut best = data[(column + 2) * dimension + column].magnitude();
        for row in (column + 3)..dimension {
            let candidate = data[row * dimension + column].magnitude();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if best > data[(column + 1) * dimension + column].magnitude() {
            // Congruence swap of rows then columns column+1 <-> pivot_row.
            for index in 0..dimension {
                data.swap((column + 1) * dimension + index, pivot_row * dimension + index);
            }
            for index in 0..dimension {
                data.swap(index * dimension + (column + 1), index * dimension + pivot_row);
            }
            sign = -sign;
        }
        let pivot = data[(column + 1) * dimension + column];
        if pivot.magnitude() < PIVOT_EPSILON {
            return T::zero();
        }
        // Rank-2 skew Schur-complement update on the trailing block, read from originals.
        let base = column + 2;
        let length = dimension - base;
        let tau: Vec<T> = (0..length).map(|k| data[(base + k) * dimension + column] / pivot).collect();
        let next: Vec<T> = (0..length).map(|k| data[(base + k) * dimension + (column + 1)]).collect();
        for row_offset in 0..length {
            let tau_row = tau[row_offset];
            let next_row = next[row_offset];
            let start = (base + row_offset) * dimension + base;
            let row = &mut data[start..start + length];
            for column_offset in 0..length {
                row[column_offset] =
                    row[column_offset] + tau_row * next[column_offset] - next_row * tau[column_offset];
            }
        }
        column += 2;
    }
    let mut pfaffian = sign;
    let mut index = 0usize;
    while index < dimension {
        pfaffian = pfaffian * data[index * dimension + (index + 1)];
        index += 2;
    }
    pfaffian
}

with_avx2!(fn pfaffian_fast<T: PfaffianScalar>(data: Vec<T>, dimension: usize) -> T => pfaffian_from_flat);
with_avx2!(fn slog_pfaffian_fast<T: SlogKernelScalar>(data: Vec<T>, dimension: usize) -> (T, f64) => slog_pfaffian_from_flat);

/// Signed Pfaffian of a single owned ``Array2``: copies it into a flat row-major buffer and runs
/// [`pfaffian_from_flat`]. This is the per-matrix worker that [`signed_pfaffian_owned`] maps over the
/// batch (the copy from the numpy view happens once in [`owned_matrices`] and once here, matching the
/// released kernel); the slog entry points use their own single-copy [`flat_matrices`] path.
fn pfaffian_one<T>(matrix: Array2<T>) -> T
where
    T: Copy
        + Zero
        + One
        + Magnitude
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let dimension = matrix.nrows();
    pfaffian_fast(matrix.iter().copied().collect(), dimension)
}

/// Log-domain signed Pfaffian ``(phase, log|Pf|)`` of a single skew-symmetric matrix (flat buffer).
///
/// The same Parlett-Reid elimination as [`pfaffian_from_flat`], but the linear product of the
/// superdiagonal entries is factored into a unit-modulus ``phase`` (native type) and a ``log|Pf|``
/// accumulated in ``f64`` (``+= |pivot|.ln()``), which neither overflows for large matrices nor
/// underflows for tiny Pfaffians. A zero pivot is detected exactly (magnitude ``== 0``, not an epsilon
/// threshold), giving ``(0, -inf)`` for that matrix; its remaining updates would be benign, so the
/// early return is equivalent. Odd ``dimension`` gives ``(0, -inf)`` and ``dimension == 0`` gives
/// ``(1, 0)``. Reconstruction ``phase * exp(log_abs)`` equals the value [`pfaffian_from_flat`] would
/// return, up to the log/exp rounding, so it matches ``signed_pfaffian_*`` within tolerance.
#[inline(always)]
fn slog_pfaffian_from_flat<T>(mut data: Vec<T>, dimension: usize) -> (T, f64)
where
    T: Copy
        + Zero
        + One
        + SlogScalar
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    if dimension % 2 == 1 {
        return (T::zero(), f64::NEG_INFINITY);
    }
    if dimension == 0 {
        return (T::one(), 0.0);
    }
    let mut sign = T::one();
    let mut column = 0usize;
    while column + 2 < dimension {
        // Partial pivoting: largest magnitude data[row, column] for row > column + 1.
        let mut pivot_row = column + 2;
        let mut best = data[(column + 2) * dimension + column].magnitude();
        for row in (column + 3)..dimension {
            let candidate = data[row * dimension + column].magnitude();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if best > data[(column + 1) * dimension + column].magnitude() {
            // Congruence swap of rows then columns column+1 <-> pivot_row; each swap flips the phase.
            for index in 0..dimension {
                data.swap((column + 1) * dimension + index, pivot_row * dimension + index);
            }
            for index in 0..dimension {
                data.swap(index * dimension + (column + 1), index * dimension + pivot_row);
            }
            sign = -sign;
        }
        let pivot = data[(column + 1) * dimension + column];
        if pivot.magnitude() == 0.0 {
            // Exact zero pivot column: Pf = 0 with no epsilon fudge.
            return (T::zero(), f64::NEG_INFINITY);
        }
        // Rank-2 skew Schur-complement update on the trailing block, read from originals.
        let base = column + 2;
        let length = dimension - base;
        let tau: Vec<T> = (0..length).map(|k| data[(base + k) * dimension + column] / pivot).collect();
        let next: Vec<T> = (0..length).map(|k| data[(base + k) * dimension + (column + 1)]).collect();
        for row_offset in 0..length {
            let tau_row = tau[row_offset];
            let next_row = next[row_offset];
            let start = (base + row_offset) * dimension + base;
            let row = &mut data[start..start + length];
            for column_offset in 0..length {
                row[column_offset] =
                    row[column_offset] + tau_row * next[column_offset] - next_row * tau[column_offset];
            }
        }
        column += 2;
    }
    // Factor the product of the superdiagonal entries into (unit phase, sum of log-magnitudes).
    let mut phase = sign;
    let mut log_abs = 0.0_f64;
    let mut index = 0usize;
    while index < dimension {
        let entry = data[index * dimension + (index + 1)];
        let magnitude = entry.magnitude();
        if magnitude == 0.0 {
            return (T::zero(), f64::NEG_INFINITY);
        }
        phase = phase * entry.unit_phase();
        log_abs += magnitude.ln();
        index += 2;
    }
    (phase, log_abs)
}

/// Build one flat row-major buffer per ``(n, n)`` slice of a ``(batch, n, n)`` view (one copy each)
/// and return them together with the shared matrix dimension ``n``.
fn flat_matrices<T: numpy::Element + Copy>(matrix: &PyReadonlyArray3<'_, T>) -> (Vec<Vec<T>>, usize) {
    let view = matrix.as_array();
    let batch = view.shape()[0];
    let dimension = view.shape()[1];
    let flats = (0..batch)
        .map(|index| view.index_axis(Axis(0), index).iter().copied().collect())
        .collect();
    (flats, dimension)
}

/// Signed Pfaffian of each owned matrix, computed in parallel across the batch above a threshold.
fn signed_pfaffian_owned<T>(matrices: Vec<Array2<T>>) -> Vec<T>
where
    T: Copy
        + Zero
        + One
        + Magnitude
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Send
        + Sync,
{
    if matrices.len() >= PARALLEL_BATCH_THRESHOLD {
        matrices.into_par_iter().map(pfaffian_one).collect()
    } else {
        matrices.into_iter().map(pfaffian_one).collect()
    }
}

/// Copy each ``(n, n)`` slice of a ``(batch, n, n)`` view into an owned matrix.
fn owned_matrices<T: numpy::Element + Clone>(matrix: &PyReadonlyArray3<'_, T>) -> Vec<Array2<T>> {
    let view = matrix.as_array();
    let batch = view.shape()[0];
    (0..batch).map(|index| view.index_axis(Axis(0), index).to_owned()).collect()
}

/// Log-domain signed Pfaffian ``(phase, log|Pf|)`` of each flat matrix, parallel above a threshold.
fn signed_slog_flat<T>(flats: Vec<Vec<T>>, dimension: usize) -> Vec<(T, f64)>
where
    T: Copy
        + Zero
        + One
        + SlogScalar
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Send
        + Sync,
{
    if flats.len() >= PARALLEL_BATCH_THRESHOLD {
        flats.into_par_iter().map(|flat| slog_pfaffian_fast(flat, dimension)).collect()
    } else {
        flats.into_iter().map(|flat| slog_pfaffian_fast(flat, dimension)).collect()
    }
}

/// Signed Pfaffian of a batch of ``float64`` skew-symmetric matrices, shape ``(batch, n, n)``.
#[pyfunction]
fn signed_pfaffian_f64<'py>(py: Python<'py>, matrix: PyReadonlyArray3<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let matrices = owned_matrices(&matrix);
    let results = py.allow_threads(|| signed_pfaffian_owned(matrices));
    Array1::from(results).into_pyarray(py)
}

/// Signed Pfaffian of a batch of ``float32`` skew-symmetric matrices, shape ``(batch, n, n)``.
#[pyfunction]
fn signed_pfaffian_f32<'py>(py: Python<'py>, matrix: PyReadonlyArray3<'py, f32>) -> Bound<'py, PyArray1<f32>> {
    let matrices = owned_matrices(&matrix);
    let results = py.allow_threads(|| signed_pfaffian_owned(matrices));
    Array1::from(results).into_pyarray(py)
}

/// Signed Pfaffian of a batch of ``float16`` skew-symmetric matrices, shape ``(batch, n, n)``.
///
/// The elimination runs entirely in half precision, so every intermediate is rounded to ``float16``.
/// This is the least accurate kernel and can overflow the narrow ``float16`` range.
#[pyfunction]
fn signed_pfaffian_f16<'py>(py: Python<'py>, matrix: PyReadonlyArray3<'py, f16>) -> Bound<'py, PyArray1<f16>> {
    let matrices = owned_matrices(&matrix);
    let results = py.allow_threads(|| signed_pfaffian_owned(matrices));
    Array1::from(results).into_pyarray(py)
}

/// Signed Pfaffian of a batch of ``complex128`` skew-symmetric matrices, shape ``(batch, n, n)``.
#[pyfunction]
fn signed_pfaffian_c128<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, Complex64>,
) -> Bound<'py, PyArray1<Complex64>> {
    let matrices = owned_matrices(&matrix);
    let results = py.allow_threads(|| signed_pfaffian_owned(matrices));
    Array1::from(results).into_pyarray(py)
}

/// Signed Pfaffian of a batch of ``complex64`` skew-symmetric matrices, shape ``(batch, n, n)``.
#[pyfunction]
fn signed_pfaffian_c64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, Complex32>,
) -> Bound<'py, PyArray1<Complex32>> {
    let matrices = owned_matrices(&matrix);
    let results = py.allow_threads(|| signed_pfaffian_owned(matrices));
    Array1::from(results).into_pyarray(py)
}

/// Log-domain signed Pfaffian of a batch of ``float64`` matrices: returns ``(phase, log|Pf|)`` arrays.
#[pyfunction]
fn signed_slog_pfaffian_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (flats, dimension) = flat_matrices(&matrix);
    let results = py.allow_threads(|| signed_slog_flat(flats, dimension));
    let (phases, logs): (Vec<f64>, Vec<f64>) = results.into_iter().unzip();
    (Array1::from(phases).into_pyarray(py), Array1::from(logs).into_pyarray(py))
}

/// Log-domain signed Pfaffian of a batch of ``float32`` matrices: returns ``(phase, log|Pf|)`` arrays.
///
/// ``log|Pf|`` is accumulated in ``f64`` for accuracy and cast to ``float32`` at the boundary.
#[pyfunction]
fn signed_slog_pfaffian_f32<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, f32>,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
    let (flats, dimension) = flat_matrices(&matrix);
    let results = py.allow_threads(|| signed_slog_flat(flats, dimension));
    let (phases, logs): (Vec<f32>, Vec<f64>) = results.into_iter().unzip();
    let logs: Vec<f32> = logs.into_iter().map(|value| value as f32).collect();
    (Array1::from(phases).into_pyarray(py), Array1::from(logs).into_pyarray(py))
}

/// Log-domain signed Pfaffian of a batch of ``complex128`` matrices: returns ``(phase, log|Pf|)``.
///
/// ``phase`` is the unit-modulus complex phasor; ``log|Pf|`` is the real (``float64``) log-magnitude.
#[pyfunction]
fn signed_slog_pfaffian_c128<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, Complex64>,
) -> (Bound<'py, PyArray1<Complex64>>, Bound<'py, PyArray1<f64>>) {
    let (flats, dimension) = flat_matrices(&matrix);
    let results = py.allow_threads(|| signed_slog_flat(flats, dimension));
    let (phases, logs): (Vec<Complex64>, Vec<f64>) = results.into_iter().unzip();
    (Array1::from(phases).into_pyarray(py), Array1::from(logs).into_pyarray(py))
}

/// Log-domain signed Pfaffian of a batch of ``complex64`` matrices: returns ``(phase, log|Pf|)``.
///
/// ``phase`` is the unit-modulus complex phasor; ``log|Pf|`` is accumulated in ``f64`` and cast to
/// the real ``float32`` type at the boundary.
#[pyfunction]
fn signed_slog_pfaffian_c64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, Complex32>,
) -> (Bound<'py, PyArray1<Complex32>>, Bound<'py, PyArray1<f32>>) {
    let (flats, dimension) = flat_matrices(&matrix);
    let results = py.allow_threads(|| signed_slog_flat(flats, dimension));
    let (phases, logs): (Vec<Complex32>, Vec<f64>) = results.into_iter().unzip();
    let logs: Vec<f32> = logs.into_iter().map(|value| value as f32).collect();
    (Array1::from(phases).into_pyarray(py), Array1::from(logs).into_pyarray(py))
}

/// Machine epsilon of a scalar type's real precision, as ``torch.finfo(dtype).eps``.
trait Precision {
    const EPSILON: f64;
}

impl Precision for f32 {
    const EPSILON: f64 = f32::EPSILON as f64;
}

impl Precision for f64 {
    const EPSILON: f64 = f64::EPSILON;
}

impl Precision for Complex<f32> {
    const EPSILON: f64 = f32::EPSILON as f64;
}

impl Precision for Complex<f64> {
    const EPSILON: f64 = f64::EPSILON;
}

/// Exponents of ``eps`` in the singularity tests of [`flagged_inverse_from_flat`], as in
/// ``PfaffianStrategy.SINGULARITY_RTOL_EXPONENT`` and ``INVERSE_RESIDUAL_RTOL_EXPONENT``.
const SINGULARITY_RTOL_EXPONENT: f64 = 0.75;
const INVERSE_RESIDUAL_RTOL_EXPONENT: f64 = 0.5;

/// Inverse of one square matrix (flat row-major) with the flags of ``PfaffianStrategy._flagged_inverse``.
///
/// A Gauss-Jordan elimination with partial pivoting, whose pivots are those of the LU factorization ``U_ii``,
/// gives the inverse and the rank flag: a zero pivot, or a pivot ratio ``min|U_ii| / max|U_ii|`` at most
/// ``eps^0.75`` (NaN-safe: a NaN ratio is flagged). A rank-flagged matrix gets the identity as its inverse.
/// Otherwise the residual flag is set when ``max |A A^{-1} - I|`` exceeds ``eps^0.5`` (or is NaN). Every step
/// is a row update, which vectorizes. Returns ``(inverse, rank_flag, residual_flag)``.
#[inline(always)]
fn flagged_inverse_from_flat<T>(matrix: &[T], dimension: usize) -> (Vec<T>, bool, bool)
where
    T: Copy
        + Zero
        + One
        + Magnitude
        + Precision
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let size = dimension;
    let identity = || {
        let mut eye = vec![T::zero(); size * size];
        for index in 0..size {
            eye[index * size + index] = T::one();
        }
        eye
    };
    // Gauss-Jordan elimination with partial pivoting: the pivots are LU's (the same rows are chosen, the rows
    // above the pivot are eliminated as well), and every step is a row update, which vectorizes.
    let mut work = matrix.to_vec();
    let mut inverse = identity();
    let mut zero_pivot = false;
    let mut smallest = f64::INFINITY;
    let mut largest = 0.0_f64;
    for step in 0..size {
        let mut pivot_row = step;
        let mut best = work[step * size + step].magnitude();
        for row in (step + 1)..size {
            let candidate = work[row * size + step].magnitude();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if pivot_row != step {
            for column in 0..size {
                work.swap(step * size + column, pivot_row * size + column);
                inverse.swap(step * size + column, pivot_row * size + column);
            }
        }
        let pivot = work[step * size + step];
        let magnitude = pivot.magnitude();
        smallest = if magnitude.is_nan() || smallest.is_nan() { f64::NAN } else { smallest.min(magnitude) };
        largest = largest.max(magnitude);
        if magnitude == 0.0 {
            zero_pivot = true; // LAPACK's info > 0: the matrix is flagged, its inverse is the identity
            break;
        }
        let scale = T::one() / pivot;
        for column in step..size {
            work[step * size + column] = work[step * size + column] * scale;
        }
        for column in 0..size {
            inverse[step * size + column] = inverse[step * size + column] * scale;
        }
        let pivot_work: Vec<T> = work[step * size + step..(step + 1) * size].to_vec();
        let pivot_inverse: Vec<T> = inverse[step * size..(step + 1) * size].to_vec();
        for row in 0..size {
            if row == step {
                continue;
            }
            let factor = work[row * size + step];
            if factor.magnitude() == 0.0 {
                continue;
            }
            let work_row = &mut work[row * size + step..(row + 1) * size];
            for (target, source) in work_row.iter_mut().zip(pivot_work.iter()) {
                *target = *target - factor * *source;
            }
            let inverse_row = &mut inverse[row * size..(row + 1) * size];
            for (target, source) in inverse_row.iter_mut().zip(pivot_inverse.iter()) {
                *target = *target - factor * *source;
            }
        }
    }
    let threshold = T::EPSILON.powf(SINGULARITY_RTOL_EXPONENT) * largest;
    // A NaN ratio or threshold is flagged, as PyTorch's ``~(min > threshold)``.
    if zero_pivot || smallest.is_nan() || threshold.is_nan() || smallest <= threshold {
        return (identity(), true, false);
    }
    // Residual max |A A^{-1} - I|, row by row: (A A^{-1})[i, :] = sum_k A[i, k] A^{-1}[k, :], a row update per k.
    let mut residual = 0.0_f64;
    let mut product = vec![T::zero(); size];
    for row in 0..size {
        product.iter_mut().for_each(|value| *value = T::zero());
        for inner in 0..size {
            let weight = matrix[row * size + inner];
            for (target, source) in product.iter_mut().zip(inverse[inner * size..(inner + 1) * size].iter()) {
                *target = *target + weight * *source;
            }
        }
        product[row] = product[row] - T::one();
        for value in product.iter() {
            let magnitude = value.magnitude();
            residual = if magnitude.is_nan() || residual.is_nan() { f64::NAN } else { residual.max(magnitude) };
        }
    }
    let residual_flag = residual.is_nan() || residual > T::EPSILON.powf(INVERSE_RESIDUAL_RTOL_EXPONENT);
    (inverse, false, residual_flag)
}

/// ``(inverse, rank_flag, residual_flag)`` of a batch, as numpy arrays.
type FlaggedArrays<'py, T> = (Bound<'py, PyArray3<T>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>);

/// The arithmetic of [`flagged_inverse_from_flat`].
trait InverseScalar: PfaffianScalar + Precision {}

impl<T: PfaffianScalar + Precision> InverseScalar for T {}

with_avx2!(
    fn flagged_inverse_fast<T: InverseScalar>(matrix: &[T], dimension: usize) -> (Vec<T>, bool, bool)
        => flagged_inverse_from_flat
);

/// [`flagged_inverse_from_flat`] over a ``(batch, n, n)`` view, parallel above a threshold, as numpy arrays.
fn flagged_inverse_batch<'py, T>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, T>,
) -> FlaggedArrays<'py, T>
where
    T: numpy::Element
        + Copy
        + Zero
        + One
        + Magnitude
        + Precision
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Send
        + Sync,
{
    let (flats, dimension) = flat_matrices(&matrix);
    let batch = flats.len();
    let results: Vec<(Vec<T>, bool, bool)> = py.allow_threads(|| {
        if batch >= PARALLEL_BATCH_THRESHOLD {
            flats.par_iter().map(|flat| flagged_inverse_fast(flat, dimension)).collect()
        } else {
            flats.iter().map(|flat| flagged_inverse_fast(flat, dimension)).collect()
        }
    });
    let mut inverses = Vec::with_capacity(batch * dimension * dimension);
    let mut rank_flags = Vec::with_capacity(batch);
    let mut residual_flags = Vec::with_capacity(batch);
    for (inverse, rank_flag, residual_flag) in results {
        inverses.extend(inverse);
        rank_flags.push(rank_flag);
        residual_flags.push(residual_flag);
    }
    let inverses = Array3::from_shape_vec((batch, dimension, dimension), inverses).expect("shape matches");
    (inverses.into_pyarray(py), Array1::from(rank_flags).into_pyarray(py), Array1::from(residual_flags).into_pyarray(py))
}

/// ``(pfaffian, inverse, rank_flag, residual_flag)`` of a batch, as numpy arrays.
type PfaffianAndInverseArrays<'py, T> =
    (Bound<'py, PyArray1<T>>, Bound<'py, PyArray3<T>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>);

/// The signed Pfaffian and the flagged inverse of each matrix of a ``(batch, n, n)`` view in one call, for the
/// forward of a graph (``RustPfaffianWithInverse``), parallel above a threshold.
fn pfaffian_and_inverse_batch<'py, T>(py: Python<'py>, matrix: PyReadonlyArray3<'py, T>) -> PfaffianAndInverseArrays<'py, T>
where
    T: numpy::Element + InverseScalar + Send + Sync,
{
    let (flats, dimension) = flat_matrices(&matrix);
    let batch = flats.len();
    let array = new_inverses::<T>(py, batch, dimension);
    // SAFETY: the new array is C-contiguous and not shared; every element is written below.
    let output = unsafe { array.as_slice_mut() }.expect("C-contiguous output");
    let results: Vec<(T, bool, bool)> = py.allow_threads(|| {
        batch_map_into(output, dimension * dimension, |item, chunk| {
            let pfaffian = pfaffian_fast(flats[item].clone(), dimension);
            let (inverse, rank_flag, residual_flag) = flagged_inverse_fast(&flats[item], dimension);
            chunk.copy_from_slice(&inverse);
            (pfaffian, rank_flag, residual_flag)
        })
    });
    let mut pfaffians = Vec::with_capacity(batch);
    let mut rank_flags = Vec::with_capacity(batch);
    let mut residual_flags = Vec::with_capacity(batch);
    for (pfaffian, rank_flag, residual_flag) in results {
        pfaffians.push(pfaffian);
        rank_flags.push(rank_flag);
        residual_flags.push(residual_flag);
    }
    (
        Array1::from(pfaffians).into_pyarray(py),
        array,
        Array1::from(rank_flags).into_pyarray(py),
        Array1::from(residual_flags).into_pyarray(py),
    )
}

macro_rules! pfaffian_and_inverse_function {
    ($scalar:ty, $name:ident) => {
        /// Signed Pfaffians, inverses and singularity flags of a batch: ``(pfaffian, inverse, rank, residual)``.
        #[pyfunction]
        fn $name<'py>(py: Python<'py>, matrix: PyReadonlyArray3<'py, $scalar>) -> PfaffianAndInverseArrays<'py, $scalar> {
            pfaffian_and_inverse_batch(py, matrix)
        }
    };
}

pfaffian_and_inverse_function!(f64, signed_pfaffian_and_inverse_f64);
pfaffian_and_inverse_function!(f32, signed_pfaffian_and_inverse_f32);
pfaffian_and_inverse_function!(Complex64, signed_pfaffian_and_inverse_c128);
pfaffian_and_inverse_function!(Complex32, signed_pfaffian_and_inverse_c64);

/// Inverse and singularity flags of a batch of ``float64`` matrices: ``(inverse, rank_flag, residual_flag)``.
#[pyfunction]
fn flagged_inverse_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, f64>,
) -> FlaggedArrays<'py, f64> {
    flagged_inverse_batch(py, matrix)
}

/// Inverse and singularity flags of a batch of ``float32`` matrices: ``(inverse, rank_flag, residual_flag)``.
#[pyfunction]
fn flagged_inverse_f32<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, f32>,
) -> FlaggedArrays<'py, f32> {
    flagged_inverse_batch(py, matrix)
}

/// Inverse and singularity flags of a batch of ``complex128`` matrices.
#[pyfunction]
fn flagged_inverse_c128<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, Complex64>,
) -> FlaggedArrays<'py, Complex64> {
    flagged_inverse_batch(py, matrix)
}

/// Inverse and singularity flags of a batch of ``complex64`` matrices.
#[pyfunction]
fn flagged_inverse_c64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'py, Complex32>,
) -> FlaggedArrays<'py, Complex32> {
    flagged_inverse_batch(py, matrix)
}

/// Arithmetic of the low-rank kernels, shared by the four precisions.
trait LowRankScalar:
    Copy
    + Zero
    + One
    + Magnitude
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Send
    + Sync
{
    fn from_f64(value: f64) -> Self;
}

impl LowRankScalar for f32 {
    fn from_f64(value: f64) -> f32 {
        value as f32
    }
}

impl LowRankScalar for f64 {
    fn from_f64(value: f64) -> f64 {
        value
    }
}

impl LowRankScalar for Complex<f32> {
    fn from_f64(value: f64) -> Complex<f32> {
        Complex::new(value as f32, 0.0)
    }
}

impl LowRankScalar for Complex<f64> {
    fn from_f64(value: f64) -> Complex<f64> {
        Complex::new(value, 0.0)
    }
}

/// Lanes of [`dot`]: independent partial sums the compiler can keep in one SIMD register.
const DOT_LANES: usize = 8;

/// ``sum_i a[i] b[i]`` with [`DOT_LANES`] independent partial sums, so the loop vectorizes without reordering
/// any one sum (a single accumulator serializes on the latency of the additions).
#[inline(always)]
fn dot<T: LowRankScalar>(left: &[T], right: &[T]) -> T {
    let length = left.len().min(right.len());
    let chunks = length / DOT_LANES;
    let mut lanes = [T::zero(); DOT_LANES];
    for chunk in 0..chunks {
        let start = chunk * DOT_LANES;
        let (left_chunk, right_chunk) = (&left[start..start + DOT_LANES], &right[start..start + DOT_LANES]);
        for lane in 0..DOT_LANES {
            lanes[lane] = lanes[lane] + left_chunk[lane] * right_chunk[lane];
        }
    }
    let mut sum = T::zero();
    for lane in lanes {
        sum = sum + lane;
    }
    for index in chunks * DOT_LANES..length {
        sum = sum + left[index] * right[index];
    }
    sum
}

/// The columns of a row-major ``(n, width)`` matrix, as ``width`` contiguous rows of length ``n``.
#[inline(always)]
fn columns<T: LowRankScalar>(vectors: &[T], size: usize, width: usize) -> Vec<T> {
    let mut result = vec![T::zero(); width * size];
    for row in 0..size {
        for column in 0..width {
            result[column * size + row] = vectors[row * width + column];
        }
    }
    result
}

/// Square matrix stored row-major, or column-major (``transposed``: the buffer holds its transpose row-major).
#[derive(Clone, Copy)]
struct Matrix<'a, T> {
    data: &'a [T],
    size: usize,
    transposed: bool,
}

impl<T: LowRankScalar> Matrix<'_, T> {
    /// Row ``index`` as an owned vector.
    #[inline(always)]
    fn row(&self, index: usize) -> Vec<T> {
        let size = self.size;
        if self.transposed {
            (0..size).map(|column| self.data[column * size + index]).collect()
        } else {
            self.data[index * size..(index + 1) * size].to_vec()
        }
    }

    /// ``vectors^T M`` for ``vectors`` of shape ``(n, width)`` row-major, as ``width`` rows of length ``n``.
    #[inline(always)]
    fn left_product(&self, vectors: &[T], width: usize) -> Vec<T> {
        let size = self.size;
        let mut rows = vec![T::zero(); width * size];
        if self.transposed {
            // (v^T M)[c][j] = sum_i v[i][c] M[i][j], and M[i][j] is data[j][i]: a dot product per row of data.
            let vector_rows = columns(vectors, size, width);
            for column in 0..size {
                let data_row = &self.data[column * size..(column + 1) * size];
                for vector in 0..width {
                    rows[vector * size + column] = dot(&vector_rows[vector * size..(vector + 1) * size], data_row);
                }
            }
        } else {
            for inner in 0..size {
                let data_row = &self.data[inner * size..(inner + 1) * size];
                for vector in 0..width {
                    let weight = vectors[inner * width + vector];
                    let row = &mut rows[vector * size..(vector + 1) * size];
                    for index in 0..size {
                        row[index] = row[index] + weight * data_row[index];
                    }
                }
            }
        }
        rows
    }
}

/// ``R = J + rows u`` (``2k x 2k``) for ``J = [[0, I_k], [-I_k, 0]]``, made exactly skew-symmetric from its upper
/// triangle when ``skew`` (as ``LowRankPfaffian._skew``), else left as computed.
#[inline(always)]
fn r_matrix<T: LowRankScalar>(rows: &[T], update: &[T], size: usize, width: usize, skew: bool) -> Vec<T> {
    let half = width / 2;
    let update_columns = columns(update, size, width);
    let mut matrix = vec![T::zero(); width * width];
    for row in 0..width {
        for column in 0..width {
            matrix[row * width + column] =
                dot(&rows[row * size..(row + 1) * size], &update_columns[column * size..(column + 1) * size]);
        }
    }
    if skew {
        let half_value = T::from_f64(0.5);
        for row in 0..width {
            for column in (row + 1)..width {
                let value = (matrix[row * width + column] - matrix[column * width + row]) * half_value;
                matrix[row * width + column] = value;
                matrix[column * width + row] = -value;
            }
            matrix[row * width + row] = T::zero();
        }
    }
    for index in 0..half {
        matrix[index * width + half + index] = matrix[index * width + half + index] + T::one();
        matrix[(half + index) * width + index] = matrix[(half + index) * width + index] - T::one();
    }
    matrix
}

/// ``Pf(R) / Pf(J)`` of an ``R`` that is skew-symmetric up to round-off, from its upper triangle.
#[inline(always)]
fn r_ratio<T: LowRankScalar>(matrix: &[T], width: usize) -> T {
    let mut skew = matrix.to_vec();
    for row in 0..width {
        skew[row * width + row] = T::zero();
        for column in (row + 1)..width {
            skew[column * width + row] = -matrix[row * width + column];
        }
    }
    let half = width / 2;
    let ratio = pfaffian_from_flat(skew, width);
    if (half * (half.saturating_sub(1)) / 2) % 2 == 1 {
        -ratio
    } else {
        ratio
    }
}

/// ``R^{-1} rows`` (``2k x n``) by Gaussian elimination with partial pivoting on ``R`` (``2k x 2k``).
#[inline(always)]
fn solve_small<T: LowRankScalar>(matrix: &[T], rows: &[T], width: usize, size: usize) -> Vec<T> {
    let mut factors = matrix.to_vec();
    let mut solution = rows.to_vec();
    for step in 0..width {
        let mut pivot_row = step;
        let mut best = factors[step * width + step].magnitude();
        for row in (step + 1)..width {
            let candidate = factors[row * width + step].magnitude();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if pivot_row != step {
            for column in 0..width {
                factors.swap(step * width + column, pivot_row * width + column);
            }
            for column in 0..size {
                solution.swap(step * size + column, pivot_row * size + column);
            }
        }
        let pivot = factors[step * width + step];
        for row in (step + 1)..width {
            let factor = factors[row * width + step] / pivot;
            for column in step..width {
                factors[row * width + column] = factors[row * width + column] - factor * factors[step * width + column];
            }
            for column in 0..size {
                solution[row * size + column] = solution[row * size + column] - factor * solution[step * size + column];
            }
        }
    }
    for step in (0..width).rev() {
        let pivot = factors[step * width + step];
        for column in 0..size {
            let mut value = solution[step * size + column];
            for inner in (step + 1)..width {
                value = value - factors[step * width + inner] * solution[inner * size + column];
            }
            solution[step * size + column] = value / pivot;
        }
    }
    solution
}

/// Writes ``A^{-1} + left^T right`` (``left`` and ``right`` of shape ``(p, n)``) into ``output`` in the layout of
/// ``inverse``: a transposed (column-major) one gets ``(A^{-1})^T + right^T left``, the transpose, so the buffers
/// are read and written in order and the caller keeps its layout. One read of ``A^{-1}`` and one write.
#[inline(always)]
fn add_outer<T: LowRankScalar>(inverse: Matrix<'_, T>, left: &[T], right: &[T], count: usize, output: &mut [T]) {
    let size = inverse.size;
    if count == 2 {
        // The rank-2 corrections of every update but the wide ones: one fused pass from the source to the output.
        let (left, right) = if inverse.transposed { (right, left) } else { (left, right) };
        let (right_first, right_second) = (&right[..size], &right[size..2 * size]);
        for row in 0..size {
            let (first, second) = (left[row], left[size + row]);
            let source = &inverse.data[row * size..(row + 1) * size];
            let target = &mut output[row * size..(row + 1) * size];
            for column in 0..size {
                target[column] = source[column] + first * right_first[column] + second * right_second[column];
            }
        }
        return;
    }
    output.copy_from_slice(inverse.data);
    add_outer_in_place(output, size, inverse.transposed, left, right, count);
}

/// Adds ``left^T right`` to the matrix held in ``data`` (its transpose when ``transposed``, which gets
/// ``right^T left``), in place: one read and one write of the matrix, row by row.
#[inline(always)]
fn add_outer_in_place<T: LowRankScalar>(
    data: &mut [T],
    size: usize,
    transposed: bool,
    left: &[T],
    right: &[T],
    count: usize,
) {
    let (left, right) = if transposed { (right, left) } else { (left, right) };
    if count == 2 {
        // The rank-2 corrections of every update but the wide ones: one fused pass per row.
        let (right_first, right_second) = (&right[..size], &right[size..2 * size]);
        for row in 0..size {
            let (first, second) = (left[row], left[size + row]);
            let target = &mut data[row * size..(row + 1) * size];
            for column in 0..size {
                target[column] = target[column] + first * right_first[column] + second * right_second[column];
            }
        }
        return;
    }
    for row in 0..size {
        let target = &mut data[row * size..(row + 1) * size];
        for term in 0..count {
            let weight = left[term * size + row];
            let right_row = &right[term * size..(term + 1) * size];
            for column in 0..size {
                target[column] = target[column] + weight * right_row[column];
            }
        }
    }
}

/// Ratio ``Pf(A - u J u^T) / Pf(A)`` of one matrix, as ``LowRankPfaffian._ratio_core``.
#[inline(always)]
fn low_rank_ratio_one<T: LowRankScalar>(inverse: Matrix<'_, T>, update: &[T], width: usize) -> T {
    let size = inverse.size;
    if width == 2 {
        // R_01 = 1 + u_0^T A^{-1} u_1: only the first row of u^T A^{-1} is needed.
        let first = inverse.left_product(&columns(update, size, width)[..size], 1);
        let second: Vec<T> = (0..size).map(|index| update[index * width + 1]).collect();
        return T::one() + dot(&first, &second);
    }
    let rows = inverse.left_product(update, width);
    r_ratio(&r_matrix(&rows, update, size, width, false), width)
}

/// Ratio and updated inverse of one matrix, as ``LowRankPfaffian._ratio_and_inverse_core``.
#[inline(always)]
fn low_rank_ratio_and_inverse_one<T: LowRankScalar>(
    inverse: Matrix<'_, T>,
    update: &[T],
    width: usize,
    output: &mut [T],
) -> T {
    let (ratio, left, right) = low_rank_factors(inverse, update, width);
    add_outer(inverse, &left, &right, width, output);
    ratio
}

/// The ratio and the factors ``(left, right)`` of the inverse update ``A^{-1} + left^T right`` of a low-rank update.
#[inline(always)]
fn low_rank_factors<T: LowRankScalar>(inverse: Matrix<'_, T>, update: &[T], width: usize) -> (T, Vec<T>, Vec<T>) {
    let size = inverse.size;
    let rows = inverse.left_product(update, width);
    if width == 2 {
        let second: Vec<T> = (0..size).map(|index| update[index * width + 1]).collect();
        let ratio = T::one() + dot(&rows[..size], &second);
        // rows^T R^{-1} rows = (rows_1^T rows_0 - rows_0^T rows_1) / r
        let left: Vec<T> = rows[size..2 * size].iter().chain(rows[..size].iter()).copied().collect();
        let right: Vec<T> = rows[..size]
            .iter()
            .map(|value| *value / ratio)
            .chain(rows[size..2 * size].iter().map(|value| -*value / ratio))
            .collect();
        return (ratio, left, right);
    }
    let matrix = r_matrix(&rows, update, size, width, true);
    let ratio = r_ratio(&matrix, width);
    let solved = solve_small(&matrix, &rows, width, size);
    (ratio, rows, solved)
}

/// [`low_rank_ratio_and_inverse_one`] on the matrix held in ``data``, updated in place.
#[inline(always)]
fn low_rank_ratio_and_inverse_in_place<T: LowRankScalar>(
    data: &mut [T],
    size: usize,
    transposed: bool,
    update: &[T],
    width: usize,
) -> T {
    let (ratio, left, right) = low_rank_factors(Matrix { data: &*data, size, transposed }, update, width);
    add_outer_in_place(data, size, transposed, &left, &right, width);
    ratio
}

/// ``column = A^{-1} e_i`` (as minus row ``i``, as the PyTorch core) and the ratio ``1 + c^T A^{-1} e_i``.
#[inline(always)]
fn replace_column<T: LowRankScalar>(inverse: Matrix<'_, T>, index: usize, change: &[T]) -> (Vec<T>, T) {
    let column: Vec<T> = inverse.row(index).into_iter().map(|value| -value).collect();
    let ratio = T::one() + dot(change, &column);
    (column, ratio)
}

/// Ratio and updated inverse of a row and column replacement, as ``_ratio_replace_and_inverse_core``.
#[inline(always)]
fn replace_ratio_and_inverse_one<T: LowRankScalar>(
    inverse: Matrix<'_, T>,
    index: usize,
    change: &[T],
    output: &mut [T],
) -> T {
    let (ratio, left, right) = replace_factors(inverse, index, change);
    add_outer(inverse, &left, &right, 2, output);
    ratio
}

/// The ratio and the factors ``(left, right)`` of the inverse update of a row and column replacement.
#[inline(always)]
fn replace_factors<T: LowRankScalar>(inverse: Matrix<'_, T>, index: usize, change: &[T]) -> (T, Vec<T>, Vec<T>) {
    let (column, ratio) = replace_column(inverse, index, change);
    // row = c^T A^{-1}; A'^{-1} = A^{-1} + (col (-row) + row^T col^T) / r
    let row = inverse.left_product(change, 1);
    let left: Vec<T> = column.iter().chain(row.iter()).copied().collect();
    let right: Vec<T> =
        row.iter().map(|value| -*value / ratio).chain(column.iter().map(|value| *value / ratio)).collect();
    (ratio, left, right)
}

/// [`replace_ratio_and_inverse_one`] on the matrix held in ``data``, updated in place.
#[inline(always)]
fn replace_ratio_and_inverse_in_place<T: LowRankScalar>(
    data: &mut [T],
    size: usize,
    transposed: bool,
    index: usize,
    change: &[T],
) -> T {
    let (ratio, left, right) = replace_factors(Matrix { data: &*data, size, transposed }, index, change);
    add_outer_in_place(data, size, transposed, &left, &right, 2);
    ratio
}

with_avx2!(fn low_rank_ratio_fast<T: LowRankScalar>(inverse: Matrix<'_, T>, update: &[T], width: usize) -> T => low_rank_ratio_one);
with_avx2!(
    fn low_rank_ratio_and_inverse_fast<T: LowRankScalar>(
        inverse: Matrix<'_, T>,
        update: &[T],
        width: usize,
        output: &mut [T],
    ) -> T => low_rank_ratio_and_inverse_one
);
with_avx2!(fn replace_ratio_fast<T: LowRankScalar>(inverse: Matrix<'_, T>, index: usize, change: &[T]) -> T => replace_ratio_one);
with_avx2!(
    fn replace_ratio_and_inverse_fast<T: LowRankScalar>(
        inverse: Matrix<'_, T>,
        index: usize,
        change: &[T],
        output: &mut [T],
    ) -> T => replace_ratio_and_inverse_one
);

with_avx2!(
    fn low_rank_ratio_and_inverse_in_place_fast<T: LowRankScalar>(
        data: &mut [T],
        size: usize,
        transposed: bool,
        update: &[T],
        width: usize,
    ) -> T => low_rank_ratio_and_inverse_in_place
);
with_avx2!(
    fn replace_ratio_and_inverse_in_place_fast<T: LowRankScalar>(
        data: &mut [T],
        size: usize,
        transposed: bool,
        index: usize,
        change: &[T],
    ) -> T => replace_ratio_and_inverse_in_place
);

/// Ratio ``1 + c^T A^{-1} e_i`` of one row and column replacement, as ``_ratio_replace_core``.
#[inline(always)]
fn replace_ratio_one<T: LowRankScalar>(inverse: Matrix<'_, T>, index: usize, change: &[T]) -> T {
    replace_column(inverse, index, change).1
}

/// Runs ``work`` on every chunk of ``output`` (one per batch element) with its index, in parallel above a
/// threshold, and returns the results.
fn batch_map_into<T: Send, R: Send, F: Fn(usize, &mut [T]) -> R + Sync + Send>(
    output: &mut [T],
    chunk: usize,
    work: F,
) -> Vec<R> {
    if output.len() / chunk.max(1) >= PARALLEL_BATCH_THRESHOLD {
        output.par_chunks_mut(chunk).enumerate().map(|(item, part)| work(item, part)).collect()
    } else {
        output.chunks_mut(chunk).enumerate().map(|(item, part)| work(item, part)).collect()
    }
}

/// Runs ``work`` on every index of a batch, in parallel above a threshold.
fn batch_map<R: Send, F: Fn(usize) -> R + Sync + Send>(batch: usize, work: F) -> Vec<R> {
    if batch >= PARALLEL_BATCH_THRESHOLD {
        (0..batch).into_par_iter().map(work).collect()
    } else {
        (0..batch).map(work).collect()
    }
}

/// Batched ratios of low-rank updates: ``inverse (b, n, n)`` (its transpose when ``transposed``),
/// ``update (b, n, 2k)``.
fn low_rank_ratio_batch<'py, T: LowRankScalar + numpy::Element>(
    py: Python<'py>,
    inverse: PyReadonlyArray3<'py, T>,
    transposed: bool,
    update: PyReadonlyArray3<'py, T>,
) -> Bound<'py, PyArray1<T>> {
    let (inverse, update) = (inverse.as_array(), update.as_array());
    let (batch, size, width) = (update.shape()[0], update.shape()[1], update.shape()[2]);
    let inverse = inverse.as_slice().expect("C-contiguous inverse");
    let update = update.as_slice().expect("C-contiguous update");
    let ratios = py.allow_threads(|| {
        batch_map(batch, |item| {
            low_rank_ratio_fast(
                Matrix { data: &inverse[item * size * size..(item + 1) * size * size], size, transposed },
                &update[item * size * width..(item + 1) * size * width],
                width,
            )
        })
    });
    Array1::from(ratios).into_pyarray(py)
}

/// Batched ratios and updated inverses of low-rank updates (see [`low_rank_ratio_batch`]).
fn low_rank_ratio_and_inverse_batch<'py, T: LowRankScalar + numpy::Element>(
    py: Python<'py>,
    inverse: PyReadonlyArray3<'py, T>,
    transposed: bool,
    update: PyReadonlyArray3<'py, T>,
) -> (Bound<'py, PyArray1<T>>, Bound<'py, PyArray3<T>>) {
    let (inverse, update) = (inverse.as_array(), update.as_array());
    let (batch, size, width) = (update.shape()[0], update.shape()[1], update.shape()[2]);
    let inverse = inverse.as_slice().expect("C-contiguous inverse");
    let update = update.as_slice().expect("C-contiguous update");
    let array = new_inverses::<T>(py, batch, size);
    // SAFETY: the new array is C-contiguous and not shared; the kernel writes every element.
    let output = unsafe { array.as_slice_mut() }.expect("C-contiguous output");
    let ratios = py.allow_threads(|| {
        batch_map_into(output, size * size, |item, chunk| {
            low_rank_ratio_and_inverse_fast(
                Matrix { data: &inverse[item * size * size..(item + 1) * size * size], size, transposed },
                &update[item * size * width..(item + 1) * size * width],
                width,
                chunk,
            )
        })
    });
    (Array1::from(ratios).into_pyarray(py), array)
}

/// A new ``(batch, size, size)`` numpy array, uninitialized, for kernels that write every element.
///
/// NumPy's allocator asks the kernel for transparent huge pages on large arrays, which Rust's does not: a batch of
/// 64 matrices of size 512 (134 MB) is mapped with 65 page faults instead of 32769, about 70 ms less per call.
fn new_inverses<'py, T: numpy::Element>(py: Python<'py>, batch: usize, size: usize) -> Bound<'py, PyArray3<T>> {
    // SAFETY: the callers write every element before the array reaches Python.
    unsafe { PyArray3::<T>::new(py, [batch, size, size], false) }
}

/// Batched ratios of row and column replacements: ``inverse (b, n, n)``, ``index (b,)``, ``change (b, n)``.
fn replace_ratio_batch<'py, T: LowRankScalar + numpy::Element>(
    py: Python<'py>,
    inverse: PyReadonlyArray3<'py, T>,
    transposed: bool,
    index: PyReadonlyArray1<'py, i64>,
    change: PyReadonlyArray2<'py, T>,
) -> Bound<'py, PyArray1<T>> {
    let (inverse, index, change) = (inverse.as_array(), index.as_array(), change.as_array());
    let (batch, size) = (change.shape()[0], change.shape()[1]);
    let inverse = inverse.as_slice().expect("C-contiguous inverse");
    let index = index.as_slice().expect("contiguous index");
    let change = change.as_slice().expect("C-contiguous change");
    let ratios = py.allow_threads(|| {
        batch_map(batch, |item| {
            replace_ratio_fast(
                Matrix { data: &inverse[item * size * size..(item + 1) * size * size], size, transposed },
                index[item] as usize,
                &change[item * size..(item + 1) * size],
            )
        })
    });
    Array1::from(ratios).into_pyarray(py)
}

/// Batched ratios and updated inverses of row and column replacements (see [`replace_ratio_batch`]).
fn replace_ratio_and_inverse_batch<'py, T: LowRankScalar + numpy::Element>(
    py: Python<'py>,
    inverse: PyReadonlyArray3<'py, T>,
    transposed: bool,
    index: PyReadonlyArray1<'py, i64>,
    change: PyReadonlyArray2<'py, T>,
) -> (Bound<'py, PyArray1<T>>, Bound<'py, PyArray3<T>>) {
    let (inverse, index, change) = (inverse.as_array(), index.as_array(), change.as_array());
    let (batch, size) = (change.shape()[0], change.shape()[1]);
    let inverse = inverse.as_slice().expect("C-contiguous inverse");
    let index = index.as_slice().expect("contiguous index");
    let change = change.as_slice().expect("C-contiguous change");
    let array = new_inverses::<T>(py, batch, size);
    // SAFETY: the new array is C-contiguous and not shared; the kernel writes every element.
    let output = unsafe { array.as_slice_mut() }.expect("C-contiguous output");
    let ratios = py.allow_threads(|| {
        batch_map_into(output, size * size, |item, chunk| {
            replace_ratio_and_inverse_fast(
                Matrix { data: &inverse[item * size * size..(item + 1) * size * size], size, transposed },
                index[item] as usize,
                &change[item * size..(item + 1) * size],
                chunk,
            )
        })
    });
    (Array1::from(ratios).into_pyarray(py), array)
}

/// Batched ratios of low-rank updates, with the inverses (``(b, n, n)``, their transposes when ``transposed``)
/// updated in place.
fn low_rank_ratio_and_inverse_in_place_batch<'py, T: LowRankScalar + numpy::Element>(
    py: Python<'py>,
    mut inverse: PyReadwriteArray3<'py, T>,
    transposed: bool,
    update: PyReadonlyArray3<'py, T>,
) -> Bound<'py, PyArray1<T>> {
    let update = update.as_array();
    let (size, width) = (update.shape()[1], update.shape()[2]);
    let update = update.as_slice().expect("C-contiguous update");
    let mut inverse = inverse.as_array_mut();
    let data = inverse.as_slice_mut().expect("C-contiguous inverse");
    let ratios = py.allow_threads(|| {
        batch_map_into(data, size * size, |item, chunk| {
            low_rank_ratio_and_inverse_in_place_fast(
                chunk,
                size,
                transposed,
                &update[item * size * width..(item + 1) * size * width],
                width,
            )
        })
    });
    Array1::from(ratios).into_pyarray(py)
}

/// Batched ratios of row and column replacements, with the inverses updated in place.
fn replace_ratio_and_inverse_in_place_batch<'py, T: LowRankScalar + numpy::Element>(
    py: Python<'py>,
    mut inverse: PyReadwriteArray3<'py, T>,
    transposed: bool,
    index: PyReadonlyArray1<'py, i64>,
    change: PyReadonlyArray2<'py, T>,
) -> Bound<'py, PyArray1<T>> {
    let (index, change) = (index.as_array(), change.as_array());
    let size = change.shape()[1];
    let index = index.as_slice().expect("contiguous index");
    let change = change.as_slice().expect("C-contiguous change");
    let mut inverse = inverse.as_array_mut();
    let data = inverse.as_slice_mut().expect("C-contiguous inverse");
    let ratios = py.allow_threads(|| {
        batch_map_into(data, size * size, |item, chunk| {
            replace_ratio_and_inverse_in_place_fast(
                chunk,
                size,
                transposed,
                index[item] as usize,
                &change[item * size..(item + 1) * size],
            )
        })
    });
    Array1::from(ratios).into_pyarray(py)
}

macro_rules! low_rank_functions {
    (
        $scalar:ty, $ratio:ident, $ratio_and_inverse:ident, $replace:ident, $replace_and_inverse:ident,
        $ratio_in_place:ident, $replace_in_place:ident
    ) => {
        /// Ratios ``Pf(A - u J u^T) / Pf(A)`` of a batch (``LowRankPfaffian.ratio``).
        #[pyfunction]
        fn $ratio<'py>(
            py: Python<'py>,
            inverse: PyReadonlyArray3<'py, $scalar>,
            transposed: bool,
            update: PyReadonlyArray3<'py, $scalar>,
        ) -> Bound<'py, PyArray1<$scalar>> {
            low_rank_ratio_batch(py, inverse, transposed, update)
        }

        /// Ratios and updated inverses of a batch (``LowRankPfaffian.ratio_and_inverse``).
        #[pyfunction]
        fn $ratio_and_inverse<'py>(
            py: Python<'py>,
            inverse: PyReadonlyArray3<'py, $scalar>,
            transposed: bool,
            update: PyReadonlyArray3<'py, $scalar>,
        ) -> (Bound<'py, PyArray1<$scalar>>, Bound<'py, PyArray3<$scalar>>) {
            low_rank_ratio_and_inverse_batch(py, inverse, transposed, update)
        }

        /// Ratios of row and column replacements of a batch (``LowRankPfaffian.ratio_replace``).
        #[pyfunction]
        fn $replace<'py>(
            py: Python<'py>,
            inverse: PyReadonlyArray3<'py, $scalar>,
            transposed: bool,
            index: PyReadonlyArray1<'py, i64>,
            change: PyReadonlyArray2<'py, $scalar>,
        ) -> Bound<'py, PyArray1<$scalar>> {
            replace_ratio_batch(py, inverse, transposed, index, change)
        }

        /// Ratios and updated inverses of row and column replacements (``ratio_replace_and_inverse``).
        #[pyfunction]
        fn $replace_and_inverse<'py>(
            py: Python<'py>,
            inverse: PyReadonlyArray3<'py, $scalar>,
            transposed: bool,
            index: PyReadonlyArray1<'py, i64>,
            change: PyReadonlyArray2<'py, $scalar>,
        ) -> (Bound<'py, PyArray1<$scalar>>, Bound<'py, PyArray3<$scalar>>) {
            replace_ratio_and_inverse_batch(py, inverse, transposed, index, change)
        }

        /// Ratios of low-rank updates of a batch, with the inverses updated in place (``PfaffianUpdater``).
        #[pyfunction]
        fn $ratio_in_place<'py>(
            py: Python<'py>,
            inverse: PyReadwriteArray3<'py, $scalar>,
            transposed: bool,
            update: PyReadonlyArray3<'py, $scalar>,
        ) -> Bound<'py, PyArray1<$scalar>> {
            low_rank_ratio_and_inverse_in_place_batch(py, inverse, transposed, update)
        }

        /// Ratios of row and column replacements of a batch, with the inverses updated in place.
        #[pyfunction]
        fn $replace_in_place<'py>(
            py: Python<'py>,
            inverse: PyReadwriteArray3<'py, $scalar>,
            transposed: bool,
            index: PyReadonlyArray1<'py, i64>,
            change: PyReadonlyArray2<'py, $scalar>,
        ) -> Bound<'py, PyArray1<$scalar>> {
            replace_ratio_and_inverse_in_place_batch(py, inverse, transposed, index, change)
        }
    };
}

low_rank_functions!(
    f64,
    low_rank_ratio_f64,
    low_rank_ratio_and_inverse_f64,
    replace_ratio_f64,
    replace_ratio_and_inverse_f64,
    low_rank_ratio_and_inverse_in_place_f64,
    replace_ratio_and_inverse_in_place_f64
);
low_rank_functions!(
    f32,
    low_rank_ratio_f32,
    low_rank_ratio_and_inverse_f32,
    replace_ratio_f32,
    replace_ratio_and_inverse_f32,
    low_rank_ratio_and_inverse_in_place_f32,
    replace_ratio_and_inverse_in_place_f32
);
low_rank_functions!(
    Complex64,
    low_rank_ratio_c128,
    low_rank_ratio_and_inverse_c128,
    replace_ratio_c128,
    replace_ratio_and_inverse_c128,
    low_rank_ratio_and_inverse_in_place_c128,
    replace_ratio_and_inverse_in_place_c128
);
low_rank_functions!(
    Complex32,
    low_rank_ratio_c64,
    low_rank_ratio_and_inverse_c64,
    replace_ratio_c64,
    replace_ratio_and_inverse_c64,
    low_rank_ratio_and_inverse_in_place_c64,
    replace_ratio_and_inverse_in_place_c64
);

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(signed_pfaffian_f64, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_f32, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_f16, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_c128, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_c64, module)?)?;
    module.add_function(wrap_pyfunction!(signed_slog_pfaffian_f64, module)?)?;
    module.add_function(wrap_pyfunction!(signed_slog_pfaffian_f32, module)?)?;
    module.add_function(wrap_pyfunction!(signed_slog_pfaffian_c128, module)?)?;
    module.add_function(wrap_pyfunction!(signed_slog_pfaffian_c64, module)?)?;
    module.add_function(wrap_pyfunction!(flagged_inverse_f64, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_and_inverse_f64, module)?)?;
    module.add_function(wrap_pyfunction!(flagged_inverse_f32, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_and_inverse_f32, module)?)?;
    module.add_function(wrap_pyfunction!(flagged_inverse_c128, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_and_inverse_c128, module)?)?;
    module.add_function(wrap_pyfunction!(flagged_inverse_c64, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_and_inverse_c64, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_f64, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_f64, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_f64, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_f64, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_in_place_f64, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_in_place_f64, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_f32, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_f32, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_f32, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_f32, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_in_place_f32, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_in_place_f32, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_c128, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_c128, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_c128, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_c128, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_in_place_c128, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_in_place_c128, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_c64, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_c64, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_c64, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_c64, module)?)?;
    module.add_function(wrap_pyfunction!(low_rank_ratio_and_inverse_in_place_c64, module)?)?;
    module.add_function(wrap_pyfunction!(replace_ratio_and_inverse_in_place_c64, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{pfaffian_one, slog_pfaffian_from_flat};
    use half::f16;
    use num_complex::Complex;
    use numpy::ndarray::{array, Array2};

    #[test]
    fn two_by_two_is_signed() {
        // pf([[0, -3], [3, 0]]) = -3
        let m = array![[0.0_f64, -3.0], [3.0, 0.0]];
        assert!((pfaffian_one(m) - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn four_by_four_matches_pfaffian_formula() {
        // For [[0,a,b,c],[-a,0,d,e],[-b,-d,0,f],[-c,-e,-f,0]], pf = a*f - b*e + c*d.
        let (a, b, c, d, e, f) = (1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0);
        let m = array![
            [0.0, a, b, c],
            [-a, 0.0, d, e],
            [-b, -d, 0.0, f],
            [-c, -e, -f, 0.0]
        ];
        let expected = a * f - b * e + c * d; // = 6 - 10 + 12 = 8
        assert!((pfaffian_one(m) - expected).abs() < 1e-9);
    }

    #[test]
    fn f32_two_by_two_is_signed() {
        // Same algorithm in single precision.
        let m = array![[0.0_f32, -3.0], [3.0, 0.0]];
        assert!((pfaffian_one(m) - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn odd_is_zero_and_empty_is_one() {
        let odd = array![[0.0_f64, 1.0, 2.0], [-1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]];
        assert_eq!(pfaffian_one(odd), 0.0);
        let empty = Array2::<f64>::zeros((0, 0));
        assert_eq!(pfaffian_one(empty), 1.0);
    }

    #[test]
    fn f16_four_by_four_matches_formula_within_half_precision() {
        // Native half-precision elimination; pf = a*f - b*e + c*d, checked at f16 tolerance.
        let value = |x: f64| f16::from_f64(x);
        let (a, b, c, d, e, f) = (0.5, 0.25, 0.75, 0.125, 0.375, 0.625);
        let zero = f16::from_f64(0.0);
        let m = array![
            [zero, value(a), value(b), value(c)],
            [-value(a), zero, value(d), value(e)],
            [-value(b), -value(d), zero, value(f)],
            [-value(c), -value(e), -value(f), zero]
        ];
        let expected = a * f - b * e + c * d;
        assert!((pfaffian_one(m).to_f64() - expected).abs() < 1e-2);
    }

    #[test]
    fn complex_two_by_two_is_the_offdiagonal() {
        // pf([[0, z], [-z, 0]]) = z for complex z.
        let z = Complex::new(1.0_f64, 2.0);
        let m = array![[Complex::new(0.0, 0.0), z], [-z, Complex::new(0.0, 0.0)]];
        let result = pfaffian_one(m);
        assert!((result - z).norm() < 1e-12);
    }

    #[test]
    fn complex_four_by_four_matches_pfaffian_formula() {
        // pf = a*f - b*e + c*d holds over the complex field as well.
        let a = Complex::new(1.0_f64, -1.0);
        let b = Complex::new(0.5, 2.0);
        let c = Complex::new(-1.5, 0.25);
        let d = Complex::new(2.0, 1.0);
        let e = Complex::new(-0.5, -0.5);
        let f = Complex::new(3.0, -2.0);
        let zero = Complex::new(0.0, 0.0);
        let m = array![
            [zero, a, b, c],
            [-a, zero, d, e],
            [-b, -d, zero, f],
            [-c, -e, -f, zero]
        ];
        let expected = a * f - b * e + c * d;
        assert!((pfaffian_one(m) - expected).norm() < 1e-9);
    }

    #[test]
    fn complex_singular_is_zero() {
        // A complex skew matrix with a zero pivot column has Pfaffian 0.
        let zero = Complex::new(0.0_f64, 0.0);
        let z = Complex::new(1.0, 1.0);
        let m = array![
            [zero, zero, zero, zero],
            [zero, zero, zero, zero],
            [zero, zero, zero, z],
            [zero, zero, -z, zero]
        ];
        assert!(pfaffian_one(m).norm() < 1e-12);
    }

    #[test]
    fn slog_two_by_two_reconstructs_signed() {
        // pf([[0, -3], [3, 0]]) = -3 -> phase = -1, log|Pf| = ln(3).
        let (phase, log_abs) = slog_pfaffian_from_flat(vec![0.0_f64, -3.0, 3.0, 0.0], 2);
        assert!((phase - (-1.0)).abs() < 1e-12);
        assert!((log_abs - 3.0_f64.ln()).abs() < 1e-12);
        assert!((phase * log_abs.exp() - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn slog_matches_linear_four_by_four() {
        let (a, b, c, d, e, f) = (1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0);
        let data = vec![0.0, a, b, c, -a, 0.0, d, e, -b, -d, 0.0, f, -c, -e, -f, 0.0];
        let linear = pfaffian_one(Array2::from_shape_vec((4, 4), data.clone()).unwrap());
        let (phase, log_abs) = slog_pfaffian_from_flat(data, 4);
        assert!((phase * log_abs.exp() - linear).abs() < 1e-9);
    }

    #[test]
    fn slog_exact_zero_pivot_is_zero_neg_inf() {
        // 4x4 with only the trailing 2x2 block nonzero: the first pivot column is zero, so Pf = 0.
        let data = vec![
            0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
        ];
        let (phase, log_abs) = slog_pfaffian_from_flat(data, 4);
        assert_eq!(phase, 0.0);
        assert_eq!(log_abs, f64::NEG_INFINITY);
    }

    #[test]
    fn slog_odd_and_empty() {
        let (phase, log_abs) = slog_pfaffian_from_flat(vec![0.0_f64; 9], 3);
        assert_eq!(phase, 0.0);
        assert_eq!(log_abs, f64::NEG_INFINITY);
        let (phase, log_abs) = slog_pfaffian_from_flat(Vec::<f64>::new(), 0);
        assert_eq!(phase, 1.0);
        assert_eq!(log_abs, 0.0);
    }

    #[test]
    fn slog_complex_two_by_two() {
        // pf([[0, z], [-z, 0]]) = z; phase * exp(log|Pf|) reconstructs z.
        let z = Complex::new(1.0_f64, 2.0);
        let zero = Complex::new(0.0_f64, 0.0);
        let (phase, log_abs) = slog_pfaffian_from_flat(vec![zero, z, -z, zero], 2);
        let reconstructed = phase * Complex::new(log_abs.exp(), 0.0);
        assert!((reconstructed - z).norm() < 1e-12);
    }
}
