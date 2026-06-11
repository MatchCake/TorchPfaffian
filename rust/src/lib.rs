use num_traits::Float;
use numpy::ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

const PIVOT_EPSILON: f64 = 1e-30;

/// Batches with at least this many matrices are evaluated in parallel across threads; smaller
/// batches run serially to avoid thread-pool overhead.
const PARALLEL_BATCH_THRESHOLD: usize = 8;

/// Signed Pfaffian of a single skew-symmetric matrix via Parlett-Reid elimination.
///
/// Generic over the floating precision so the same algorithm serves ``f32`` and ``f64`` inputs.
/// The matrix is copied into a flat row-major buffer so the hot rank-2 Schur update runs over a
/// contiguous row slice, which the compiler can auto-vectorize (far cheaper than per-element
/// strided ndarray indexing).
fn pfaffian_one<T: Float>(matrix: Array2<T>) -> T {
    let dimension = matrix.nrows();
    if dimension % 2 == 1 {
        return T::zero();
    }
    if dimension == 0 {
        return T::one();
    }
    let mut data: Vec<T> = matrix.iter().copied().collect(); // row-major, len dimension * dimension
    let epsilon = T::from(PIVOT_EPSILON).unwrap();
    let mut sign = T::one();
    let mut column = 0usize;
    while column + 2 < dimension {
        // Partial pivoting: largest |data[row, column]| for row > column + 1.
        let mut pivot_row = column + 2;
        let mut best = data[(column + 2) * dimension + column].abs();
        for row in (column + 3)..dimension {
            let candidate = data[row * dimension + column].abs();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if best > data[(column + 1) * dimension + column].abs() {
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
        if pivot.abs() < epsilon {
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

/// Signed Pfaffian of each owned matrix, computed in parallel across the batch above a threshold.
///
/// The batch elements are independent, so they are mapped over rayon threads; the per-matrix
/// Parlett-Reid elimination itself stays sequential. The caller releases the GIL around this.
fn signed_pfaffian_owned<T: Float + Send + Sync>(matrices: Vec<Array2<T>>) -> Vec<T> {
    if matrices.len() >= PARALLEL_BATCH_THRESHOLD {
        matrices.into_par_iter().map(pfaffian_one).collect()
    } else {
        matrices.into_iter().map(pfaffian_one).collect()
    }
}

/// Copy each ``(n, n)`` slice of a ``(batch, n, n)`` view into an owned matrix.
fn owned_matrices<T: Float + numpy::Element>(matrix: &PyReadonlyArray3<'_, T>) -> Vec<Array2<T>> {
    let view = matrix.as_array();
    let batch = view.shape()[0];
    (0..batch).map(|index| view.index_axis(Axis(0), index).to_owned()).collect()
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

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(signed_pfaffian_f64, module)?)?;
    module.add_function(wrap_pyfunction!(signed_pfaffian_f32, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::pfaffian_one;
    use numpy::ndarray::array;

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
        let empty = numpy::ndarray::Array2::<f64>::zeros((0, 0));
        assert_eq!(pfaffian_one(empty), 1.0);
    }
}
