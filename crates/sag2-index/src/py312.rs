//! CPython 3.12's `sum()` for floats.
//!
//! Python 3.12 replaced the naive accumulation in `builtin_sum` with Neumaier
//! compensated summation (`Objects/bltinmodule.c`).  The reference
//! implementation depends on it in two places that reach a gate:
//!
//! * the pruning bound in `CorpusIndex._retrieve_snapshot` (`sag2.py:514`),
//!   which is a `sum()` over the unselected IDFs; and
//! * `rank_dpp`'s three `sum()` calls (`sag2.py:133/142/163`), which P2 owns.
//!
//! The reference repository declares `requires-python = ">=3.11"` but its CI
//! does not pin an interpreter (design §11, divergence D5); the port targets
//! 3.12 semantics and `tools/export_golden.py` refuses to run on anything older.

/// `sum(iterable)` for a stream of Python floats, as of CPython 3.12.
pub fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    // CPython only folds the compensation back in when it is a non-zero finite
    // value, which keeps an infinite total from turning into NaN.
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::sum;

    #[test]
    fn compensates_where_naive_accumulation_does_not() {
        // Python 3.12: sum([1e16, 1.0, -1e16]) == 1.0; Python 3.11 returns 0.0.
        assert_eq!(sum([1e16, 1.0, -1e16]), 1.0);
    }

    #[test]
    fn equals_naive_addition_for_exact_sums() {
        assert_eq!(sum([0.5, 0.25, 0.125]), 0.875);
        assert_eq!(sum([]), 0.0);
    }
}
