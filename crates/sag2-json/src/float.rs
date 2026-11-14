//! `repr(float)` compatibility with CPython 3.12.
//!
//! `sag2.py` formats every outward-facing float through `json.dumps`, and
//! CPython's JSON encoder writes floats with `float.__repr__`. The parity
//! contract (design §1.0, layer S) therefore depends on producing the exact
//! same bytes as CPython 3.12, not merely the same numeric value.
//!
//! CPython renders finite floats from `_Py_dg_dtoa` (David Gay, mode 0: the
//! shortest digit string that round-trips) and then picks fixed or exponential
//! notation in `format_float_short`:
//!
//! * exponential when `decpt <= -4 || decpt > 16`, where `value` is
//!   `0.<digits> * 10^decpt`;
//! * the exponent is `decpt - 1`, always signed and zero-padded to at least two
//!   digits (`1e+16`, `1e-07`, `1e+300`);
//! * integer-valued floats keep a `.0` suffix (`1.0`, never `1`);
//! * `-0.0` keeps its sign, and `inf`/`-inf`/`nan` are spelled
//!   `inf`/`-inf`/`nan` (a NaN's sign bit is not rendered).
//!
//! The shortest digits come from `ryu`, which is verified to produce the same
//! correctly-rounded shortest sequence as Gay's `dtoa`. Ryu's own notation
//! thresholds are irrelevant: its output is parsed back into
//! `(digits, decpt)` and re-rendered with CPython's rules.

/// Format `value` exactly like CPython 3.12's `repr(float)`.
///
/// ```
/// # use sag2_json::py_repr;
/// assert_eq!(py_repr(1e16), "1e+16");
/// assert_eq!(py_repr(1e-7), "1e-07");
/// assert_eq!(py_repr(1.0), "1.0");
/// assert_eq!(py_repr(-0.0), "-0.0");
/// assert_eq!(py_repr(f64::NAN), "nan");
/// ```
pub fn py_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "inf"
        } else {
            "-inf"
        }
        .to_owned();
    }

    let negative = value.is_sign_negative();
    let (digits, decpt) = shortest_digits(value.abs());

    let mut out = String::with_capacity(digits.len() + 8);
    if negative {
        out.push('-');
    }

    let digit_count = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digit_count > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let exponent = decpt - 1;
        if exponent < 0 {
            out.push('-');
        } else {
            out.push('+');
        }
        let magnitude = exponent.unsigned_abs();
        // CPython uses `%+03d`: sign plus at least two exponent digits.
        if magnitude < 10 {
            out.push('0');
        }
        out.push_str(&magnitude.to_string());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt >= digit_count {
        out.push_str(&digits);
        for _ in 0..decpt - digit_count {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        let split = decpt as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    }
    out
}

/// Shortest round-trip digits of a finite, non-negative value, plus the
/// decimal-point position (CPython's `decpt`).
fn shortest_digits(value: f64) -> (String, i32) {
    debug_assert!(value.is_finite() && !value.is_sign_negative());
    if value == 0.0 {
        return ("0".to_owned(), 1);
    }

    let mut buffer = ryu::Buffer::new();
    split_decimal(buffer.format_finite(value))
}

/// Convert a rendered shortest decimal (`123.456`, `0.001234`, `1.2e30`,
/// `1e-7`, ...) into `(digits, decpt)`.
fn split_decimal(rendered: &str) -> (String, i32) {
    let (mantissa, exponent) = match rendered.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (
            mantissa,
            exponent.parse::<i32>().expect("ryu emits a valid exponent"),
        ),
        None => (rendered, 0),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));

    let mut all = String::with_capacity(integer.len() + fraction.len());
    all.push_str(integer);
    all.push_str(fraction);

    let digits = all.trim_start_matches('0');
    let leading_zeros = all.len() - digits.len();
    let decpt = integer.len() as i32 - leading_zeros as i32 + exponent;

    let mut digits = digits.to_owned();
    if digits.is_empty() {
        // Only reachable for zero, which is handled by the caller.
        digits.push('0');
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    (digits, decpt)
}

#[cfg(test)]
mod tests {
    use super::py_repr;

    #[test]
    fn cpython_notation_thresholds() {
        assert_eq!(py_repr(1e15), "1000000000000000.0");
        assert_eq!(py_repr(1e16), "1e+16");
        assert_eq!(py_repr(0.0001), "0.0001");
        assert_eq!(py_repr(0.00001), "1e-05");
        assert_eq!(py_repr(1.7976931348623157e308), "1.7976931348623157e+308");
        assert_eq!(py_repr(5e-324), "5e-324");
    }

    #[test]
    fn cpython_special_values() {
        assert_eq!(py_repr(0.0), "0.0");
        assert_eq!(py_repr(-0.0), "-0.0");
        assert_eq!(py_repr(f64::INFINITY), "inf");
        assert_eq!(py_repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(py_repr(f64::NAN), "nan");
        assert_eq!(py_repr(-f64::NAN), "nan");
    }
}
