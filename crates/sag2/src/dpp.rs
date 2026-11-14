//! `rank_dpp` (`sag2.py:126-167`) - greedy DPP reordering of the final head.
//!
//! The reference's numerics are load-bearing (design §9 R5):
//!
//! * The norm, the kernel dot product and the projection are built-in `sum()`
//!   calls, which CPython 3.12 implements with Neumaier compensation
//!   (`sag2.py:133/142/163`).  [`py312_sum`] reproduces that exactly.
//! * `Counter(diversity_tokens(content)).items()` iterates in first-occurrence
//!   order, and the `sum()` above runs over that order.  The vectors here are
//!   therefore insertion-ordered `Vec<(String, f64)>`, never a hash map.
//! * `max(remaining, key=...)` returns the **first** maximal element, not the
//!   last; the tie-break is observable in `order`.
//! * `math.ulp(max(residuals))` bounds the tail, and every residual update is
//!   clamped with `max(0.0, ...)`.

use sag2_core::diversity_tokens;
use sag2_index::py312_sum;

use crate::Candidate;

/// Why `rank_dpp` cannot reproduce the reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DppError {
    /// `idf[term]` is missing; Python raises `KeyError` (`sag2.py:130`).
    MissingIdf(String),
    /// The vector norm is zero; Python raises `ZeroDivisionError`.
    ZeroNorm,
}

impl std::fmt::Display for DppError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DppError::MissingIdf(term) => {
                write!(formatter, "diversity_idf has no weight for {term:?}")
            }
            DppError::ZeroNorm => write!(formatter, "a diversity vector has a zero norm"),
        }
    }
}

impl std::error::Error for DppError {}

/// `math.ulp(x)`: `nextafter(|x|, inf) - |x|`, with CPython's three special
/// cases (`nan -> nan`, `inf -> inf`, `0 -> 5e-324`).
fn ulp(value: f64) -> f64 {
    if value.is_nan() {
        value
    } else if value.is_infinite() {
        value.abs()
    } else if value == 0.0 {
        f64::from_bits(1)
    } else {
        let magnitude = value.abs();
        f64::from_bits(magnitude.to_bits() + 1) - magnitude
    }
}

/// One candidate's normalised term vector, in first-occurrence order.
fn vector(candidate: &Candidate, idf: &std::collections::BTreeMap<String, f64>) -> Result<Vec<(String, f64)>, DppError> {
    let mut counts: Vec<(String, i64)> = Vec::new();
    for term in diversity_tokens(&candidate.hit.content) {
        match counts.iter_mut().find(|(existing, _)| *existing == term) {
            Some((_, count)) => *count += 1,
            None => counts.push((term, 1)),
        }
    }
    let mut weighted = Vec::with_capacity(counts.len());
    for (term, count) in counts {
        let weight = idf.get(&term).copied().ok_or(DppError::MissingIdf(term.clone()))?;
        weighted.push((term, (1.0 + (count as f64).ln()) * weight));
    }
    let norm = py312_sum(weighted.iter().map(|(_, value)| value * value)).sqrt();
    if norm == 0.0 {
        return Err(DppError::ZeroNorm);
    }
    Ok(weighted
        .into_iter()
        .map(|(term, value)| (term, value / norm))
        .collect())
}

/// `sag2.py:126-167` `rank_dpp`, reordering `candidates` in place.
pub fn rank_dpp(
    candidates: &mut Vec<Candidate>,
    idf: &std::collections::BTreeMap<String, f64>,
) -> Result<(), DppError> {
    let mut vectors = Vec::with_capacity(candidates.len());
    for candidate in candidates.iter() {
        vectors.push(vector(candidate, idf)?);
    }
    let scores: Vec<f64> = candidates.iter().map(|candidate| candidate.jev_score).collect();
    if scores.is_empty() {
        return Ok(());
    }
    let mut kernel = vec![vec![0.0_f64; candidates.len()]; candidates.len()];
    for i in 0..candidates.len() {
        for j in 0..candidates.len() {
            let dot = py312_sum(vectors[i].iter().map(|(term, value)| {
                let other = vectors[j]
                    .iter()
                    .find(|(existing, _)| existing == term)
                    .map_or(0.0, |(_, weight)| *weight);
                value * other
            }));
            kernel[i][j] = scores[i] * scores[j] * dot;
        }
    }
    let mut residuals: Vec<f64> = scores.iter().map(|score| score * score).collect();
    for (i, value) in residuals.iter().enumerate() {
        kernel[i][i] = *value;
    }
    let mut maximum = residuals[0];
    for value in &residuals[1..] {
        if *value > maximum {
            maximum = *value;
        }
    }
    let tolerance = candidates.len() as f64 * ulp(maximum);
    let mut coefficients: Vec<Vec<f64>> = vec![Vec::new(); candidates.len()];
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    let mut order: Vec<usize> = Vec::new();
    while !remaining.is_empty() {
        // `max(remaining, key=lambda i: residuals[i])` keeps the first maximum.
        let mut winner = remaining[0];
        for &candidate in &remaining[1..] {
            if residuals[candidate] > residuals[winner] {
                winner = candidate;
            }
        }
        if residuals[winner] <= tolerance {
            order.extend(remaining.iter().copied());
            break;
        }
        order.push(winner);
        let position = remaining
            .iter()
            .position(|candidate| *candidate == winner)
            .expect("the winner is in the remaining order");
        remaining.remove(position);
        let denominator = residuals[winner].sqrt();
        for &candidate in &remaining {
            let projection = (kernel[winner][candidate]
                - py312_sum(
                    coefficients[winner]
                        .iter()
                        .zip(coefficients[candidate].iter())
                        .map(|(left, right)| left * right),
                ))
                / denominator;
            coefficients[candidate].push(projection);
            let value = residuals[candidate] - projection * projection;
            // `max(0.0, value)`: the first argument wins ties and NaN.
            residuals[candidate] = if value > 0.0 { value } else { 0.0 };
        }
    }
    let mut reordered = Vec::with_capacity(candidates.len());
    for position in order {
        reordered.push(candidates[position].clone());
    }
    *candidates = reordered;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sag2_index::Hit;

    use super::{rank_dpp, ulp};
    use crate::Candidate;

    fn candidate(id: i64, content: &str, score: f64) -> Candidate {
        Candidate {
            hit: Hit {
                id,
                document_id: id,
                title: String::new(),
                heading: String::new(),
                content: content.to_owned(),
                full_doc: None,
                spans: Vec::new(),
                characters: 0,
                source_id: String::new(),
                score: 0.0,
            },
            hop: 0,
            fusion_score: 0.0,
            jev_score: score,
            jev_model: None,
            previous_jev_score: None,
        }
    }

    fn idf(terms: &[(&str, f64)]) -> BTreeMap<String, f64> {
        terms
            .iter()
            .map(|(term, value)| ((*term).to_owned(), *value))
            .collect()
    }

    #[test]
    fn ulp_matches_cpython() {
        assert_eq!(ulp(0.0), 5e-324);
        assert_eq!(ulp(1.0), 2.0_f64.powi(-52));
        assert_eq!(ulp(-1.0), 2.0_f64.powi(-52));
        assert_eq!(ulp(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn selects_the_first_maximum_and_keeps_residual_order() {
        // Three candidates with equal scores and a shared term: the initial
        // residuals tie, so a "last maximum" implementation picks C first and
        // returns a different order.
        let mut candidates = vec![
            candidate(1, "x", 1.0),
            candidate(2, "y", 1.0),
            candidate(3, "x y", 1.0),
        ];
        rank_dpp(&mut candidates, &idf(&[("x", 1.0), ("y", 1.0)])).expect("dpp");
        let ids: Vec<i64> = candidates.iter().map(|c| c.hit.id).collect();
        assert_eq!(ids, [1, 2, 3]);
    }

    #[test]
    fn keeps_the_reference_order_for_equal_candidates() {
        let mut candidates = vec![candidate(7, "same", 0.4), candidate(8, "same", 0.4)];
        rank_dpp(&mut candidates, &idf(&[("same", 1.0)])).expect("dpp");
        assert_eq!(
            candidates.iter().map(|c| c.hit.id).collect::<Vec<_>>(),
            [7, 8]
        );
    }


    #[test]
    fn reorders_the_c0_head_like_the_reference() {
        // C0/Anchor's final three candidates, with the index's `diversity_idf`
        // values: the reference's `rank_dpp` returns [Multiple, High, Anchor].
        let contents = [
            ("Alice Paul birthplace Bob Smith.", 3, 0.95),
            (
                "Anchor birthplace depicts Alice Paul. Alice Paul. Bob Smith.",
                1,
                0.8,
            ),
            ("Anchor highname commonname.", 8, 0.7),
        ];
        let idf_values = [
            ("alice", 1.780158557549575),
            ("paul", 1.780158557549575),
            ("birthplace", 2.791759469228055),
            ("bob", 1.780158557549575),
            ("smith", 1.780158557549575),
            ("anchor", 2.791759469228055),
            ("depicts", 3.4849066497880004),
            ("highname", 3.4849066497880004),
            ("commonname", 3.0794415416798357),
        ];
        let idf = idf(&idf_values);
        let mut candidates: Vec<Candidate> = contents
            .iter()
            .map(|(content, id, score)| candidate(*id, content, *score))
            .collect();
        rank_dpp(&mut candidates, &idf).expect("dpp");
        let ids: Vec<i64> = candidates.iter().map(|c| c.hit.id).collect();
        assert_eq!(ids, [3, 8, 1]);
    }

    #[test]
    fn matches_the_reference_where_compensated_summation_matters() {
        // These two inputs (generated from the reference `rank_dpp` with
        // `PYTHONHASHSEED=0`) flip their last candidates when the norm, the
        // kernel dot product and the projection use naive addition instead of
        // CPython 3.12's compensated `sum()` (design §9 R5).  The gate fixtures
        // are not sensitive to the last bit, so R5's guard lives here.
        let idf_values = [
            ("w0", 0.7242596137760313),
            ("w1", 2.437378041851008),
            ("w2", 3.7532731122912852),
            ("w3", 2.2688244296286744),
            ("w4", 0.9513694736373618),
            ("w5", 0.5859537164236872),
        ];
        let case_one = [
            ("w2 w3 w0 w0", 0.05),
            ("w3 w4 w5 w0 w1", 0.7),
            ("w0 w2 w4", 0.8),
            ("w3 w0 w5 w3 w2 w5", 0.99),
            ("w5 w5 w2 w1 w2 w4", 0.5),
            ("w1 w3", 0.99),
            ("w4 w3 w2 w4 w0", 0.95),
            ("w0 w0 w2 w3 w5 w1", 0.99),
        ];
        let case_two = [
            ("w1 w3 w4 w3 w2 w4", 0.1),
            ("w0 w5 w4 w3 w0", 0.5),
            ("w5 w1 w2 w0 w4 w2", 0.05),
            ("w3 w5 w3 w3 w2", 0.8),
            ("w1 w1 w1 w1", 0.95),
            ("w2 w2 w1 w3", 0.01),
            ("w1 w5 w2 w4", 0.99),
            ("w5 w3 w0 w0 w4", 0.5),
        ];
        for (case, expected) in [(case_one, [4, 6, 3, 2, 7, 5, 8, 1]), (case_two, [7, 5, 4, 2, 1, 3, 6, 8])] {
            let mut candidates: Vec<Candidate> = case
                .iter()
                .enumerate()
                .map(|(position, (content, score))| {
                    candidate((position + 1) as i64, content, *score)
                })
                .collect();
            rank_dpp(&mut candidates, &idf(&idf_values)).expect("dpp");
            let ids: Vec<i64> = candidates.iter().map(|c| c.hit.id).collect();
            assert_eq!(ids, expected);
        }
    }

    #[test]
    fn reports_a_missing_idf_weight() {
        let mut candidates = vec![candidate(1, "absent", 1.0)];
        let error = rank_dpp(&mut candidates, &idf(&[])).expect_err("missing weight");
        assert_eq!(error, super::DppError::MissingIdf("absent".to_owned()));
    }
}
