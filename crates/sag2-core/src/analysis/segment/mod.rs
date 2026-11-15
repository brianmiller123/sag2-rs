//! Language segmenters for the multilingual profiles.
//!
//! A profile whose language writes without spaces ([`thai`]) replaces the
//! shared §4 jieba pipeline with its own word boundaries;
//! [`crate::analysis::tokenize_with`] keeps the certified ASCII model and the
//! final `HAS_CONTENT`/stopword filter around it, so every profile ends in the
//! same filters.

pub mod thai;
