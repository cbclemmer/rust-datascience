use std::collections::HashMap;

pub type InputTup = (String, String);
pub type ValidationMap = HashMap<String, bool>;

// n_gram
// (total num words, word -> probability)
// probability = num times word appears / total num words for type
pub type NgramBag = (usize, HashMap<String, f32>);
pub type NgramMap = HashMap<String, NgramBag>;


// Markov Chain
pub type StateMap = HashMap<String, HashMap<String, f32>>;
pub type StateTotals = HashMap<String, HashMap<String, i32>>;