use std::collections::HashMap;

use crate::types::{StateMap, StateTotals};
use crate::util::{get_word_pairs, feed_totals_multi};
use crate::types::InputTup;

pub mod file;

pub struct MarkovChain {
    pub states: StateMap
}

impl MarkovChain {
    pub fn new() -> MarkovChain {
        MarkovChain { states: HashMap::new() }
    }

    fn calculate_states(totals: StateTotals) -> StateMap {
        let mut sm = StateMap::new();
        for (from_state, to_map) in totals {
            let mut state_to_map: HashMap<String, f32> = HashMap::new();
            let from_total = to_map
                .clone()
                .into_iter()
                .reduce(|(_, tot), (_, all_tot)| (String::from(""), all_tot + tot))
                .expect("totaling error").1;
            for (to_state, total) in to_map {
                state_to_map.insert(to_state, total as f32 / from_total as f32);
            }
            sm.insert(from_state, state_to_map);
        }
        sm
    }

    pub fn train_file(text_file: &str, white_list_file: &str) -> StateMap {
        println!("Getting input data from file");
        let input_data = get_word_pairs(text_file, None, Some(white_list_file));
        println!("Training from file data");
        MarkovChain::train(&input_data)
    }

    pub fn train(input_data: &Vec<InputTup>) -> StateMap {
        let totals = feed_totals_multi(input_data, None);
        MarkovChain::calculate_states(totals)
    }

    pub fn predict(sm: StateMap, state: String) -> String {
        let o_to_map = sm.get(&state);
        if o_to_map.is_none() {
            return String::from("");
        }
        o_to_map
            .expect("map err")
            .clone()
            .into_iter()
            .max_by(|(_, prob1), (_, prob2)| prob1.total_cmp(prob2))
            .expect("max err").0
    }
}