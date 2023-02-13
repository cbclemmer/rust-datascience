use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::util::{InputTup, multi_thread_process_list};

pub type StateMap = HashMap<String, HashMap<String, f32>>;
pub type StateTotals = HashMap<String, HashMap<String, i32>>;

pub struct MarkovChain {
    pub states: StateMap
}

impl MarkovChain {
    fn feed(mut totals: StateTotals, from_state: String, to_state: String) -> StateTotals {
        let o_from_map = totals.get(&from_state);
        if o_from_map.is_some() {
            let mut current_total = 0;
            let mut to_map = o_from_map.expect("map err").clone();
            let o_total = to_map.get(&to_state);
            if o_total.is_some() {
                current_total = o_total.expect("map err").clone();
            }
            to_map.insert(to_state, current_total + 1);
            totals.insert(from_state, to_map);
        } else {
            let mut to_map = HashMap::new();
            to_map.insert(to_state, 1);
            totals.insert(from_state, to_map);
        }
        totals
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

    pub fn train(input_data: Vec<InputTup>) -> StateMap {
        let mut totals = StateTotals::new();
        
        for (from_state, to_state) in input_data {
            totals = MarkovChain::feed(totals.clone(), from_state, to_state);
        }

        // multi_thread_process_list(input_data, totals, 16, f_thread, f_return);
        let states = MarkovChain::calculate_states(totals);
        states
    }

    pub fn new(input_data: Vec<InputTup>) -> MarkovChain {
        let states = MarkovChain::train(input_data);
        MarkovChain { states }
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