
use crate::util::feed_totals_multi;
use crate::types::{StateTotals, StateMap};

pub struct Word2Vec {
    pub total: i64,
    pub data: StateMap
}

impl Word2Vec {
    fn calculate_states(totals: StateTotals, total: i32) -> StateMap {
        let mut sm = StateMap::new();
        for (from_state, to_map) in totals {
            let mut state_to_map: HashMap<String, f32> = HashMap::new();
            for (to_state, total) in to_map {
                state_to_map.insert(to_state, total as f32 / total as f32);
            }
            sm.insert(from_state, state_to_map);
        }
        sm
    }

    pub fn train_file(text_file: &str, bl_word_file: Option<&str>, wl_word_file: Option<&str>) {
        let input_data = get_word_pairs(text_file, bl_word_file, wl_word_file);
        Word2Vec::train(input_data);
    }

    pub fn train(input_data: &Vec<InputTup>) -> StateMap {
        // order does not matter so it will be inserted twice
        let mut totals = feed_totals_multi(input_data, None);
        let rev_input_data = input_data
            .into_iter()
            .map(|(wd1, wd2)| (wd2, wd1))
            .collect_vec();
        totals = feed_totals_multi(input_data, totals);
        Word2Vec::calculate_states(totals, input_data.len())
    }
}