use std::{fs::File, io::{Write, Read}};

use crate::markov_chain::*;

impl MarkovChain {
    // from_word|"to_word"prob,"to_word"prob...\n
    pub fn save(&self, file_name: &str) {
        let mut file_data = String::from("");
        for (from_word, map) in &self.states {
            file_data = format!("{}{}|", file_data, from_word);
            for (to_word, prob) in map {
                file_data = format!("{}\"{}\"{}", file_data, to_word, prob.to_string());
            }
            file_data = format!("{}\n", file_data);
        }
        let mut file = File::create(file_name).expect("Error creating file object");
        file.write_all(file_data.as_bytes()).expect("Error writing to file");
    }

    pub fn load(file_name: &str) -> MarkovChain {
        let mut file = File::open(file_name).expect("Error creating file object");
        let mut file_contents = String::new();        
        file.read_to_string(&mut file_contents).expect("Error reading file");
        if file_contents.eq("") { panic!("Error loading markov model: File empty") }
        let lines = file_contents.split("\n");
        let mut maps = HashMap::new();
        for line in lines {
            let mut from_word = String::new();
            let mut current_to_word = String::new();
            let mut current_prob_s = String::new();
            let mut found_from_state = false;
            let mut finding_to_word = true;
            let mut map = HashMap::new();
            for c in line.chars() {
                if !found_from_state {
                    if c == '|' {
                        found_from_state = true;
                        continue;
                    }
                    from_word = from_word + &c.to_string();
                }
                if c == '"' {
                    if !finding_to_word && !current_prob_s.eq("") {
                        map.insert(current_to_word.clone(), current_prob_s.clone().parse::<f32>().unwrap());
                        current_to_word = String::new();
                        current_prob_s = String::new();
                    }
                    finding_to_word = !finding_to_word;
                }
                if finding_to_word {
                    current_to_word = current_to_word + &c.to_string();
                } else {
                    current_prob_s = current_prob_s + &c.to_string();
                }
            }
            maps.insert(from_word.clone(), map);
        }
        MarkovChain { states: maps }
    }
}