use std::{fs::File, io::{Write, BufReader, BufRead}};

use crate::markov_chain::*;

impl MarkovChain {
    // from_word|"to_word"prob,"to_word"prob...\n
    pub fn save(&self, file_name: &str) {
        println!("Saving markov chain to file: {}", file_name);
        let mut file_data = String::from("");
        let mut i = 0;
        let mut file = File::create(file_name).expect("Error creating file object");
        let num_states = self.states.len();
        for (from_word, map) in &self.states {
            if i % 100  == 0 {
                println!("Wrote {} lines of {}", i, num_states);
            }
            file_data = format!("{}{}|", file_data, from_word);
            for (to_word, prob) in map {
                file_data = format!("{}\"{}\"{}", file_data, to_word, prob.to_string());
            }
            file_data = format!("{}\n", file_data);
            file.write(file_data.as_bytes()).expect("Error writing to file");
            i = i + 1;
        }
    }

    pub fn load(file_name: &str) -> MarkovChain {
        let file = File::open(file_name).expect("Error creating file object");
        let reader = BufReader::new(file);
        let mut maps = HashMap::new();
        let mut i = 0;
        for ln in reader.lines() {
            let line = ln.expect("Error reading line");
            if i % 100 == 0 {
                println!("read {} lines", i);
            }
            let mut from_word = String::new();
            let mut current_to_word = String::new();
            let mut current_prob_s = String::new();
            let mut found_from_state = false;
            let mut finding_to_word = false;
            let mut map = HashMap::new();
            for c in line.chars() {
                if !found_from_state {
                    if c == '|' {
                        found_from_state = true;
                        continue;
                    }
                    from_word = from_word + &c.to_string();
                    continue;
                }
                if c == '"' {
                    if !finding_to_word && !current_prob_s.eq("") {
                        map.insert(current_to_word.clone(), current_prob_s.clone().parse::<f32>().unwrap());
                        current_to_word = String::new();
                        current_prob_s = String::new();
                    }
                    finding_to_word = !finding_to_word;
                    continue;
                }
                if finding_to_word {
                    current_to_word = current_to_word + &c.to_string();
                } else {
                    current_prob_s = current_prob_s + &c.to_string();
                }
            }
            maps.insert(from_word.clone(), map);
            println!("{}", from_word);
            i = i + 1;
        }
        MarkovChain { states: maps }
    }
}