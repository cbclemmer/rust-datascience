use itertools::Itertools;
use std::fs::File;

use crate::n_gram::*;

impl NGram {
    pub fn save(&self, file_name: &str) {
        let mut file_data = String::from("");
        for i in 0..self.max_grams {
            for (type_name, (total, g_map)) in self.ngram_maps.index(i as usize).clone() {
                // Positive,123|
                file_data = file_data + &type_name + "," + &total.to_string() + "|";
                let ret_vec = g_map.into_iter().collect_vec();
                let map_string = reduce::<(String, f32), String>(&ret_vec, &String::from(""), |(word, prob), acc| {
                    // "foo"0.123"bar"0.234
                    acc + "\"" + word + "\"" + &prob.to_string()
                });
                file_data = file_data + &map_string + "\n"
            }
            file_data = file_data + "<<GRAM>>\n"; // at the end of each gram map
        }
        let mut file = File::create(file_name).expect("Creating object file error");
        file.write_all(file_data.as_bytes()).expect("Writing file error");
    }

    pub fn load(file_name: &str) -> NGram {
        let mut file = File::open(file_name).expect("Creating file object error");
        let mut file_contents = String::new();
        file.read_to_string(&mut file_contents).expect("Reading file error");
        if file_contents.eq("") { panic!("Loading n-gram model: File empty") }
        let lines = file_contents.split("\n");
        let mut gram_maps: Vec<NgramMap> = Vec::new();
        let mut max_grams = 0;

        let mut bm: NgramMap = NgramMap::new();
        for g_map in lines {
            if g_map.eq("") { continue; }
            if g_map.eq("<<GRAM>>") {
                gram_maps.push(bm.clone());
                bm.clear();
                max_grams = max_grams + 1;
                continue;
            }
            let mut type_name = String::new();
            let mut map_total_s = String::new();
            
            let mut words: HashMap<String, f32> = HashMap::new();
            let mut current_word = String::new();
            let mut current_prob = String::new();

            let mut found_type_name = false;
            let mut found_type_total = false;
            let mut finding_word = false;
            for c in g_map.chars() {
                if !found_type_name {
                    if c == ',' {
                        found_type_name = true;
                        continue;
                    }
                    type_name = type_name + &c.to_string();
                    continue;
                }
                if !found_type_total {
                    if c == '|' {
                        found_type_total = true;
                        continue;
                    }
                    map_total_s = map_total_s + &c.to_string();
                    continue;
                }
                if c == '\"' {
                    if !finding_word && !current_prob.eq("") {
                        words.insert(current_word.clone(), current_prob.clone().parse::<f32>().unwrap());
                        current_word = String::new();
                        current_prob = String::new();
                    }
                    finding_word = !finding_word;
                    continue;
                }
                if finding_word {
                    current_word = current_word + &c.to_string();
                } else {
                    current_prob = current_prob + &c.to_string();
                }
            }
            let type_total = map_total_s.parse::<usize>().unwrap();
            bm.insert(type_name, (type_total, words));
        }
        NGram { ngram_maps: gram_maps, max_grams }
    }

    pub fn parse(ngram_file_path: &str, input: Vec<String>, output_file_path: &str) {
        let ngram = NGram::load(ngram_file_path);

        let mut results = Vec::new();
        for sentence in input {
            let res = ngram.test_sentence(&sentence);
            results.push(res + "\n");
        }
        let mut output_file = File::create(output_file_path).expect("Error creating output file");
        output_file.write(results.concat().as_bytes()).expect("Error writing result to file");
    }
}