use itertools::Itertools;
use std::fs::File;

use crate::n_gram::*;

impl NGram {
    pub fn save(&self, file_name: &str) {
        let mut file_data = String::from("");
        file_data = file_data + &self.num_grams.to_string() + "\n";
        for (bag_name, (total, bag)) in self.bags.clone() {
            // Positive,2,123|
            file_data = file_data + &bag_name + "," + &total.to_string() + "|";
            let bag_vec = bag.into_iter().collect_vec();
            let bag_string = reduce::<(String, f32), String>(&bag_vec, &String::from(""), |(word, prob), acc| {
                // "foo"0.123"bar"0.234
                acc + "\"" + word + "\"" + &prob.to_string()
            });
            file_data = file_data + &bag_string + "\n"
        }
        let mut file = File::create(file_name).expect("Creating object file error");
        file.write_all(file_data.as_bytes()).expect("Writing file error");
    }

    pub fn load(file_name: &str) -> NGram {
        let mut file = File::open(file_name).expect("Creating file object error");
        let mut file_contents = String::new();
        file.read_to_string(&mut file_contents).expect("Reading file error");
        if file_contents.eq("") { panic!("Loading n-gram model: File empty") }
        let bags = file_contents.split("\n");
        let mut bm = BagMap::new();
        let mut num_grams: i8 = 0;
        let mut first = true;
        for bag in bags {
            if bag.eq("") { continue; }
            if first {
                num_grams = bag.parse::<i8>().unwrap();
                first = false;
                continue;
            }
            let mut bag_name = String::new();
            let mut bag_total_s = String::new();
            
            let mut words: HashMap<String, f32> = HashMap::new();
            let mut current_word = String::new();
            let mut current_prob = String::new();

            let mut found_bag_name = false;
            let mut found_bag_total = false;
            let mut finding_word = false;
            for c in bag.chars() {
                if !found_bag_name {
                    if c == ',' {
                        found_bag_name = true;
                        continue;
                    }
                    bag_name = bag_name + &c.to_string();
                    continue;
                }
                if !found_bag_total {
                    if c == '|' {
                        found_bag_total = true;
                        continue;
                    }
                    bag_total_s = bag_total_s + &c.to_string();
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
            let bag_total = bag_total_s.parse::<usize>().unwrap();
            bm.insert(bag_name, (bag_total, words));
        }
        NGram { bags: bm, num_grams }
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