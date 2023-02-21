use itertools::Itertools;
use std::{collections::HashMap, io::{Write, Read}};
use std::collections::VecDeque;

pub mod config;
pub mod learn;
pub mod file;

use crate::util::{InputTup, multi_thread_process_list, reduce};

// (total num words, word -> probability)
// probability = num times word appears / total num words
pub type WordBag = (usize, HashMap<String, f32>);
pub type BagMap = HashMap<String, WordBag>;

pub struct NGram {
    pub bags: BagMap,
    pub num_grams: i8
}

impl NGram {
    fn create_grams(s: &String, n: usize) -> Vec<String> {
        let mut ret_val = Vec::new();
        let mut last_words: VecDeque<String> = VecDeque::new();
        for wd in s.split(" ") {
            if wd.eq("") { continue; }
            last_words.push_back(String::from(wd));
            if last_words.len() < n {
                continue;
            }
            let mut gram = String::from("");
            let mut first = true;
            for g_wd in last_words.clone() {
                let w = if first { g_wd } else { format!(" {}", g_wd) };
                if first {
                    first = false;
                }
                gram.push_str(&w);
            }
            last_words.pop_front();
            ret_val.push(gram);
        }
        ret_val
    }

    fn train_word_vector(&mut self, bag_name: &String, input_data: &Vec<String>) {
        let gram_groups = input_data.iter()
            .flat_map(|s| NGram::create_grams(s, self.num_grams as usize))
            .sorted()
            .group_by(|s| String::from(s.to_owned()));

        let grams = gram_groups
            .into_iter()
            .map(|(wd, grp)| (wd, grp.count()))
            .collect_vec();
        
        let o_wv = self.bags.get(bag_name);
        let (current_total, mut wv) = if o_wv.is_some() 
            { o_wv.expect("ERR").clone() } 
            else { (0 as usize, HashMap::new() as HashMap<String, f32>) };

        let input_length = input_data.len();
        let total_inputs = input_length + current_total;
        for (wd, input_count) in grams {
            if wd.eq("") {continue;}
            let o_current_prob = wv.get(&wd);
            let current_prob = if o_current_prob.is_some()
                { o_current_prob.expect("ERR").clone() }
                else { 0 as f32 };

            let current_count = f32::ceil(current_prob * current_total as f32) as usize;
            let total_count = input_count + current_count;

            let prob = total_count as f32 / total_inputs as f32;
            wv.insert(String::from(wd), prob);
        }
        self.bags.insert(bag_name.clone(), (total_inputs, wv));
    }

    pub fn new(input_data: &Vec<InputTup>, num_grams: i8) -> NGram {
        let mut bow = NGram { bags: BagMap::new(), num_grams };
        bow.train(input_data);
        bow
    }
    
    pub fn train(&mut self, input_data: &Vec<InputTup>) {
        let input_groups = input_data.iter()
            .filter(|tup| tup.0 != "")
            .sorted_by(|tup1, tup2| tup1.0.cmp(&tup2.0))
            .group_by(|&tup| tup.0.to_owned());
        
        for (key, group) in &input_groups {
            let wv_input = group.map(|tup| tup.1.to_owned()).collect_vec();
            self.train_word_vector(&key, &wv_input);
        }
    }

    fn test_gram(bow: &BagMap, gram: &String) -> String {
        let mut best_prob: (String, f32) = (String::from(""), 0.0);
        for (bag_name, (_, bag)) in bow.into_iter() {
            let m_prob = bag.get(gram);
            if m_prob.is_none() {
                continue;
            }
            let prob = m_prob.expect("");
            if *prob > best_prob.1 {
                best_prob = (bag_name.to_owned(), prob.to_owned());
            }
        }
        String::from(best_prob.0)
    }

    pub fn test_sentence_static(bow: &BagMap, num_grams: i8, sentence: &String) -> String {
        let mut totals_hm: HashMap<String, i32> = HashMap::new();
        for wd in NGram::create_grams(sentence, num_grams as usize) {
            let best_bag = NGram::test_gram(bow, &String::from(wd));
            if best_bag.eq("") { continue; }
            let m_total = totals_hm.get(&best_bag);
            let total: i32 = if m_total.is_none() { 1 } else { m_total.expect("") + 1 };
            totals_hm.insert(best_bag, total);
        }
    
        let mut best_bag = (String::from(""), 0);
        for (bag_name, total) in totals_hm.into_iter() {
            // println!("{}: {}", bag_name, total);
            if total > best_bag.1 {
                best_bag = (bag_name, total)
            }
        }
    
        best_bag.0
    }

    pub fn test_sentence(&self, sentence: &String) -> String {
        NGram::test_sentence_static(&self.bags, self.num_grams, sentence)
    }

    pub fn validate(bags: &BagMap, num_grams: i8, input: &Vec<InputTup>) -> f32 {
        let num_inputs = input.len();
        let f_thread = |(c_bags, c_num_grams): (BagMap, i8), chunk: &Vec<(String, String)>| -> Vec<bool> {
            let mut correct_vec = Vec::new();
            for (tweet_type, sentence) in chunk {
                correct_vec.push(&NGram::test_sentence_static(&c_bags, c_num_grams, sentence) == tweet_type)
            }
            correct_vec
        };
        
        let results = multi_thread_process_list(input, (bags.clone(), num_grams), 16, f_thread, None);
        let num_correct = results.into_iter().filter(|b| *b).collect_vec().len() as i32;
        
        num_correct as f32 / num_inputs as f32
    }
}