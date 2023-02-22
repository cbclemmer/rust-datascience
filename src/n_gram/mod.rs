use itertools::Itertools;
use std::collections::HashMap;
use std::io::{Write, Read};
use std::ops::Index;
use std::collections::VecDeque;

pub mod config;
pub mod learn;
pub mod file;

use crate::util::InputTup;
use crate::util::{multi_thread_process_list, reduce};

// (total num words, word -> probability)
// probability = num times word appears / total num words for type
pub type NgramBag = (usize, HashMap<String, f32>);
pub type NgramMap = HashMap<String, NgramBag>;

pub struct NGram {
    pub ngram_maps: Vec<NgramMap>,
    pub max_grams: i8
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

    fn train_gram_vector(&self, type_name: &String, input_data: &Vec<String>, num_grams: i8) -> NgramBag {
        let gram_groups = input_data.iter()
            .flat_map(|s| NGram::create_grams(s, num_grams as usize))
            .sorted()
            .group_by(|s| String::from(s.to_owned()));

        let grams = gram_groups
            .into_iter()
            .map(|(wd, grp)| (wd, grp.count()))
            .collect_vec();
        
        let g_index = (num_grams - 1) as usize;
        let new_wv = (0 as usize, HashMap::new());
        let o_wv = if g_index >= self.ngram_maps.len() 
            { Some(&new_wv) } else 
            { self.ngram_maps.index((num_grams - 1) as usize).get(type_name) };

        let (current_total, mut wv) = if o_wv.is_some() 
            { o_wv.unwrap().clone() } 
            else { (0 as usize, HashMap::new() as HashMap<String, f32>) };

        let input_length = input_data.len();
        let total_inputs = input_length + current_total;
        for (wd, input_count) in grams {
            if wd.eq("") {continue;}
            let o_current_prob = wv.get(&wd);
            let current_prob = if o_current_prob.is_some()
                { o_current_prob.unwrap().clone() }
                else { 0 as f32 };

            let current_count = f32::ceil(current_prob * current_total as f32) as usize;
            let total_count = input_count + current_count;

            let prob = total_count as f32 / total_inputs as f32;
            wv.insert(String::from(wd), prob);
        }
        (total_inputs, wv)
    }


    pub fn new(input_data: &Vec<InputTup>, max_grams: i8) -> NGram {
        let mut bow = NGram { ngram_maps: Vec::new(), max_grams };
        bow.train(input_data);
        bow
    }
    
    pub fn train(&mut self, input_data: &Vec<InputTup>) {
        let input_groups = input_data.iter()
            .filter(|tup| tup.0 != "")
            .sorted_by(|tup1, tup2| tup1.0.cmp(&tup2.0))
            .group_by(|&tup| tup.0.to_owned());

        let mut input_group_vec = Vec::new();
        for (name, g) in input_groups.into_iter() {
            let wv_input = g.map(|tup| tup.1.to_owned()).collect_vec();
            input_group_vec.push((name, wv_input));
        }
        
        for i in 1..(self.max_grams+1) {
            let mut bm = HashMap::new();
            for (key, wv_input) in &input_group_vec {
                bm.insert(key.clone(), self.train_gram_vector(&key, &wv_input, i));
            }
            self.ngram_maps.push(bm);
        }
    }

    fn test_gram(bow: &NgramMap, gram: &String) -> String {
        let mut best_prob: (String, f32) = (String::from(""), 0.0);
        for (type_name, (_, g_map)) in bow.into_iter() {
            let m_prob = g_map.get(gram);
            if m_prob.is_none() {
                continue;
            }
            let prob = m_prob.expect("");
            if *prob > best_prob.1 {
                best_prob = (type_name.to_owned(), prob.to_owned());
            }
        }
        String::from(best_prob.0)
    }

    pub fn test_sentence_static(bow: &Vec<NgramMap>, sentence: &String) -> String {
        let mut totals_hm: HashMap<String, i32> = HashMap::new();
        let mut found_words: Vec<String> = Vec::new();
        for i in (1..(bow.len()+1)).rev() {
            for gram in NGram::create_grams(sentence, i) {
                let best_type = NGram::test_gram(bow.index(i-1), &gram);
                if best_type.eq("") { continue; }
                let gram_words = gram.split(" ").collect_vec();
                let first_word = gram_words.index(0);
                let o_found_word = found_words.clone().into_iter().find(|wd| wd.eq(first_word));
                if o_found_word.is_some() {
                    continue;
                } else {
                    found_words.push(String::from(*first_word));
                }
                let m_total = totals_hm.get(&best_type);
                let total: i32 = if m_total.is_none() { 1 } else { m_total.unwrap() + 1 };
                totals_hm.insert(best_type, total);
            }
        }
    
        let mut best_type = (String::from(""), 0);
        for (type_name, total) in totals_hm.into_iter() {
            if total > best_type.1 {
                best_type = (type_name, total)
            }
        }

        if best_type.0.eq("") {
            best_type.0 = String::from("Inconclusive");
        }
    
        best_type.0
    }

    pub fn test_sentence(&self, sentence: &String) -> String {
        NGram::test_sentence_static(&self.ngram_maps, sentence)
    }

    pub fn validate(gram_maps: &Vec<NgramMap>, input: &Vec<InputTup>) -> f32 {
        let num_inputs = input.len();
        let f_thread = |c_maps: Vec<NgramMap>, chunk: &Vec<(String, String)>| -> Vec<bool> {
            let mut correct_vec = Vec::new();
            for (tweet_type, sentence) in chunk {
                correct_vec.push(&NGram::test_sentence_static(&c_maps, sentence) == tweet_type)
            }
            correct_vec
        };
        
        let results = multi_thread_process_list(input, gram_maps.clone(), 16, f_thread, None);
        let num_correct = results.into_iter().filter(|b| *b).collect_vec().len() as i32;
        
        num_correct as f32 / num_inputs as f32
    }
}