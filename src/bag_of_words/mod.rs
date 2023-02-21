use itertools::Itertools;
use json::parse;
use std::{collections::HashMap, io::{Write, Read}, fs::File};

pub mod config;
pub mod learn;
pub mod file;

use crate::util::{InputTup, multi_thread_process_list, reduce};

use self::config::*;

// (total num words, word -> probability)
// probability = num times word appears / total num words
pub type WordBag = (usize, HashMap<String, f32>);
pub type BagMap = HashMap<String, WordBag>;

pub struct BagOfWords {
    pub bags: BagMap
}

impl BagOfWords {
    fn train_word_vector(&mut self, bag_name: &String, input_data: &Vec<String>) {
        let word_group = input_data.iter()
            .flat_map(|s| s.split(" "))
            .sorted()
            .group_by(|s| String::from(s.to_owned()));

        let words = word_group
            .into_iter()
            .map(|(wd, grp)| (wd, grp.count()))
            .collect_vec();
        
        let o_wv = self.bags.get(bag_name);
        let (current_total, mut wv) = if o_wv.is_some() 
            { o_wv.expect("ERR").clone() } 
            else { (0 as usize, HashMap::new() as HashMap<String, f32>) };

        let input_length = input_data.len();
        let total_inputs = input_length + current_total;
        for (wd, input_count) in words {
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

    pub fn new(input_data: &Vec<InputTup>) -> BagOfWords {
        let mut bow = BagOfWords { bags: BagMap::new() };
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

    fn test_word(bow: &BagMap, word: &String) -> String {
        let mut best_prob: (String, f32) = (String::from(""), 0.0);
        for (bag_name, (_, bag)) in bow.into_iter() {
            let m_prob = bag.get(word);
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

    pub fn test_sentence_static(bow: &BagMap, sentence: &String) -> String {
        let mut totals_hm: HashMap<String, i32> = HashMap::new();
        for wd in sentence.split(" ") {
            let best_bag = BagOfWords::test_word(bow, &String::from(wd));
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
        BagOfWords::test_sentence_static(&self.bags, sentence)
    }

    pub fn validate(bags: &BagMap, input: &Vec<InputTup>) -> f32 {
        let num_inputs = input.len();
        let f_thread = |ctx: BagMap, chunk: &Vec<(String, String)>| -> Vec<bool> {
            let mut correct_vec = Vec::new();
            for (tweet_type, sentence) in chunk {
                correct_vec.push(&BagOfWords::test_sentence_static(&ctx, sentence) == tweet_type)
            }
            correct_vec
        };
        
        let results = multi_thread_process_list(input, bags.clone(), 16, f_thread, None);
        let num_correct = results.into_iter().filter(|b| *b).collect_vec().len() as i32;
        
        num_correct as f32 / num_inputs as f32
    }

    pub fn read_config(file_name: &str) -> LearnConfig {
        let mut config = LearnConfig { 
            prune_selection: PruneSelectionConfig { 
                probability: false, 
                similarity: false, 
                count: false, 
                randomizer: false 
            },
            prune_probability: None,
            prune_similarity: None,
            prune_count: None,
            randomizer: None
        };
        let mut file = File::open(file_name).expect("Creating file object error");
        let mut file_contents = String::new();
        file.read_to_string(&mut file_contents).expect("Reading file error");
        if file_contents.eq("") { panic!("Loading bag of words: File empty") }
        let json_data = parse(&file_contents).unwrap();

        let probability_s = "probability";
        if json_data.has_key(probability_s) {
            config.prune_selection.probability = true;
            config.prune_probability = Some(PruneProbabilityConfig::from_json(&json_data[probability_s]));
        }

        let similarity_s = "similarity";
        if json_data.has_key(similarity_s) {
            config.prune_selection.similarity = true;
            config.prune_similarity = Some(PruneSimilarityConfig::from_json(&json_data[similarity_s]));
        }

        let count_s = "count";
        if json_data.has_key(count_s) {
            config.prune_selection.count = true;
            config.prune_count = Some(PruneCountConfig::from_json(&json_data[count_s]));
        }

        let randomizer_s = "randomizer";
        if json_data.has_key(randomizer_s) {
            config.prune_selection.randomizer = true;
            config.randomizer = Some(RandomizerConfig::from_json(&json_data[randomizer_s]));
        }

        let selection_s = "selection";
        if json_data.has_key(selection_s) {
            config.prune_selection = PruneSelectionConfig::from_json(&json_data[selection_s]);
        }

        config
    }
}