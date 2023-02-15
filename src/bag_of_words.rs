use itertools::Itertools;
use std::collections::HashMap;
use rand::Rng;
use std::time::{Instant, Duration};

use crate::util::{InputTup, multi_thread_process_list, get_percent};

// (total num words, word -> probability)
// probability = num times word appears / total num words
pub type WordBag = (usize, HashMap<String, f32>);
pub type BagMap = HashMap<String, WordBag>;

pub struct BagOfWords {
    pub bags: BagMap
}

impl BagOfWords {
    // Find minimum and maximum probabilities of each word
    // if there is not a substantial difference between the different probabilities
    // assume that it is not useful and remove it from the bags
    // fn prune(mut self) {
        
    // }

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
        for (wd, input_count) in words.clone() {
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
        let bow = BagOfWords { bags: BagMap::new() };
        bow.train(input_data)
    }
    
    pub fn train(mut self, input_data: &Vec<InputTup>) -> BagOfWords {
        let input_groups = input_data.iter()
            .filter(|tup| tup.0 != "")
            .sorted_by(|tup1, tup2| tup1.0.cmp(&tup2.0))
            .group_by(|&tup| tup.0.to_owned());
        
        for (key, group) in &input_groups {
            let wv_input = group.map(|tup| tup.1.to_owned()).collect_vec();
            self.train_word_vector(&key, &wv_input);
        }
        
        self
    }

    fn test_word(bow: BagMap, word: &String) -> String {
        let mut best_prob: (String, f32) = (String::from(""), 0.0);
        let word_clone = &word.clone();
        for (bag_name, (_, bag)) in bow.clone().into_iter() {
            let m_prob = bag.get(word_clone);
            if m_prob.is_none() {
                continue;
            }
            let prob = m_prob.expect("");
            if *prob > best_prob.1 {
                best_prob = (bag_name, prob.to_owned());
            }
        }
        String::from(best_prob.0)
    }

    pub fn test_sentence_static(bow: BagMap, sentence: &String) -> String {
        let mut totals_hm: HashMap<String, i32> = HashMap::new();
        for wd in sentence.split(" ") {
            let best_bag = BagOfWords::test_word(bow.clone(), &String::from(wd));
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
        BagOfWords::test_sentence_static(self.bags.clone(), sentence)
    }

    pub fn test(bags: &BagMap, input: &Vec<InputTup>) -> f32 {
        let num_inputs = input.len();
        let f_thread = |ctx: BagMap, chunk: &Vec<(String, String)>| -> Vec<bool> {
            let mut correct_vec = Vec::new();
            for (tweet_type, sentence) in chunk {
                correct_vec.push(&BagOfWords::test_sentence_static(ctx.clone(), sentence) == tweet_type)
            }
            correct_vec
        };
        
        let f_return = |ret: Vec<bool>, num_iterated: i32, list_size: usize| {
            let num_correct = ret.into_iter().filter(|b| *b).collect_vec().len() as i32;
            let result: f32 = f32::ceil(num_correct as f32 / num_iterated as f32 * 100.0);
            let per_complete = f32::ceil(num_iterated as f32 / list_size as f32 * 100.0);
            // println!("{}% correct, {}% complete, {} processed records", result, per_complete, num_iterated);
        };
        
        let results = multi_thread_process_list(input, bags.clone(), 16, f_thread, f_return);
        let num_correct = results.into_iter().filter(|b| *b).collect_vec().len() as i32;
        
        num_correct as f32 / num_inputs as f32
    }

    fn prune_probability(&mut self, input: &Vec<InputTup>) {
        println!("\n\n\nOptimizing by minimum probability");
        println!("Testing accuracy...");
        let initial_accuracy = BagOfWords::test(&self.bags, input);
        println!("Accuracy: {}%", f32::ceil(initial_accuracy * 100 as f32));

        let mut min_prob = 0.00001;
        let mut current_accuracy: f32 = 0 as f32;
        let mut last_accuracy: f32;
        loop {
            println!("\nPruning...");
            println!("Min probability: {}%", min_prob * 100 as f32);
            let mut tmp_bags = self.bags.clone();
            let mut removed_words = 0;
            for (bag_name, (total_inputs, mut bag_map)) in tmp_bags.clone() {
                for (word, prob) in bag_map.clone() {
                    if prob < min_prob {
                        removed_words = removed_words + 1;
                        bag_map.remove(&word);
                    }
                }
                tmp_bags.insert(bag_name, (total_inputs, bag_map));
            }
            println!("Removed Words: {}", removed_words);
            last_accuracy = current_accuracy;
            println!("\nTesting accuracy...");
            current_accuracy = BagOfWords::test(&self.bags, input);
            println!("Accuracy: {}%", get_percent(&current_accuracy));
            if current_accuracy > initial_accuracy - 0.1  as f32 {
                println!("\nAccuracy target met!");
                self.bags = tmp_bags;
                min_prob = min_prob * 10 as f32;
            } else {
                min_prob = min_prob / 10 as f32;
                break;
            }
        }

        println!("\nPruning by probability complete");
        println!("Min probability: {}%\nAccuracy: {}%\n\n\n", &min_prob * 100 as f32, get_percent(&last_accuracy));
    }

    fn prune_similarity(&self, max_deviation: &f32) -> BagMap {
        let word_groups = self.bags
            .clone()
            .into_iter()
            .flat_map(|(_, (_, bag))| bag.into_iter())
            .sorted_by(|(wd1, _), (wd2, _)| wd1.cmp(wd2))
            .group_by(|(wd, _)| String::from(wd));

        let words = word_groups
            .into_iter()
            .map(|(wd, _)| wd)
            .dedup()
            .collect_vec();

        let mut remove_words = Vec::new();
        println!("Words: {}", words.clone().into_iter().collect_vec().len());
        for word in &words {
            // Get all probabilies for this word in every bag
            let mut probabilies: Vec<(&String, &f32)> = Vec::new();

            for (bag_name, (_, bag)) in &self.bags {
                let o_prob = bag.get(word);
                if o_prob.is_none() {
                    continue;
                }
                probabilies.push((bag_name, o_prob.expect("ERR")));
            }

            // Test each probability for a deviation from all other bags
            let mut remove = true;
            for (_, prob) in &probabilies {
                for (_, prob2) in &probabilies {
                    let deviation = f32::abs(**prob - **prob2);
                    // if any item has a deviation from the others greater than the max deviation
                    // remove it from the bags
                    if deviation > *max_deviation {
                        remove = false;
                    }
                }
            }

            if remove {
                remove_words.push(word);
            }
        }

        let mut ret_bags = self.bags.clone();
        for (bag_name, (size, mut bag)) in self.bags.clone() {
            for word in &remove_words {
                bag.remove(*word);
            }
            ret_bags.insert(bag_name.to_owned(), (size, bag));
        }

        println!("Removed words: {}", remove_words.len());
        ret_bags
    }

    fn prune_similarity_loop(&mut self, input: &Vec<InputTup>) {
        println!("\n\n\nOptimizing by maximum deviation");
        let mut max_deviation = 0.000001;
        let mut last_accuracy: f32;
        println!("Testing Accuracy...");
        let initial_accuracy = BagOfWords::test(&self.bags, input);
        let mut new_accuracy = initial_accuracy;
        println!("Accuracy: {}%", get_percent(&initial_accuracy));
        loop {
            println!("\nPruning...");
            println!("Max deviation: {}", max_deviation);
            let tmp_bags = self.prune_similarity(&max_deviation);
            last_accuracy = new_accuracy;

            println!("\nTesting Accuracy...");
            new_accuracy = BagOfWords::test(&tmp_bags, input);
            println!("Accuracy: {}%", get_percent(&new_accuracy));
            if new_accuracy > initial_accuracy - 0.1 as f32 {
                max_deviation = max_deviation * 10 as f32;
                self.bags = tmp_bags;
            } else {
                max_deviation = max_deviation / 10 as f32;
                break;
            }
        }
        println!("\nOptimizing by similarity complete");
        println!("Optimized to max deviation: {} with accuracy: {}%\n\n", max_deviation * 100 as f32, get_percent(&last_accuracy));
    }

    // , step_size: f32, num_iterations: i32, modify_amount: i32
    pub fn learn(mut self, input: &Vec<InputTup>) {
        println!("\nRunning learning procedure");
        println!("Num inputs: {}", input.len());
        // find minimum probability for words that still make the outcome reasonably accurate
        self.prune_similarity_loop(&input.clone().into_iter().take(100).collect_vec());
        self.prune_probability(&input.clone().into_iter().take(100).collect_vec());
    }
}