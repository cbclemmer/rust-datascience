use itertools::Itertools;
use std::{collections::HashMap, ops::Index, io::{Write, Read}};
use rand::Rng;
use std::time::Instant;
use std::fs::File;

use crate::util::{InputTup, multi_thread_process_list, get_percent, reduce};

// (total num words, word -> probability)
// probability = num times word appears / total num words
pub type WordBag = (usize, HashMap<String, f32>);
pub type BagMap = HashMap<String, WordBag>;

pub struct PruneProbabilityConfig {
    // Starts at this probability
    pub starting_probability: f32,
    // Will not reduce accuracy beyond the original accuracy - this number
    pub max_accuracy_reduction: f32,
    // Will multiply the minimum probability by this number every iteration
    pub probability_multiplyer: f32,
}

pub struct PruneSimilarityConfig {
    // Starts at this deviation
    pub starting_deviation: f32,
    // Will not reduce accuracy beyond the original accuracy - this number
    pub max_accuracy_reduction: f32,
    // Will multiply the minimum probability by this number every iteration
    pub probability_multiplyer: f32
}

#[derive(Clone)]
pub struct RandomizerConfig {
    pub num_params: i32,
    pub step_size: f32,
    pub iterations: i32
}

// impl Clone for RandomizerConfig {
//     clone() {

//     }
// }

pub struct LearnConfig {
    pub prune_probability: PruneProbabilityConfig,
    pub prune_similarity: PruneSimilarityConfig,
    pub randomizer: RandomizerConfig
}

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

    pub fn test(bags: &BagMap, input: &Vec<InputTup>) -> f32 {
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

    pub fn save(&self, file_name: &str) {
        let mut file_data = String::from("");
        for (bag_name, (total, bag)) in self.bags.clone() {
            // Positive,123|
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

    pub fn load(file_name: &str) -> BagOfWords {
        let mut file = File::open(file_name).expect("Creating file object error");
        let mut file_contents = String::new();
        file.read_to_string(&mut file_contents).expect("Reading file error");
        if file_contents.eq("") { panic!("Loading bag of words: File empty") }
        let bags = file_contents.split("\n");
        let mut bm = BagMap::new();
        for bag in bags {
            if bag.eq("") { continue; }
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
        BagOfWords { bags: bm }
    }

    // TODO: decrease probability until there are no removed words, then increase
    fn prune_probability(&self, input: &Vec<InputTup>, o_initial_accuracy: Option<f32>, config: PruneProbabilityConfig) -> (f32, BagMap) {
        println!("\n\n\nOptimizing by minimum probability");
        println!("Testing accuracy...");
        let initial_accuracy = if o_initial_accuracy.is_none()
            { BagOfWords::test(&self.bags, input) }
            else { o_initial_accuracy.expect("ERR") };

        println!("Accuracy: {}%", f32::ceil(initial_accuracy * 100 as f32));

        let mut min_prob = config.starting_probability;
        let mut current_accuracy: f32 = 0 as f32;
        let mut last_accuracy: f32;
        let mut ret_bags = self.bags.clone();
        loop {
            println!("\nPruning...");
            println!("Min probability: {}%", min_prob * 100 as f32);
            let mut tmp_bags = ret_bags.clone();
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
            current_accuracy = BagOfWords::test(&tmp_bags, input);
            println!("Accuracy: {}%", get_percent(&current_accuracy));
            if current_accuracy > initial_accuracy - config.max_accuracy_reduction  as f32 {
                println!("\nAccuracy target met!");
                ret_bags = tmp_bags;
                min_prob = min_prob * config.probability_multiplyer;
            } else {
                min_prob = min_prob / config.probability_multiplyer;
                break;
            }
        }

        println!("\nPruning by probability complete");
        println!("Min probability: {}%\nAccuracy: {}%\n\n\n", &min_prob * 100 as f32, get_percent(&last_accuracy));
        (last_accuracy, ret_bags)
    }

    fn get_words_in_bags(bags: &BagMap) -> Vec<String> {
        let word_groups = bags
            .clone()
            .into_iter()
            .flat_map(|(_, (_, bag))| bag.into_iter())
            .sorted_by(|(wd1, _), (wd2, _)| wd1.cmp(wd2))
            .group_by(|(wd, _)| String::from(wd));
    
        word_groups
            .into_iter()
            .map(|(wd, _)| wd)
            .dedup()
            .collect_vec()
    }

    fn prune_similarity(&self, max_deviation: &f32) -> BagMap {
        let words = BagOfWords::get_words_in_bags(&self.bags);

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
        remove_words.dedup();
        println!("Removed words: {}", remove_words.len());
        ret_bags
    }

    fn prune_similarity_loop(&self, input: &Vec<InputTup>, o_initial_accuracy: Option<f32>, config: PruneSimilarityConfig) -> (f32, BagMap) {
        println!("\n\n\nOptimizing by maximum deviation");
        let mut max_deviation = config.starting_deviation;
        let mut last_accuracy: f32;
        println!("Testing Accuracy...");
        let initial_accuracy = if o_initial_accuracy.is_none()
            { BagOfWords::test(&self.bags, input) }
            else { o_initial_accuracy.expect("ERR") };

        let mut new_accuracy = initial_accuracy;
        println!("Accuracy: {}%", get_percent(&initial_accuracy));
        let mut ret_bags = self.bags.clone();
        let mut found_low = false;
        loop {
            println!("\nPruning...");
            println!("Max deviation: {}", max_deviation);
            let tmp_bags = self.prune_similarity(&max_deviation);
            last_accuracy = new_accuracy;

            println!("\nTesting Accuracy...");
            new_accuracy = BagOfWords::test(&tmp_bags, input);
            println!("Accuracy: {}%", get_percent(&new_accuracy));
            if new_accuracy < initial_accuracy - config.max_accuracy_reduction as f32 && !found_low {
                println!("Still finding minimum deviation");
                max_deviation = max_deviation / config.probability_multiplyer;
                continue;
            }

            if new_accuracy > initial_accuracy - config.max_accuracy_reduction as f32 && !found_low {
                found_low = true;
            }

            if new_accuracy > initial_accuracy - config.max_accuracy_reduction {
                max_deviation = max_deviation * config.probability_multiplyer;
                ret_bags = tmp_bags;
            } else {
                max_deviation = max_deviation / config.probability_multiplyer;
                break;
            }
        }
        println!("\nOptimizing by similarity complete");
        println!("Optimized to max deviation: {} with accuracy: {}%\n\n", max_deviation * 100 as f32, get_percent(&last_accuracy));
        (last_accuracy, ret_bags)
    }

    fn randomize_inputs(bags: BagMap, words: &Vec<String>, config: RandomizerConfig) -> BagMap {
        // let timer = Instant::now();
        let mut rng = rand::thread_rng();
        let num_params = config.num_params;
        let step_size = config.step_size;
        
        let word_length = words.len();
        let mut random_words: Vec<String> = Vec::new();
        for _ in 0..num_params {
            let idx = (rng.gen::<f32>() as f32 * word_length as f32) as usize;
            random_words.push(words.index(idx).clone());
        }
        
        let mut ret_bags: BagMap = bags.clone();
        for (bag_name, (total, mut wd_hm)) in bags {
            for wd in &random_words {
                let o_current_prob = wd_hm.get(wd);
                if o_current_prob.is_none() { continue; }
                let current_prob = o_current_prob.expect("ERR");
                let step_positive = if rng.gen::<f32>() > 0.5 as f32 { 1 as f32 } else { -1 as f32};
                let step = step_positive * step_size;
                wd_hm.insert(wd.to_owned(), current_prob + step);
            }
            ret_bags.insert(bag_name.to_owned(), (total, wd_hm));
        }
        // println!("Randomizer time: {:?}", timer.elapsed());
        
        ret_bags
    }

    fn learn_randomizer_loop(&self, input: &Vec<InputTup>, o_initial_accuracy: Option<f32>, config: RandomizerConfig) -> (f32, BagMap) {
        println!("\n\n\nStarting randomizer");
        println!("\nTesting Accuracy...");
        let initial_accuracy = if o_initial_accuracy.is_none()
            { BagOfWords::test(&self.bags, input) }
            else { o_initial_accuracy.expect("ERR") };
        
        let mut current_accuracy = initial_accuracy;
        let mut ret_bags = self.bags.clone();
        let mut improve_timer = Instant::now();
        let mut last_improvement_iterations = 0;
        for i in 0..config.iterations {
            if i % 100 == 0 {
                println!("{} iterations", i);
            }
            let f_thread = |((input_ctx, words, config), bags_ctx): ((Vec<InputTup>, Vec<String>, RandomizerConfig), BagMap), _: &Vec<i32>| -> Vec<(f32, BagMap)> {
                let new_bags = BagOfWords::randomize_inputs(bags_ctx, &words, config);
                let new_accuracy = BagOfWords::test(&new_bags, &input_ctx);
                vec![(new_accuracy, new_bags)]
            };
            
            let iter_list = 0..config.iterations;
            let words = BagOfWords::get_words_in_bags(&self.bags);
            let ret_vec = multi_thread_process_list(&iter_list.collect_vec(), ((input.clone(), words, config.clone()), ret_bags.clone()), 12, f_thread, None);
            for (new_accuracy, new_bags) in ret_vec {
                if new_accuracy > current_accuracy {
                    println!("Accuracy: {}%", get_percent(&new_accuracy));
                    println!("Time taken: {:?}", improve_timer.elapsed());
                    println!("Iterations needed: {}", last_improvement_iterations);
                    last_improvement_iterations = 0;
                    improve_timer = Instant::now();
                    current_accuracy = new_accuracy;
                    ret_bags = new_bags;
                }
            }
            last_improvement_iterations = last_improvement_iterations + 1;
        }

        (current_accuracy, ret_bags)
    }

    // , step_size: f32, num_iterations: i32, modify_amount: i32
    pub fn learn(&mut self, input: &Vec<InputTup>, o_learn_config: Option<LearnConfig>) {
        let learn_config: LearnConfig;
        if o_learn_config.is_some() {
            learn_config = o_learn_config.expect("Unwrap error");
        } else {
            learn_config = LearnConfig {
                prune_probability: PruneProbabilityConfig {
                    starting_probability: 0.00001,
                    max_accuracy_reduction: 0.1,
                    probability_multiplyer: 2.0
                },
                prune_similarity: PruneSimilarityConfig { 
                    starting_deviation: 0.00000001, 
                    max_accuracy_reduction: 0.1, 
                    probability_multiplyer: 2.0
                },
                randomizer: RandomizerConfig { 
                    num_params: 10, 
                    step_size: 0.001,
                    iterations: 1000
                }
            }
        }

        println!("\nRunning learning procedure");
        println!("Num inputs: {}", input.len());
        // find minimum probability for words that still make the outcome reasonably accurate
        let (accuracy, bags) = self.prune_similarity_loop(&input, None, learn_config.prune_similarity);
        self.bags = bags;
        
        let (_, bags) = self.prune_probability(&input, Some(accuracy), learn_config.prune_probability);
        self.bags = bags;

        let (_, bags) = self.learn_randomizer_loop(input, None, learn_config.randomizer);        
        self.bags = bags;
    }
}