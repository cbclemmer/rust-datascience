use itertools::Itertools;
use rand::Rng;
use std::ops::Index;
use std::time::Instant;

use crate::util::{get_percent, multi_thread_process_list};
use crate::types::InputTup;
use crate::n_gram::{NGram, NgramMap};
use crate::n_gram::config::*;

impl NGram {
    // TODO: decrease probability until there are no removed words, then increase
    // fn prune_probability(&self, input: &Vec<InputTup>, o_initial_accuracy: &Option<f32>, config: PruneProbabilityConfig) -> (f32, BagMap) {
    //     println!("\n\n\nOptimizing by minimum probability");
    //     println!("Testing accuracy...");
    //     let initial_accuracy = if o_initial_accuracy.is_none()
    //         { NGram::validate(&self.bags, self.num_grams, input) }
    //         else { o_initial_accuracy.expect("ERR") };

    //     println!("Accuracy: {}%", f32::ceil(initial_accuracy * 100 as f32));

    //     let mut min_prob = config.starting_probability;
    //     let mut current_accuracy: f32 = 0 as f32;
    //     let mut last_accuracy: f32;
    //     let mut ret_bags = self.bags.clone();
    //     loop {
    //         println!("\nPruning...");
    //         println!("Min probability: {}%", min_prob * 100 as f32);
    //         let mut tmp_bags = ret_bags.clone();
    //         let mut removed_words = 0;
    //         for (bag_name, (total_inputs, mut bag_map)) in tmp_bags.clone() {
    //             for (word, prob) in bag_map.clone() {
    //                 if prob < min_prob {
    //                     removed_words = removed_words + 1;
    //                     bag_map.remove(&word);
    //                 }
    //             }
    //             tmp_bags.insert(bag_name, (total_inputs, bag_map));
    //         }
    //         println!("Removed Words: {}", removed_words);
    //         last_accuracy = current_accuracy;
    //         println!("\nTesting accuracy...");
    //         current_accuracy = NGram::validate(&tmp_bags, self.num_grams, input);
    //         println!("Accuracy: {}%", get_percent(&current_accuracy));
    //         if current_accuracy > initial_accuracy - config.max_accuracy_reduction  as f32 {
    //             println!("\nAccuracy target met!");
    //             ret_bags = tmp_bags;
    //             min_prob = min_prob * config.probability_multiplyer;
    //         } else {
    //             min_prob = min_prob / config.probability_multiplyer;
    //             break;
    //         }
    //     }

    //     println!("\nPruning by probability complete");
    //     println!("Min probability: {}%\nAccuracy: {}%\n\n\n", &min_prob * 100 as f32, get_percent(&last_accuracy));
    //     (last_accuracy, ret_bags)
    // }

    fn get_words_in_maps(bags: &Vec<NgramMap>) -> Vec<String> {
        let clone = bags.clone();
        let word_groups = clone
            .index(0) // 1 gram
            .into_iter()
            .flat_map(|(_, (_, bag))| bag.into_iter())
            .sorted_by(|(wd1, _), (wd2, _)| wd1.cmp(wd2))
            .group_by(|(wd, _)| String::from(*wd));
    
        word_groups
            .into_iter()
            .map(|(wd, _)| wd)
            .dedup()
            .collect_vec()
    }

    // fn prune_similarity(&self, max_deviation: &f32) -> BagMap {
    //     let words = NGram::get_words_in_bags(&self.bags);

    //     let mut remove_words = Vec::new();
    //     println!("Words: {}", words.clone().into_iter().collect_vec().len());
    //     for word in &words {
    //         // Get all probabilies for this word in every bag
    //         let mut probabilies: Vec<(&String, &f32)> = Vec::new();

    //         for (bag_name, (_, bag)) in &self.bags {
    //             let o_prob = bag.get(word);
    //             if o_prob.is_none() {
    //                 continue;
    //             }
    //             probabilies.push((bag_name, o_prob.expect("ERR")));
    //         }

    //         // Test each probability for a deviation from all other bags
    //         let mut remove = true;
    //         for (_, prob) in &probabilies {
    //             for (_, prob2) in &probabilies {
    //                 let deviation = f32::abs(**prob - **prob2);
    //                 // if any item has a deviation from the others greater than the max deviation
    //                 // remove it from the bags
    //                 if deviation > *max_deviation {
    //                     remove = false;
    //                 }
    //             }
    //         }

    //         if remove {
    //             remove_words.push(word);
    //         }
    //     }

    //     let mut ret_bags = self.bags.clone();
    //     for (bag_name, (size, mut bag)) in self.bags.clone() {
    //         for word in &remove_words {
    //             bag.remove(*word);
    //         }
    //         ret_bags.insert(bag_name.to_owned(), (size, bag));
    //     }
    //     remove_words.dedup();
    //     println!("Removed words: {}", remove_words.len());
    //     ret_bags
    // }

    // fn prune_similarity_loop(&self, input: &Vec<InputTup>, o_initial_accuracy: &Option<f32>, config: PruneSimilarityConfig) -> (f32, BagMap) {
    //     println!("\n\n\nOptimizing by maximum deviation");
    //     let mut max_deviation = config.starting_deviation;
    //     let mut last_accuracy: f32;
    //     println!("Testing Accuracy...");
    //     let initial_accuracy = if o_initial_accuracy.is_none()
    //         { NGram::validate(&self.bags, self.num_grams, input) }
    //         else { o_initial_accuracy.expect("ERR") };

    //     let mut new_accuracy = initial_accuracy;
    //     println!("Accuracy: {}%", get_percent(&initial_accuracy));
    //     let mut ret_bags = self.bags.clone();
    //     let mut found_low = false;
    //     loop {
    //         println!("\nPruning...");
    //         println!("Max deviation: {}", max_deviation);
    //         let tmp_bags = self.prune_similarity(&max_deviation);
    //         last_accuracy = new_accuracy;

    //         println!("\nTesting Accuracy...");
    //         new_accuracy = NGram::validate(&tmp_bags, self.num_grams, input);
    //         println!("Accuracy: {}%", get_percent(&new_accuracy));
    //         if new_accuracy < initial_accuracy - config.max_accuracy_reduction as f32 && !found_low {
    //             println!("Still finding minimum deviation");
    //             max_deviation = max_deviation / config.probability_multiplyer;
    //             continue;
    //         }

    //         if new_accuracy > initial_accuracy - config.max_accuracy_reduction as f32 && !found_low {
    //             found_low = true;
    //         }

    //         if new_accuracy > initial_accuracy - config.max_accuracy_reduction {
    //             max_deviation = max_deviation * config.probability_multiplyer;
    //             ret_bags = tmp_bags;
    //         } else {
    //             max_deviation = max_deviation / config.probability_multiplyer;
    //             break;
    //         }
    //     }
    //     println!("\nOptimizing by similarity complete");
    //     println!("Optimized to max deviation: {} with accuracy: {}%\n\n", max_deviation * 100 as f32, get_percent(&last_accuracy));
    //     (last_accuracy, ret_bags)
    // }

    fn prune_count(&self, config: &PruneCountConfig) -> Vec<NgramMap> {
        let mut ret_vec = Vec::new();
        let mut removed_words = Vec::new();
        for g_map in &self.ngram_maps {
            let mut ret_map = g_map.clone();
            for (type_name, (total, bag)) in g_map {
                let mut map_copy = bag.clone();
                for (word, prob) in bag {
                    let count = (prob * *total as f32) + config.adjust_amount;
                    if count <= config.min_count as f32 {
                        map_copy.remove(word);
                        removed_words.push(word);
                    }
                }
                ret_map.insert(type_name.clone(), (total.clone(), map_copy));
            }
            ret_vec.push(ret_map);
        }
        let word_count = NGram::get_words_in_maps(&ret_vec).len();
        removed_words.dedup();
        println!("Pruned by count, removed {} words", removed_words.len());
        println!("Words left: {}", word_count);
        ret_vec
    }

    fn randomize_inputs(gram_maps: &Vec<NgramMap>, words: &Vec<String>, config: RandomizerConfig) -> Vec<NgramMap> {
        let mut rng = rand::thread_rng();
        let num_params = config.num_params;
        let step_size = config.step_size;
        
        let word_length = words.len();
        let mut random_words: Vec<String> = Vec::new();
        for _ in 0..num_params {
            let idx = (rng.gen::<f32>() as f32 * word_length as f32) as usize;
            random_words.push(words.index(idx).clone());
        }
        
        let mut ret_val = Vec::new();
        for g_map in gram_maps {
            let mut ret_map: NgramMap = g_map.clone();
            for (type_name, (total, mut wd_hm)) in g_map.clone() {
                for wd in &random_words {
                    let o_current_prob = wd_hm.get(wd);
                    if o_current_prob.is_none() { continue; }
                    let current_prob = o_current_prob.expect("ERR");
                    let step_positive = if rng.gen::<f32>() > 0.5 as f32 { 1 as f32 } else { -1 as f32};
                    let step = step_positive * step_size;
                    wd_hm.insert(wd.to_owned(), current_prob + step);
                }
                ret_map.insert(type_name.to_owned(), (total, wd_hm));
            }
            ret_val.push(ret_map);
        }
        
        ret_val
    }

    fn learn_randomizer_loop(&self, input: &Vec<InputTup>, o_initial_accuracy: &Option<f32>, config: RandomizerConfig) -> (f32, Vec<NgramMap>) {
        println!("\n\n\nStarting randomizer");
        println!("\nTesting Accuracy...");
        let initial_accuracy = if o_initial_accuracy.is_none()
            { NGram::validate(&self.ngram_maps, input) }
            else { o_initial_accuracy.expect("ERR") };
        
        let mut current_accuracy = initial_accuracy;
        let mut ret_maps = self.ngram_maps.clone();
        let mut improve_timer = Instant::now();
        let mut last_improvement_iterations = 0;
        for i in 0..config.iterations {
            if i % 100 == 0 {
                println!("{} iterations", i);
            }
            let f_thread = |((input_ctx, words, config), maps): ((Vec<InputTup>, Vec<String>, RandomizerConfig), Vec<NgramMap>), _: &Vec<i32>| -> Vec<(f32, Vec<NgramMap>)> {
                let new_maps = NGram::randomize_inputs(&maps, &words, config);
                let new_accuracy = NGram::validate(&new_maps, &input_ctx);
                vec![(new_accuracy, new_maps)]
            };
            
            let iter_list = 0..config.iterations;
            let words = NGram::get_words_in_maps(&self.ngram_maps);
            let ret_vec = multi_thread_process_list(&iter_list.collect_vec(), ((input.clone(), words, config.clone()), ret_maps.clone()), 12, f_thread, None);
            for (new_accuracy, new_maps) in ret_vec {
                if new_accuracy > current_accuracy {
                    println!("Accuracy: {}%", get_percent(&new_accuracy));
                    println!("Time taken: {:?}", improve_timer.elapsed());
                    println!("Iterations needed: {}", last_improvement_iterations);
                    last_improvement_iterations = 0;
                    improve_timer = Instant::now();
                    current_accuracy = new_accuracy;
                    ret_maps = new_maps;
                }
            }
            last_improvement_iterations = last_improvement_iterations + 1;
        }

        (current_accuracy, ret_maps)
    }

    pub fn learn(&mut self, input: &Vec<InputTup>, o_learn_config: Option<LearnConfig>) {
        let mut learn_config: LearnConfig;

        if o_learn_config.is_some() {
            learn_config = o_learn_config.expect("Unwrap error");
            if learn_config.prune_probability.is_none() {
                learn_config.prune_probability = Some(PruneProbabilityConfig::default());
            }
            if learn_config.prune_similarity.is_none() {
                learn_config.prune_similarity = Some(PruneSimilarityConfig::default());
            }
            if learn_config.randomizer.is_none() {
                learn_config.randomizer = Some(RandomizerConfig::default());
            }
            if learn_config.prune_count.is_none() {
                learn_config.prune_count = Some(PruneCountConfig::default());
            }
        } else {
            learn_config = LearnConfig {
                prune_probability: Some(PruneProbabilityConfig::default()),
                prune_similarity: Some(PruneSimilarityConfig::default()),
                randomizer: Some(RandomizerConfig::default()),
                prune_count: Some(PruneCountConfig::default()),
                prune_selection: PruneSelectionConfig {
                    probability: false,
                    similarity: false,
                    count: false,
                    randomizer: false
                }
            }
        }

        println!("\nRunning learning procedure");
        println!("Num inputs: {}", input.len());
        
        // find minimum probability for words that still make the outcome reasonably accurate
        let accuracy = None;
        // if learn_config.prune_selection.similarity {
        //     let (tmp_accuracy, bags) = self.prune_similarity_loop(&input, &accuracy, learn_config.prune_similarity.expect("config err"));
        //     accuracy = Some(tmp_accuracy);
        //     self.bags = bags;
        // }
        
        // if learn_config.prune_selection.probability {
        //     let (tmp_accuracy, bags) = self.prune_probability(&input, &accuracy, learn_config.prune_probability.expect("config err"));
        //     accuracy = Some(tmp_accuracy);
        //     self.bags = bags;
        // }

        if learn_config.prune_selection.probability {
            let (_, maps) = self.learn_randomizer_loop(input, &accuracy, learn_config.randomizer.expect("config err"));
            self.ngram_maps = maps;
        }

        if learn_config.prune_selection.count {
            self.ngram_maps = self.prune_count(&learn_config.prune_count.expect("config err"));
        }
    }
}