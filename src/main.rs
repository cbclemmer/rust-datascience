use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::ops::Index;

use csv::Reader;
use itertools::Itertools;

use lib::markov_chain::MarkovChain;
use lib::n_gram::NGram;
use lib::util::clean_words;
use lib::util::get_stop_words;
use lib::util::get_input_data_csv;
use regex::Regex;

// fn write() {
//     println!("Getting training data");
    
//     let stop_word_file = String::from("data/stop_words.txt");
//     let training_data = get_input_data_csv("data/twitter_training.csv", &stop_word_file);
//     println!("Training data");
//     let mut bow = NGram::new(&training_data, 3);
    
//     println!("Getting validation data");
//     let validation_data = get_input_data_csv("data/twitter_validation.csv", &stop_word_file);
    
//     let bow_config = NGram::read_config("data/bow_config.json");
//     bow.learn(&validation_data, Some(bow_config));
//     let prob = NGram::validate(&bow.ngram_maps, &validation_data);
//     println!("Accuracy: {}", prob * 100 as f32);
//     bow.save("data/bow.dat")
// }

// fn read() {
//     let bow = NGram::load("data/bow.dat");
//     let stop_word_file = String::from("data/stop_words.txt");
//     let validation_data = get_input_data_csv("data/twitter_validation.csv", &stop_word_file);
//     let accuracy = NGram::validate(&bow.ngram_maps, &validation_data);
//     println!("Accuracy: {}", accuracy);
// }

// fn remove_byte_codes(s: &str) -> String {
//     let reg = Regex::new(r"(\\x[a-f0-9][a-f0-9])").unwrap();
//     reg.replace_all(s, "").to_string()
// }

// fn parse() {
//     let mut file = File::open("data/Task-1 tweets_1000.csv").expect("Creating file object error");
//     let mut file_contents = String::new();
//     file.read_to_string(&mut file_contents).expect("Reading file error");
//     if file_contents.eq("") { panic!("Loading n-gram model: File empty") }

//     let stop_words = get_stop_words("data/stop_words.txt");
//     let input = file_contents
//         .split("\n")
//         .map(|s| remove_byte_codes(s))
//         .map(|s| clean_words(&String::from(s), &stop_words))
//         .collect_vec();
//     NGram::parse("data/bow.dat", input, "data/parsed.csv");
// }

fn main() {
    // let mut mc = MarkovChain::new();
    // mc.states = MarkovChain::train_file("data/wikisent2.txt", "data/popular_words.txt");
    // mc.save("data/mc.dat");
    let mc = MarkovChain::load("data/mc.dat");
    
}


// let in_file = fs::read_to_string("data/unigram_freq.csv").expect("Error reading input");
// let mut rdr = Reader::from_reader(in_file.as_bytes());
// let mut out_file = File::create("data/popular_words.txt").expect("Error creating out file");
// let words = rdr.records()
//     .map(|r| r.expect("Error parsing record"))
//     .map(|r| (String::from(r.index(0)), r.index(1).parse::<i64>().unwrap()))
//     .sorted_by(|(_, n1), (_, n2)| n1.cmp(&n2))
//     .map(|(wd, _)| format!("{}\n", wd))
//     .rev()
//     .take(10000)
//     .collect_vec();

// let mut i = 0;
// for wd in words {
//     if i % 100 == 0 {
//         println!("Wrote {} lines", i);
//     }
//     out_file.write(wd.as_bytes()).expect("Error writing line to file");
//     i = i + 1;
// }
