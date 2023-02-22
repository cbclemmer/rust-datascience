use std::fs::File;
use std::io::Read;

use itertools::Itertools;

use lib::n_gram::NGram;
use lib::util::clean_words;
use lib::util::get_stop_words;
use lib::util::get_input_data_csv;

fn write() {
    println!("Getting training data");
    
    let stop_word_file = String::from("data/stop_words.txt");
    let training_data = get_input_data_csv("data/twitter_training.csv", &stop_word_file);
    println!("Training data");
    let mut bow = NGram::new(&training_data, 3);
    
    println!("Getting validation data");
    let validation_data = get_input_data_csv("data/twitter_validation.csv", &stop_word_file);
    
    let bow_config = NGram::read_config("data/bow_config.json");
    bow.learn(&validation_data, Some(bow_config));
    let prob = NGram::validate(&bow.ngram_maps, &validation_data);
    println!("Accuracy: {}", prob * 100 as f32);
    bow.save("data/bow.dat")
}

fn read() {
    let bow = NGram::load("data/bow.dat");
    let stop_word_file = String::from("data/stop_words.txt");
    let validation_data = get_input_data_csv("data/twitter_validation.csv", &stop_word_file);
    let accuracy = NGram::validate(&bow.ngram_maps, &validation_data);
    println!("Accuracy: {}", accuracy);
}

fn parse() {
    let mut file = File::open("data/Task-1 tweets_1000.csv").expect("Creating file object error");
    let mut file_contents = String::new();
    file.read_to_string(&mut file_contents).expect("Reading file error");
    if file_contents.eq("") { panic!("Loading n-gram model: File empty") }

    let stop_words = get_stop_words("data/stop_words.txt");
    let input = file_contents
        .split("\n")
        .map(|s| clean_words(&String::from(s), &stop_words))
        .collect_vec();
    NGram::parse("data/bow.dat", input, "data/parsed.csv");
}

fn main() {
    // validate_bow()
    // validate_mc();
    write();
    // read();
    // parse();
}
