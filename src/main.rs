use std::ops::Index;
use csv::Reader;
use itertools::Itertools;
use std::fs;

use bag_of_words::BagOfWords;
use bag_of_words::InputTup;

fn strip_special_characters(input: String) -> String {
    let special_characters = "!@#$%^&*()_+-=[]{}\\|;':\",./<>?0123456789";
    input
        .to_lowercase()
        .chars()
        .filter(|c| !special_characters.contains(*c))
        .collect()
}

fn clean_words(input: String, bl_words: Vec<String>, wl_words: Vec<String>) -> String {
    let stripped_input = strip_special_characters(input);
    let words = stripped_input.split(" ");

    let mut new_s = String::from("");
    for wd in words {
        if bl_words.contains(&String::from(wd)) {
            continue;
        }
        if !wl_words.contains(&String::from(wd)) {
            continue;
        }
        new_s.push(' ');
        new_s.push_str(wd);
    }
    String::from(new_s)
}

fn get_word_list(file_path: String) -> Vec<String> {
    let file_data = fs::read_to_string(file_path).expect("Error reading file");
    
    let mut ret_vec: Vec<String> = Vec::new();
    for wd in file_data.split("\n").flat_map(|s| s.split(",")) {
        ret_vec.push(String::from(wd).to_lowercase());
    }

    ret_vec
}

fn get_input_data(file_path: String) -> Vec<InputTup> {
    let mut training_data: Vec<InputTup> = Vec::new();
    
    let blacklist_words = fs::read_to_string("data/bow_blacklist.txt")
        .expect("Error reading blacklist file")
        .split("\n")
        .map(|s| strip_special_characters(String::from(s)))
        .collect_vec();

    let pos_folder = String::from("data/parts_of_speech/");

    let mut adjectives_file = pos_folder.clone();
    adjectives_file.push_str("adjective.txt");

    let mut adverbs_file = pos_folder.clone();
    adverbs_file.push_str("adverb.txt");

    let mut nouns_file = pos_folder.clone();
    nouns_file.push_str("noun.txt");

    let mut verbs_file = pos_folder.clone();
    verbs_file.push_str("verb.txt");

    let mut whitelist_words = get_word_list(adjectives_file);
    whitelist_words.append(&mut get_word_list(adverbs_file));
    whitelist_words.append(&mut get_word_list(nouns_file));
    whitelist_words.append(&mut get_word_list(verbs_file));
    
    let file_contents = fs::read_to_string(file_path)
        .expect("error reading input file");

    let mut rdr = Reader::from_reader(file_contents.as_bytes());

    for result in rdr.records() {
        let r = result.ok().expect("Error parsing record");
        let sentiment = String::from(r.index(2));
        let tweet = String::from(r.index(3));
        let pair = (sentiment, clean_words(tweet, blacklist_words.clone(), whitelist_words.clone()));
        training_data.push(pair);
    }
    training_data
}

fn main() {
    println!("Getting training data");
    let training_data = get_input_data(String::from("data/twitter_training.csv"));
    println!("Training data");
    let bow = BagOfWords::new(&training_data);

    println!("Getting validation data");
    let validation_data = get_input_data(String::from("data/twitter_validation.csv"));

    let num_validation_tweets = validation_data.len();
    let mut num_correct = 0;
    let mut num_iterated = 0;
    for (tweet_type, tweet) in validation_data {
        if bow.test_sentence(tweet) == tweet_type {
            num_correct += 1;
        }
        num_iterated += 1;
        let result: f32 = num_correct as f32 / num_iterated as f32;
        println!("Correct {}%", result * 100.0)
    }

    let result: f32 = num_correct as f32 / num_validation_tweets as f32;
    println!("Final result: {}%", result * 100.0)
}


// let mut top_keys: Vec<(&String, &f32)> = Vec::new();
    // for tup in bow.bags.get("Positive").expect("err") {
    //     top_keys.push(tup);
    // }
    // // let a  = top_keys.iter().sorted_by(|(_, prob1), (_, prob2)| (**prob1);
    // let a = bow.bags.get("Positive")
    //     .expect("err")
    //     .iter()
    //     .sorted_by(|(_, prob1), (_, prob2)| prob1.partial_cmp(prob2).expect("err"))
    //     // .rev()
    //     .take(30)
    //     .collect_vec();

    // for b in a {
    //     println!("{}: {}", b.0, b.1);
    // }