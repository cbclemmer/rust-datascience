use std::ops::Index;
use csv::Reader;
use itertools::Itertools;
use std::fs;
use std::thread;
use std::sync::mpsc;

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

fn clean_words(input: String, bl_words: Vec<String>) -> String {
    let stripped_input = strip_special_characters(input);
    let words = stripped_input.split(" ");

    let mut new_s = String::from("");
    for wd in words {
        if bl_words.contains(&String::from(wd)) {
            continue;
        }
        new_s.push(' ');
        new_s.push_str(wd);
    }
    String::from(new_s)
}

fn get_input_data(file_path: String) -> Vec<InputTup> {
    let mut training_data: Vec<InputTup> = Vec::new();
    
    let blacklist_words = fs::read_to_string("data/bow_blacklist.txt")
        .expect("Error reading blacklist file")
        .split("\n")
        .map(|s| strip_special_characters(String::from(s)))
        .collect_vec();

    let file_contents = fs::read_to_string(file_path)
        .expect("error reading input file");

    let mut rdr = Reader::from_reader(file_contents.as_bytes());

    let mut num_records = 0;
    for result in rdr.records() {
        let r = result.ok().expect("Error parsing record");
        num_records += 1;
        let sentiment = String::from(r.index(2));
        let tweet = String::from(r.index(3));
        let pair = (sentiment, clean_words(tweet, blacklist_words.clone()));
        training_data.push(pair);
    }
    println!("{} csv records", num_records);
    training_data
}

fn main() {
    println!("Getting training data");
    let training_data = get_input_data(String::from("data/twitter_training.csv"));
    println!("Training data");
    let bow = BagOfWords::new(&training_data);

    println!("Getting validation data");
    let validation_data = get_input_data(String::from("data/twitter_validation.csv"));

    let (tx, rx) = mpsc::channel::<i32>();

    let num_threads = 16;
    let num_validation_tweets = validation_data.len();
    let num_in_chunk = f32::ceil(num_validation_tweets as f32 / num_threads as f32) as i32;
    println!("{} total records, {} in chunk", num_validation_tweets, num_in_chunk);
    
    let mut threads_spawned = 0;

    let iter = validation_data.into_iter();
    for i in 0..num_threads {
        threads_spawned += 1;

        let validation_chunk = iter.clone().skip((i * num_in_chunk) as usize).take(num_in_chunk as usize);
        let bow_clone = bow.bags.clone();
        let btx = tx.clone();
        thread::spawn(move || {
            let mut num_correct = 0;
            for (tweet_type, tweet) in validation_chunk {
                let bc = bow_clone.clone();
                if BagOfWords::test_sentence_static(bc, tweet) == tweet_type {
                    num_correct += 1;
                }
            }
            btx.send(num_correct).expect("Error sending data from thread");
        });
    }

    let mut num_correct = 0;
    let mut num_iterated = 0;
    for _ in 0..(threads_spawned-1) {
        let correct = rx.recv();
        num_correct += correct.expect("Thread error");
        num_iterated += num_in_chunk;
        let result: f32 = f32::ceil(num_correct as f32 / num_iterated as f32 * 100.0);
        let per_complete = f32::ceil(num_iterated as f32 / num_validation_tweets as f32 * 100.0);
        println!("{}% correct, {}% complete, {} processed records", result, per_complete, num_iterated);
    }

    let result: f32 = num_correct as f32 / num_validation_tweets as f32;
    println!("Final result: {}%", result * 100.0)
}