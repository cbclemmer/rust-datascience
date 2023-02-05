use std::ops::Index;
use csv::Reader;
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

fn get_input_data(file_path: String) -> Vec<InputTup> {
    let file_contents = fs::read_to_string(file_path)
        .expect("error reading input file");
    let mut rdr = Reader::from_reader(file_contents.as_bytes());
    let mut training_data: Vec<InputTup> = Vec::new();
    for result in rdr.records() {
        let r = result.ok().expect("Error parsing record");
        let sentiment = String::from(r.index(2));
        let tweet = String::from(r.index(3));
        let pair = (sentiment, strip_special_characters(tweet));
        training_data.push(pair);
    }
    training_data
}

fn main() {
    let training_data = get_input_data(String::from("data/twitter_training.csv"));
    let bow = BagOfWords::new(&training_data);

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


    // let test_tweet = strip_special_characters(String::from("Check out this epic streamer"));
    // let prediction = bow.test_sentence(String::from(test_tweet));
    // println!("{}", prediction);
}
