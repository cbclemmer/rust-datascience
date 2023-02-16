use itertools::Itertools;

use lib::bag_of_words::BagMap;
use lib::bag_of_words::BagOfWords;
use lib::markov_chain::{MarkovChain, StateMap};
use lib::util::{get_input_data_csv, get_markov_data, multi_thread_process_list};

// fn validate_bow() {
//     println!("Getting training data");
    
//     let stop_word_file = String::from("data/stop_words.txt");
//     let training_data = get_input_data_csv(String::from("data/twitter_training.csv"), &stop_word_file);
//     println!("Training data");
//     let bow = BagOfWords::new(&training_data, &0.0001);
    
//     println!("Getting validation data");
//     let validation_data = get_input_data_csv(String::from("data/twitter_validation.csv"), &stop_word_file);
//     let num_validation_tweets = validation_data.len();
    
//     let f_thread = |ctx: BagMap, chunk: Vec<(String, String)>| -> Vec<bool> {
//         let mut correct_vec = Vec::new();
//         for (tweet_type, tweet) in chunk {
//             let bc = ctx.clone();
//             correct_vec.push(BagOfWords::test_sentence_static(bc, tweet) == tweet_type)
//         }
//         correct_vec
//     };
    
//     let f_return = |ret: Vec<bool>, num_iterated: i32, list_size: usize| {
//         let num_correct = ret.into_iter().filter(|b| *b).collect_vec().len() as i32;
//         let result: f32 = f32::ceil(num_correct as f32 / num_iterated as f32 * 100.0);
//         let per_complete = f32::ceil(num_iterated as f32 / list_size as f32 * 100.0);
//         println!("{}% correct, {}% complete, {} processed records", result, per_complete, num_iterated);
//     };
    
//     let results = multi_thread_process_list(validation_data, bow.bags, 16, f_thread, f_return);
//     let num_correct = results.into_iter().filter(|b| *b).collect_vec().len() as i32;
    
//     let result: f32 = num_correct as f32 / num_validation_tweets as f32;
//     println!("Final result: {}%", result * 100.0)
// }

// fn validate_mc() {
//     println!("Getting training data");
//     let stop_word_file = String::from("data/stop_words.txt");
//     let training_data = get_markov_data(String::from("data/bee_movie.txt"), &stop_word_file);
//     let num_training_data = training_data.len();
//     println!("Training markov chain");
//     let mc = MarkovChain::new(training_data.clone());
//     println!("Validating chain");
    
//     let f_thread = |ctx: StateMap, chunk: Vec<(String, String)>| -> Vec<bool> {
//         let mut correct_vec = Vec::new();
//         for (current_word, next_word) in chunk {
//             let prediction = MarkovChain::predict(ctx.clone(), current_word);
//             correct_vec.push(prediction.eq(&next_word));
//         }
//         correct_vec
//     };

//     let f_return = |_, _, _| { };

//     let results = multi_thread_process_list(training_data, mc.states, 16, f_thread, f_return);

//     // println!("Correct: {}%", f32::ceil((num_correct as f32 / training_data.len() as f32) * 100 as f32))
//     let num_correct = results.into_iter().filter(|b| *b).collect_vec().len() as i32;
    
//     let result: f32 = num_correct as f32 / num_training_data as f32;
//     println!("Final result: {}%", result * 100.0)
// }

fn write() {
    println!("Getting training data");
    
    let stop_word_file = String::from("data/stop_words.txt");
    let training_data = get_input_data_csv(String::from("data/twitter_training.csv"), &stop_word_file);
    println!("Training data");
    let mut bow = BagOfWords::new(&training_data);
    
    println!("Getting validation data");
    let validation_data = get_input_data_csv(String::from("data/twitter_validation.csv"), &stop_word_file);
    
    // bow.learn(&validation_data);
    bow.save("data/bow.dat")
}

fn read() {
    let bow = BagOfWords::load("data/bow.dat");
    let stop_word_file = String::from("data/stop_words.txt");
    let validation_data = get_input_data_csv(String::from("data/twitter_validation.csv"), &stop_word_file);
    let accuracy = BagOfWords::test(&bow.bags, &validation_data);
    println!("Accuracy: {}", accuracy);
}

fn main() {
    // validate_bow()
    // validate_mc();
    write();
    read();
}
