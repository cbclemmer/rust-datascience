use std::ops::Index;
use csv::Reader;
use itertools::Itertools;
use std::fs;
use std::thread;
use std::sync::mpsc;

use bag_of_words::BagOfWords;
use bag_of_words::InputTup;
use bag_of_words::BagMap;

#[link(name = "lib", kind = "static")]
extern "C" {
    fn add(a: i32, b: i32) -> i32;
}

fn multi_thread_process_list<T1, T2, T3> (
    list: Vec<T1>, 
    context: T3,
    num_threads: i8, 
    f_thread: fn(T3, Vec<T1>) -> Vec<T2>, 
    f_return: fn(Vec<T2>, i32, usize) -> ()
) -> Vec<T2> 
    where 
        T1: 'static + Send + Clone, 
        T2: 'static + Send + Clone,
        T3: 'static + Send + Clone
    {
    let (tx, rx) = mpsc::channel::<Vec<T2>>();

    let list_size = list.len();
    let num_in_chunk = f32::ceil(list_size as f32 / num_threads as f32) as i32;
    println!("{} total records, {} in chunk", list_size, num_in_chunk);
    
    let mut threads_spawned: i8 = 0;
    let iter = list.into_iter();

    for i in 0..num_threads {
        let ctx = tx.clone();

        threads_spawned += 1;
        
        let c = context.clone();
        let list_chunk = iter.clone().skip((i as i32 * num_in_chunk) as usize).take(num_in_chunk as usize).collect_vec();
        thread::spawn(move || {
            ctx.send(f_thread(c, list_chunk)).expect("Error sending data from thread");
        });
    }

    let mut ret_val: Vec<T2> = Vec::new();
    for i in 1..threads_spawned+1 {
        let mut rec = rx.recv().expect("Recieved from thread error");
        ret_val.append(&mut rec);
        f_return(ret_val.clone(), i as i32 * num_in_chunk, list_size);
    }
    ret_val
}

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
    let blacklist_words = fs::read_to_string("data/bow_blacklist.txt")
        .expect("Error reading blacklist file")
        .split("\n")
        .map(|s| strip_special_characters(String::from(s)))
        .collect_vec();

    let file_contents = fs::read_to_string(file_path)
        .expect("error reading input file");

    let mut rdr = Reader::from_reader(file_contents.as_bytes());

    let records = rdr.records()
        .into_iter()
        .map(|r| r.expect("Error parsing record"))
        .map(|r| (String::from(r.index(2)), String::from(r.index(3))))
        .collect_vec();

    let f_thread = |bl_words: Vec<String>, chunk: Vec<(String, String)>| -> Vec<(String, String)> {
        let mut ret = Vec::new();
        for (sentiment, tweet) in chunk {
            let pair = (sentiment, clean_words(tweet, bl_words.clone()));
            ret.push(pair);
        }
        ret
    };

    let f_return = |_: Vec<(String, String)>, _: i32, _: usize| { };

    multi_thread_process_list(records, blacklist_words, 16, f_thread, f_return)
}

fn main() {
    let c = unsafe { add(1, 2) };
    println!("{}", c);
    return;

    println!("Getting training data");
    let training_data = get_input_data(String::from("data/twitter_training.csv"));
    println!("Training data");
    let bow = BagOfWords::new(&training_data, 0.0001);

    println!("Getting validation data");
    let validation_data = get_input_data(String::from("data/twitter_validation.csv"));
    let num_validation_tweets = validation_data.len();

    let f_thread = |ctx: BagMap, chunk: Vec<(String, String)>| -> Vec<bool> {
        let mut correct_vec = Vec::new();
        for (tweet_type, tweet) in chunk {
            let bc = ctx.clone();
            correct_vec.push(BagOfWords::test_sentence_static(bc, tweet) == tweet_type)
        }
        correct_vec
    };

    let f_return = |ret: Vec<bool>, num_iterated: i32, list_size: usize| {
        let num_correct = ret.into_iter().filter(|b| *b).collect_vec().len() as i32;
        let result: f32 = f32::ceil(num_correct as f32 / num_iterated as f32 * 100.0);
        let per_complete = f32::ceil(num_iterated as f32 / list_size as f32 * 100.0);
        println!("{}% correct, {}% complete, {} processed records", result, per_complete, num_iterated);
    };

    let results = multi_thread_process_list(validation_data, bow.bags, 16, f_thread, f_return);
    let num_correct = results.into_iter().filter(|b| *b).collect_vec().len() as i32;

    let result: f32 = num_correct as f32 / num_validation_tweets as f32;
    println!("Final result: {}%", result * 100.0)
}