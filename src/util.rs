use std::ops::Index;
use csv::Reader;
use itertools::Itertools;
use std::fs;
use std::thread;
use std::sync::mpsc;

pub type InputTup = (String, String);

pub fn multi_thread_process_list<T1, T2, T3> (
    list: &Vec<T1>, 
    context: T3,
    num_threads: i8, 
    f_thread: fn(T3, &Vec<T1>) -> Vec<T2>, 
    f_return: Option<fn(Vec<T2>, i32, usize) -> ()>
) -> Vec<T2> 
    where 
        T1: 'static + Send + Clone, 
        T2: 'static + Send + Clone,
        T3: 'static + Send + Clone
    {
    let (tx, rx) = mpsc::channel::<Vec<T2>>();

    let list_size = list.len();
    let num_in_chunk = f32::ceil(list_size as f32 / num_threads as f32) as i32;
    // println!("{} total records, {} in chunk", list_size, num_in_chunk);
    
    let mut threads_spawned: i8 = 0;
    let iter = list.clone().into_iter();

    for i in 0..num_threads {
        let ctx = tx.clone();

        threads_spawned += 1;
        
        let c = context.clone();
        let list_chunk = iter.clone().skip((i as i32 * num_in_chunk) as usize).take(num_in_chunk as usize).collect_vec();
        thread::spawn(move || {
            ctx.send(f_thread(c, &list_chunk)).expect("Error sending data from thread");
        });
    }

    let mut ret_val: Vec<T2> = Vec::new();
    for i in 1..threads_spawned+1 {
        let mut rec = rx.recv().expect("Recieved from thread error");
        ret_val.append(&mut rec);
        if f_return.is_some() {
            f_return.expect("ERR F RETURN")(ret_val.clone(), i as i32 * num_in_chunk, list_size);
        }
    }
    ret_val
}

pub fn strip_special_characters(input: &String) -> String {
    let special_characters = "!@#$%^&*()_+-=[]{}\\|;':\",./<>?0123456789";
    input
        .to_lowercase()
        .chars()
        .filter(|c| !special_characters.contains(*c))
        .collect()
}

pub fn clean_words(input: &String, bl_words: &Vec<String>) -> String {
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

fn get_stop_words(file_path: &String) -> Vec<String> {
    fs::read_to_string(file_path)
        .expect("Error reading stop word file")
        .split("\n")
        .map(|s| strip_special_characters(&String::from(s)))
        .collect_vec()
}

pub fn get_input_data_csv(csv_file: String, stop_word_file: &String) -> Vec<InputTup> {
    let stop_words = get_stop_words(stop_word_file);

    let file_contents = fs::read_to_string(csv_file)
        .expect("error reading input file");

    let mut rdr = Reader::from_reader(file_contents.as_bytes());

    let records = rdr.records()
        .into_iter()
        .map(|r| r.expect("Error parsing record"))
        .map(|r| (String::from(r.index(2)), String::from(r.index(3))))
        .collect_vec();

    let f_thread = |bl_words: Vec<String>, chunk: &Vec<(String, String)>| -> Vec<(String, String)> {
        let mut ret = Vec::new();
        for (sentiment, tweet) in chunk {
            let pair = (String::from(sentiment), clean_words(tweet, &bl_words));
            ret.push(pair);
        }
        ret
    };

    multi_thread_process_list(&records, stop_words, 16, f_thread, None)
}

pub fn get_markov_data(text_file: String, stop_word_file: &String) -> Vec<InputTup> {
    let stop_words = get_stop_words(&stop_word_file);

    let file_contents = fs::read_to_string(text_file).expect("error reading input file");
    let cleaned_text = clean_words(&file_contents, &stop_words);
    let mut last_word = "";
    let mut ret: Vec<InputTup> = Vec::new();
    for word in cleaned_text.split(" ") {
        if last_word.eq(&String::from("")) { 
            last_word = word.clone();
            continue; 
        }
        ret.push((String::from(last_word), String::from(word)));
        last_word = word.clone();
    }
    ret
}

pub fn get_percent(prob: &f32) -> f32 { 
    f32::ceil(prob * 10000 as f32) / 100 as f32 
}