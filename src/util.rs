use std::collections::HashMap;
use std::ops::Index;
use csv::Reader;
use itertools::Itertools;
use std::fs;
use std::thread;
use std::sync::mpsc;

use crate::types::{InputTup, ValidationMap, StateTotals};

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
    let special_characters = "!@#$%^&*()_+-=[]{}\\|;':\",./<>?0123456789\n\r";
    input
        .to_lowercase()
        .chars()
        .map(|c| {
            if special_characters.contains(c) {
                ' '
            } else {
                c
            }
        })
        .collect()
}

pub fn clean_words(input: &String, bl_words: Option<&ValidationMap>, wl_words: Option<&ValidationMap>) -> String {
    let stripped_input = strip_special_characters(input);
    let words = stripped_input.split(" ");

    let mut new_s = String::from("");
    for wd in words {
        if bl_words.is_some() {
            if bl_words.unwrap().get(&String::from(wd)).is_some() {
                continue;
            }
        }
        if wl_words.is_some() {
            if !wl_words.unwrap().get(&String::from(wd)).is_some() {
                continue;
            }
        }
        new_s.push(' ');
        new_s.push_str(wd);
    }
    String::from(new_s)
}

pub fn get_optional_validation_map(file_path: Option<&str>) -> ValidationMap {
    let mut words = ValidationMap::new();
    if file_path.is_some() {
        words = get_validation_map(file_path.unwrap());
    }
    words
}

pub fn get_validation_map(file_path: &str) -> ValidationMap {
    let mut hm = HashMap::new();
    let words = fs::read_to_string(file_path)
        .expect("Error reading stop word file")
        .split("\n")
        .map(|s| strip_special_characters(&String::from(s)))
        .collect_vec();
    for wd in words {
        hm.insert(wd, true);
    }
    hm
}

pub fn get_input_data_csv(csv_file: &str, stop_word_file: Option<&str>, wl_word_file: Option<&str>) -> Vec<InputTup> {
    let stop_words = get_optional_validation_map(stop_word_file);
    let wl_words = get_optional_validation_map(wl_word_file);

    let file_contents = fs::read_to_string(csv_file)
        .expect("error reading input file");

    let mut rdr = Reader::from_reader(file_contents.as_bytes());

    let records = rdr.records()
        .into_iter()
        .map(|r| r.expect("Error parsing record"))
        .map(|r| (String::from(r.index(2)), String::from(r.index(3))))
        .collect_vec();

    let f_thread = |(bl_words, wl_words): (ValidationMap, ValidationMap), chunk: &Vec<(String, String)>| -> Vec<(String, String)> {
        let mut ret = Vec::new();
        for (sentiment, tweet) in chunk {
            let pair = (String::from(sentiment), clean_words(tweet, Some(&bl_words), Some(&wl_words)));
            ret.push(pair);
        }
        ret
    };

    multi_thread_process_list(&records, (stop_words, wl_words), 16, f_thread, None)
}

pub fn get_word_pairs(text_file_path: &str, stop_word_file: Option<&str>, white_list_file: Option<&str>) -> Vec<InputTup> {
    let err = format!("Error reading input file: {}", text_file_path);
    let file_contents = fs::read_to_string(text_file_path).expect(&err);

    let stop_words = get_optional_validation_map(stop_word_file);
    let white_list_words = get_optional_validation_map(white_list_file);

    let cleaned_text = clean_words(&file_contents, Some(&stop_words), Some(&white_list_words));

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

pub fn reduce<VT, RT>(list: &Vec<VT>, initial: &RT, f: fn(&VT, RT) -> RT) -> RT where RT: Clone {
    let mut ret_val = initial.to_owned();
    for item in list {
        ret_val = f(item, ret_val);
    }
    ret_val
}

pub fn feed_totals(totals: &StateTotals, from_state: &String, to_state: &String) -> HashMap<String, i32> {
    let o_from_map = totals.get(from_state);
    let to_state_c = to_state.clone();
    if o_from_map.is_some() {
        let mut current_total = 0;
        let mut to_map = o_from_map.expect("map err").clone();
        let o_total = to_map.get(to_state);
        if o_total.is_some() {
            current_total = o_total.expect("map err").clone();
        }
        to_map.insert(to_state_c, current_total + 1);
        return to_map
    } else {
        let mut to_map = HashMap::new();
        to_map.insert(to_state_c, 1);
        return to_map
    }
}

pub fn feed_totals_multi(input_data: Vec<InputTup>, initial_data: Option<&StateTotals>) -> StateTotals {
    let mut totals = if initial_data.is_some()
        { initial_data.unwrap().clone() }
        else { StateTotals::new() };

    let f_thread = |_, chunk: &Vec<InputTup>| -> Vec<StateTotals> {
        let mut totals = StateTotals::new();
        for (from_state, to_state) in chunk {
            totals.insert(from_state.clone(), feed_totals(&totals, from_state, to_state));
        }
        let mut ret_val = Vec::new();
        ret_val.push(totals);
        ret_val
    };

    let results = multi_thread_process_list(&input_data, 0, 16, f_thread, None);
    let groups = results
            .into_iter()
            .flat_map(|hm| hm.into_iter())
            .sorted_by(|(from1, _), (from2, _)| from1.cmp(&from2))
            .group_by(|(from, _)| from.to_owned());
        
        for (from, from_list) in groups.into_iter() {
            let to_list_groups = from_list.into_iter()
                .flat_map(|(_, to_hm)| {
                    to_hm.into_iter().collect_vec()
                })
                .sorted_by(|(to1, _), (to2, _)| to1.cmp(&to2))
                .group_by(|(to, _)| to.to_owned());

            let to_list = to_list_groups
                .into_iter()
                .map(|(to, total_groups)| {
                    let total: i32 = total_groups
                        .into_iter()
                        .map(|(_, total)| total)
                        .sum();
                    (to, total)
                })
                .filter(|(wd, _)| !wd.eq(""))
                .sorted_by(|(_, tot1), (_, tot2)| tot1.cmp(tot2))
                .rev()
                .collect_vec();
            
            let mut to_hm: HashMap<String, i32> = HashMap::new();
            for (to, total) in to_list.into_iter().take(100) {
                to_hm.insert(to, total);
            }
            if to_hm.clone().into_iter().len() > 0 {
                totals.insert(from, to_hm);
            }
        }
        totals
}