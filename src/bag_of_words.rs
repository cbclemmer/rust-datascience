use itertools::Itertools;
use std::collections::HashMap;

pub type InputTup = (String, String);
pub type WordBag = HashMap<String, f32>;
pub type BagMap = HashMap<String, WordBag>;

pub struct BagOfWords {
    pub bags: BagMap
}

impl BagOfWords {
    // Find minimum and maximum probabilities of each word
    // if there is not a substantial difference between the different probabilities
    // assume that it is not useful and remove it from the bags
    // fn prune(mut self) {
    //     let words = self.bags.into_iter()
    //         .flat_map(|(_, bag)| bag.into_iter())
    //         .group_by(|(wd, _)| String::from(wd));

    //     for (wd, probs) in words.into_iter() {
    //         // let min = 
    //     }
    // }

    fn train_word_vector(input_data: Vec<String>, min_prob: f32) -> WordBag {
        let word_group = input_data.iter()
            .flat_map(|s| s.split(" "))
            .sorted()
            .group_by(|s| String::from(s.to_owned()));

        let words = word_group
            .into_iter()
            .map(|(wd, grp)| (wd, grp.count()))
            .collect_vec();
        
        let mut hm = WordBag::new();
        let input_length = input_data.len();
        for (wd, count) in words.clone() {
            if wd.eq("") {continue;}
            let prob = count as f32 / input_length as f32;
            if prob < min_prob {continue;}
            hm.insert(String::from(wd), prob);
        }
        hm
    }

    pub fn new(input_data: &Vec<InputTup>, min_prob: f32) -> BagOfWords {
        let input_groups = input_data.iter()
            .filter(|tup| tup.0 != "")
            .sorted_by(|tup1, tup2| tup1.0.cmp(&tup2.0))
            .group_by(|&tup| tup.0.to_owned());
        
        let mut hm = BagMap::new();
        for (key, group) in &input_groups {
            let wv_input = group.map(|tup| tup.1.to_owned()).collect_vec();
            let wv = BagOfWords::train_word_vector(wv_input, min_prob);
            hm.insert(String::from(key), wv);
        }
        
        return BagOfWords { bags: hm }
    }

    fn test_word(bow: BagMap, word: String) -> String {
        let mut best_prob: (String, f32) = (String::from(""), 0.0);
        let word_clone = &word.clone();
        for bag in bow.clone().into_iter() {
            let m_prob = bag.1.get(word_clone);
            if m_prob.is_none() {
                continue;
            }
            let prob = m_prob.expect("");
            if *prob > best_prob.1 {
                best_prob = (bag.0, prob.to_owned());
            }
        }
        String::from(best_prob.0)
    }

    pub fn test_sentence_static(bow: BagMap, sentence: String) -> String {
        let mut totals_hm: HashMap<String, i32> = HashMap::new();
        for wd in sentence.split(" ") {
            let best_bag = BagOfWords::test_word(bow.clone(), String::from(wd));
            if best_bag.eq("") { continue; }
            let m_total = totals_hm.get(&best_bag);
            let total: i32 = if m_total.is_none() { 1 } else { m_total.expect("") + 1 };
            totals_hm.insert(best_bag, total);
        }
    
        let mut best_bag = (String::from(""), 0);
        for (bag_name, total) in totals_hm.into_iter() {
            // println!("{}: {}", bag_name, total);
            if total > best_bag.1 {
                best_bag = (bag_name, total)
            }
        }
    
        best_bag.0
    }

    pub fn test_sentence(&self, sentence: String) -> String {
        BagOfWords::test_sentence_static(self.bags.clone(), sentence)
    }
}