use itertools::Itertools;
use std::collections::HashMap;

pub type InputTup<'a> = (&'a str, &'a str);
pub type WordBag = HashMap<String, f32>;
pub type BagMap = HashMap<String, WordBag>;

pub struct BagOfWords {
    bags: BagMap
}

impl BagOfWords {
    fn train_word_vector<'a>(input_data: Vec<&str>) -> WordBag {
        let words = input_data.iter()
            .flat_map(|s| s.split(" "))
            .sorted()
            .group_by(|s| s.clone());
        
        let mut hm = WordBag::new();
        for (wd, grp) in &words {
            let prob = grp.count() as f32 / words.into_iter().count() as f32;
            hm.insert(String::from(wd), prob);
        }
        hm
    }

    pub fn new(input_data: &Vec<InputTup>) -> BagOfWords {
        let input_groups = input_data.iter()
            .filter(|tup| tup.0 != "")
            .sorted_by(|tup1, tup2| tup1.0.cmp(tup2.0))
            .group_by(|&tup| tup.0);
        
        let mut hm = BagMap::new();
        for (key, group) in &input_groups {
            let wv_input = group.map(|tup| tup.1).collect_vec();
            let wv = BagOfWords::train_word_vector(wv_input);
            hm.insert(String::from(key), wv);
        }
        
        return BagOfWords { bags: hm }
    }

    fn test_word(&self, word: String) -> String {
        let mut best_prob: (String, f32) = (String::from(""), 0.0);
        let word_clone = &word.clone();
        for bag in self.bags.clone().into_iter() {
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

    pub fn test_sentence(&self, sentence: String) -> String {
        let mut totals_hm: HashMap<String, i32> = HashMap::new();
        for wd in sentence.split(" ") {
            let best_bag = self.test_word(String::from(wd));
            let m_total = totals_hm.get(&best_bag);
            let total: i32 = if m_total.is_none() { 1 } else { m_total.expect("") + 1 };
            totals_hm.insert(best_bag, total);
        }

        let mut best_bag = (String::from(""), 0);
        for total in totals_hm.into_iter() {
            if total.1 > best_bag.1 {
                best_bag = total
            }
        }

        best_bag.0
    }
}