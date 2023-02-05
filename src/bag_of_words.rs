use itertools::Itertools;
use std::collections::HashMap;

pub type InputTup<'a> = (&'a str, &'a str);
pub type WordVector<'a> = HashMap<&'a str, f32>;
pub type Bow<'a> = HashMap<&'a str, WordVector<'a>>;
pub struct BagOfWords<'a> {
    data: Bow<'a>
}

impl BagOfWords<'_> {
    fn train_word_vector<'a>(input_data: Vec<&'a str>) -> WordVector<'a> {
        let words = input_data.iter()
            .flat_map(|s| s.split(" "))
            .sorted()
            .group_by(|s| s.clone());
        
        let mut hm = WordVector::new();
        for (wd, grp) in &words {
            let prob = grp.count() as f32 / words.into_iter().count() as f32;
            hm.insert(wd, prob);
        }
        hm
    }

    pub fn new<'a>(input_data: &'a Vec<InputTup>) -> BagOfWords<'a> {
        let input_groups = input_data.iter()
            .filter(|tup| tup.0 != "")
            .sorted_by(|tup1, tup2| tup1.0.cmp(tup2.0))
            .group_by(|&tup| tup.0);
        
        let mut hm = Bow::new();
        for (key, group) in &input_groups {
            let wv_input = group.map(|tup| tup.1).collect_vec();
            let wv = BagOfWords::train_word_vector(wv_input);
            hm.insert(key, wv);
        }
        
        return BagOfWords { data: hm }
    }
}