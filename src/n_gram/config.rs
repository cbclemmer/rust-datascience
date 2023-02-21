use std::{str::FromStr, fmt::Debug, fs::File, io::Read};
use json::{JsonValue, parse};

use crate::n_gram::NGram;

/*
Config file structure:
if the config has a key for the pruning type it is automatically selected to be run
if the "selection" object has a key set to true then it will use the default config
if the object for a pruning type is missing some elements, then the default config will be used
{
    probability: {
        starting_probability: 0,
        max_accuracy_reduction: 0,
        probability_multiplyer: 0
    },
    similarity: {
        starting_deviation: 0,
        max_accuracy_reduction: 0,
        probability_multiplyer: 0
    },
    count: {
        min_count: 0,
        adjust_amount: 0
    },
    randomizer: {
        num_params: 0,
        step_size: 0,
        iterations: 0
    },
    selection: {
        probability: true,
        similarity: true,
        count: true,
        randomizer: true
    }
}
*/

fn get_json<T>(obj: &JsonValue, c: &str, k: &str, def: T) -> T where T: FromStr, <T as FromStr>::Err: Debug {
    let err_s = "Error parsing: ".to_owned() + c.clone() + "-" + k.clone();
    if obj.has_key(k) {
        obj[k].dump().parse::<T>().expect(&err_s)
    } else {
        def
    }
}

pub struct PruneProbabilityConfig {
    // Starts at this probability
    pub starting_probability: f32,
    // Will not reduce accuracy beyond the original accuracy - this number
    pub max_accuracy_reduction: f32,
    // Will multiply the minimum probability by this number every iteration
    pub probability_multiplyer: f32,
}

impl PruneProbabilityConfig {
    pub fn default() -> PruneProbabilityConfig {
        PruneProbabilityConfig { 
            starting_probability: 0.00001,
            max_accuracy_reduction: 0.1,
            probability_multiplyer: 2.0
        }
    }

    pub fn from_json(obj: &JsonValue) -> PruneProbabilityConfig {
        let mut tmp_config = PruneProbabilityConfig::default();
        let probability_s = "probability";
        tmp_config.starting_probability = get_json(obj, probability_s, "starting_probability", tmp_config.starting_probability);
        tmp_config.starting_probability = get_json(obj, probability_s, "max_accuracy_reduction", tmp_config.max_accuracy_reduction);
        tmp_config.starting_probability = get_json(obj, probability_s, "probability_multiplyer", tmp_config.probability_multiplyer);
        tmp_config
    }
}

pub struct PruneSimilarityConfig {
    // Starts at this deviation
    pub starting_deviation: f32,
    // Will not reduce accuracy beyond the original accuracy - this number
    pub max_accuracy_reduction: f32,
    // Will multiply the minimum probability by this number every iteration
    pub probability_multiplyer: f32
}

impl PruneSimilarityConfig {
    pub fn default() -> PruneSimilarityConfig {
        PruneSimilarityConfig { 
            starting_deviation: 0.00000001, 
            max_accuracy_reduction: 0.1, 
            probability_multiplyer: 2.0
        }
    }

    pub fn from_json(obj: &JsonValue) -> PruneSimilarityConfig {
        let mut tmp_config = PruneSimilarityConfig::default();
        let similarity_s = "similarity";
        tmp_config.starting_deviation = get_json(obj, similarity_s, "starting_deviation", tmp_config.starting_deviation);
        tmp_config.max_accuracy_reduction = get_json(obj, similarity_s, "max_accuracy_reduction", tmp_config.max_accuracy_reduction);
        tmp_config.probability_multiplyer = get_json(obj, similarity_s, "probability_multiplyer", tmp_config.probability_multiplyer);
        tmp_config
    }
}

pub struct PruneCountConfig {
    pub min_count: i32,
    pub adjust_amount: f32
}

impl PruneCountConfig {
    pub fn default() -> PruneCountConfig {
        PruneCountConfig { 
            min_count: 2,
            adjust_amount: 0.01
        }
    }

    pub fn from_json(obj: &JsonValue) -> PruneCountConfig {
        let mut tmp_config = PruneCountConfig::default();
        let count_s = "count";
        tmp_config.min_count = get_json::<i32>(obj, count_s, "min_count", tmp_config.min_count);
        tmp_config.adjust_amount = get_json(obj, count_s, "adjust_amount", tmp_config.adjust_amount);
        tmp_config
    }
}

#[derive(Clone)]
pub struct RandomizerConfig {
    pub num_params: i32,
    pub step_size: f32,
    pub iterations: i32
}

impl RandomizerConfig {
    pub fn default() -> RandomizerConfig {
        RandomizerConfig { 
            num_params: 10, 
            step_size: 0.001,
            iterations: 1000
        }
    }

    pub fn from_json(obj: &JsonValue) -> RandomizerConfig {
        let mut tmp_config = RandomizerConfig::default();
        let rand_s = "randomizer";
        tmp_config.num_params = get_json(obj, rand_s, "num_params", tmp_config.num_params);
        tmp_config.step_size = get_json(obj, rand_s, "step_size", tmp_config.step_size);
        tmp_config.num_params = get_json(obj, rand_s, "iterations", tmp_config.iterations);
        tmp_config
    }
}

pub struct PruneSelectionConfig {
    pub probability: bool,
    pub similarity: bool,
    pub count: bool,
    pub randomizer: bool
}

impl PruneSelectionConfig {
    pub fn default() -> PruneSelectionConfig {
        PruneSelectionConfig { 
            probability: false, 
            similarity: false, 
            count: false, 
            randomizer: false 
        }
    }

    pub fn from_json(obj: &JsonValue) -> PruneSelectionConfig {
        let mut tmp_config = PruneSelectionConfig::default();
        let select_s = "selection";
        tmp_config.probability = get_json(obj, select_s, "probability", false);
        tmp_config.similarity = get_json(obj, select_s, "similarity", false);
        tmp_config.count = get_json(obj, select_s, "count", false);
        tmp_config.randomizer = get_json(obj, select_s, "randomizer", false);
        tmp_config
    }
}

pub struct LearnConfig {
    pub prune_selection: PruneSelectionConfig,
    pub prune_probability: Option<PruneProbabilityConfig>,
    pub prune_similarity: Option<PruneSimilarityConfig>,
    pub prune_count: Option<PruneCountConfig>,
    pub randomizer: Option<RandomizerConfig>,
}

impl NGram {
    pub fn read_config(file_name: &str) -> LearnConfig {
        let mut config = LearnConfig { 
            prune_selection: PruneSelectionConfig { 
                probability: false, 
                similarity: false, 
                count: false, 
                randomizer: false 
            },
            prune_probability: None,
            prune_similarity: None,
            prune_count: None,
            randomizer: None
        };
        let mut file = File::open(file_name).expect("Creating file object error");
        let mut file_contents = String::new();
        file.read_to_string(&mut file_contents).expect("Reading file error");
        if file_contents.eq("") { panic!("Loading bag of words: File empty") }
        let json_data = parse(&file_contents).unwrap();

        let probability_s = "probability";
        if json_data.has_key(probability_s) {
            config.prune_selection.probability = true;
            config.prune_probability = Some(PruneProbabilityConfig::from_json(&json_data[probability_s]));
        }

        let similarity_s = "similarity";
        if json_data.has_key(similarity_s) {
            config.prune_selection.similarity = true;
            config.prune_similarity = Some(PruneSimilarityConfig::from_json(&json_data[similarity_s]));
        }

        let count_s = "count";
        if json_data.has_key(count_s) {
            config.prune_selection.count = true;
            config.prune_count = Some(PruneCountConfig::from_json(&json_data[count_s]));
        }

        let randomizer_s = "randomizer";
        if json_data.has_key(randomizer_s) {
            config.prune_selection.randomizer = true;
            config.randomizer = Some(RandomizerConfig::from_json(&json_data[randomizer_s]));
        }

        let selection_s = "selection";
        if json_data.has_key(selection_s) {
            config.prune_selection = PruneSelectionConfig::from_json(&json_data[selection_s]);
        }

        config
    }
}