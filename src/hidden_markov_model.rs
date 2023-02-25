use std::collections::HashMap;

use itertools::Itertools;

use crate::markov_chain::MarkovChain;
use crate::util::InputTup;

pub struct HiddenMarkovModel {
    pub state_chain: MarkovChain, // State -> State -> Probability
    pub observation_chain: MarkovChain, // State -> Observation -> Probability
    pub initial_probabilities: HashMap<String, f32>
}

impl HiddenMarkovModel {
    pub fn clone(hmm: &HiddenMarkovModel) -> HiddenMarkovModel {
        HiddenMarkovModel { 
            state_chain: MarkovChain{ states: hmm.state_chain.states.clone() }, 
            observation_chain: MarkovChain{ states: hmm.observation_chain.states.clone() }, 
            initial_probabilities: hmm.initial_probabilities.clone()
        }
    }

    // input: vector of state -> observation in the order the states appear
    pub fn train(input: Vec<InputTup>) -> HiddenMarkovModel {
        let state_transitions_flat = input
            .clone()
            .into_iter()
            .map(|(from, _)| from)
            .collect_vec();

        let mut last_state = String::from(state_transitions_flat.first().expect("No input items"));
        let mut state_transitions: Vec<InputTup> = Vec::new();
        for current_state in state_transitions_flat.into_iter().skip(1) {
            state_transitions.push((last_state, current_state.to_owned()));
            last_state = current_state;
        }
        let mut state_chain = MarkovChain::new();
        state_chain.states = MarkovChain::train(state_transitions);
        let mut observation_chain = MarkovChain::new();
        observation_chain.states = MarkovChain::train(input.clone());
        let initial_probabilities_groups = input
            .clone()
            .into_iter()
            .group_by(|(state, _)| state.to_owned());

        let total_inputs: f32 = input.len() as f32;
        let initial_probabilities_vec = initial_probabilities_groups
            .into_iter()
            .map(|(state, obs)| (state, obs.count() as f32 / total_inputs))
            .collect_vec();

        let mut initial_probabilities = HashMap::new();
        for (state, prob) in initial_probabilities_vec {
            initial_probabilities.insert(state, prob);
        }
        HiddenMarkovModel { state_chain , observation_chain, initial_probabilities }
    }

    pub fn predict(&self, observations: &Vec<String>) -> String {
        let mut best_prob = (String::from(""), 0 as f32);
        let all_observations = self.observation_chain.states
            .clone()
            .into_iter()
            .map(|a| a.0)
            .collect_vec();
        for obs in all_observations {
            let mut new_obs = observations.clone();
            new_obs.push(obs.clone());
            let prob = self.compute_probability(&new_obs);
            if prob > best_prob.1 {
                best_prob = (obs, prob);
            }
        }
        best_prob.0
    }

    fn compute_probability(&self, observations: &Vec<String>) -> f32 {
        let states = self
            .state_chain
            .states
            .clone()
            .into_iter()
            .map(|(state, _)| state)
            .collect_vec();
        
        let mut total_prob = 0 as f32;
        for state in states {
            total_prob = total_prob + self.alpha(observations.len() as i8, &state, &observations);
        }

        total_prob
    }

    fn prob_obs_given_state(&self, obs: &String, state: &String) -> f32 {
        let o_state_hm = self.observation_chain.states.get(state);
        if o_state_hm.is_none() {
            return 0 as f32
        }
        let state_hm = o_state_hm.expect("ERR");
        let o_obs_hm = state_hm.get(obs);
        if o_obs_hm.is_none() {
            return 0 as f32
        }
        o_obs_hm.expect("ERR").clone()
    }

    fn prob_state_given_state(&self, from_state: &String, to_state: &String) -> f32 {
        let o_state_from_hm = self.state_chain.states.get(from_state);
        if o_state_from_hm.is_none()  {
            return 0 as f32;
        }
        let state_from_hm = o_state_from_hm.expect("ERR");
        let o_state_to_hm = state_from_hm.get(to_state);
        if o_state_to_hm.is_none() {
            return 0 as f32;
        }
        o_state_to_hm.expect("ERR").clone()
    }

    fn alpha(&self, depth: i8, state_i: &String, observations: &Vec<String>) -> f32 {
        let first_observation = observations.first().expect("empty observation list");
        if depth == 1 {
            let prob_initial = self.initial_probabilities
                .get(state_i)
                .expect("Alpha: error looking up initial probability");
            let prob_condition = self.prob_obs_given_state(
                first_observation,
                state_i
            );
            return prob_initial * prob_condition;
        }

        let mut total_prob = 0 as f32;
        for (state_j, _) in self.state_chain.states.clone() {
            let a = self.alpha(
                depth - 1, 
                &state_j, 
                observations
            );

            let prob_state_giv_state = self.prob_state_given_state(&state_i, &state_j);
            let obs = observations.clone().into_iter().rev().skip(depth as usize).last().expect("ERR");
            let prob_obs_giv_state = self.prob_obs_given_state(&obs, state_i);
            total_prob = total_prob + a + prob_state_giv_state + prob_obs_giv_state;
        }
        total_prob
    }
}