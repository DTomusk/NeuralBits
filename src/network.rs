extern crate rand;

use rand::prelude::*;
use rand::distributions::StandardNormal;

pub struct Network {
    pub num_layers: usize,
    pub sizes: Vec<i32>,
    pub biases: Vec<Vec<f64>>,
    pub weights: Vec<Vec<Vec<f64>>>,
}

impl Network {
    pub fn new(sizes: Vec<i32>) -> Network {
        let v = SmallRng::from_entropy().sample(StandardNormal);
        let my_layers = sizes.len();
        let mut my_biases: Vec<Vec<f64>> = vec![];
        let mut my_weights: Vec<Vec<Vec<f64>>> = vec![];
        let mut previous_weights = sizes[0];

        for x in &sizes[1..] {
            println!("Number of biases in layer:{}", x);
            let mut temp_bias: Vec<f64> = vec![];
            let mut temp_weight: Vec<Vec<f64>> = vec![];
            for i in 0..*x {
                temp_bias.push(SmallRng::from_entropy().sample(StandardNormal));
                let mut node_weights: Vec<f64> = vec![];
                println!("Number of weights per node:{}", previous_weights);
                for i in 0..previous_weights {
                    node_weights.push(SmallRng::from_entropy().sample(StandardNormal));
                }
                temp_weight.push(node_weights);
            }
            my_biases.push(temp_bias);
            my_weights.push(temp_weight);
            previous_weights = *x;
        };

        let my_network = Network {
            num_layers: my_layers,
            sizes: sizes,
            biases: my_biases,
            weights: my_weights,
        };
        my_network
    }

    pub fn display(&self) {
        println!("Number of layers: {}", self.num_layers);
        for x in &self.sizes {
            println!("Layer sizes: {}", x);
        };
        for x in &self.biases {
            println!("Layer");
            for y in x {
                println!("Bias: {}", y);
            }
        }
        for x in &self.weights {
            println!("Layer");
            for y in x {
                println!("Node");
                for z in y {
                    println!("Weight: {}", z);
                }
            }
        }
    }
}
