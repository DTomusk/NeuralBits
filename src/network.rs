extern crate rand;

use rand::prelude::*;
use rand::distributions::StandardNormal;

#[derive(Debug)]
pub struct Network {
    num_layers: usize,
    sizes: Vec<i32>,
    biases: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
}

impl Network {
    pub fn new(sizes: Vec<i32>) -> Network {
        let mut my_net = Network {
            num_layers: sizes.len(),
            sizes,
            biases: vec![],
            weights: vec![],
        };

        my_net.set_biases();
        my_net.set_weights();

        my_net
    }

    fn set_biases(&mut self) {
        // for each value in sizes (besides the first) push a vector containing sizes[i] number of N(0,1) elements
        // create a vector for the layer then push that to biases
        for i in 1..self.num_layers {
            let mut current_bias: Vec<f64> = vec![];
            let num = self.sizes[i] as usize;
            for j in 0..num {
                current_bias.push(SmallRng::from_entropy().sample(StandardNormal));
            }
            self.biases.push(current_bias);
        }
    }

    fn set_weights(&mut self) {
        println!("{:?}", self.sizes);
        let mut w = self.sizes.clone();
        w.remove(0);
        let dim = w.iter().zip(self.sizes.iter());
        println!("{:?}", dim);

        // each element of dim represents the dimensions of an array of weights
        // each tuple in dim is a layer and each layer contains an array
        for (x, y) in dim {
            let mut layer: Vec<Vec<f64>> = vec![];
            for i in 0..*x {
                let mut array: Vec<f64> = vec![];
                for j in 0..*y {
                    array.push(SmallRng::from_entropy().sample(StandardNormal));
                }
                layer.push(array);
            }
            self.weights.push(layer);
        }
    }
}
