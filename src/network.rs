extern crate rand;

use rand::prelude::*;
use rand::distributions::StandardNormal;

const E: f64 = 2.71828;

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
            for _ in 0..num {
                current_bias.push(SmallRng::from_entropy().sample(StandardNormal));
            }
            self.biases.push(current_bias);
        }
    }

    // each node has a number of weights equal to the number of nodes in the previous layer
    fn set_weights(&mut self) {
        println!("{:?}", self.sizes);
        let mut w = self.sizes.clone();
        w.remove(0);
        let dim = w.iter().zip(self.sizes.iter());

        // each element of dim represents the dimensions of an array of weights
        // each tuple in dim is a layer and each layer contains an array
        for (x, y) in dim {
            let mut layer: Vec<Vec<f64>> = vec![];
            for _ in 0..*x {
                let mut array: Vec<f64> = vec![];
                for _ in 0..*y {
                    array.push(SmallRng::from_entropy().sample(StandardNormal));
                }
                layer.push(array);
            }
            self.weights.push(layer);
        }
    }

    // I'm having a lot of fun using .iter() and stuff
    // here the network is given an input a which is dot producted with every weight for each node, summed with the biases and sigmoided through all the layers
    pub fn feed_forward(&mut self, mut a: Vec<f64>) -> Vec<f64> {
        let mut t: Vec<f64> = vec![];
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            for n in w.iter() {
                t.push(dot_product(n.to_vec(), a.clone()));
            }
            a = sigmoid(t.iter().zip(b.iter()).map(|(x,y)| x + y).collect());
            t = vec![];
        }
        println!("{:#?}", a);
        a
    }
}

fn sigmoid(z: Vec<f64>) -> Vec<f64> {
    let mut v: Vec<f64> = vec![];
    for x in z {
        v.push(1.0 / (1.0 + E.powf(-x)));
    }
    v
}

pub fn dot_product(a: Vec<f64>, b: Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x,y)| x * y).sum()
}
