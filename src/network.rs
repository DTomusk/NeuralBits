extern crate rand;

use rand::prelude::*;
use rand::distributions::StandardNormal;
use rand::thread_rng;
use rand::seq::SliceRandom;

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
        let nl = sizes.len();
        let mut my_net = Network {
            num_layers: nl,
            sizes: sizes.clone(),
            biases: Network::set_biases(&sizes, nl, true),
            weights: Network::set_weights(&sizes, nl, true),
        };

        my_net
    }

    fn set_biases(sizes: &Vec<i32>, num_layers: usize, random: bool) -> Vec<Vec<f64>> {
        // for each value in sizes (besides the first) push a vector containing sizes[i] number of N(0,1) elements
        // create a vector for the layer then push that to biases
        let mut b: Vec<Vec<f64>> = vec![];
        for i in 1..num_layers {
            let mut current_bias: Vec<f64> = vec![];
            let num = sizes[i] as usize;
            for _ in 0..num {
                if random {
                    current_bias.push(SmallRng::from_entropy().sample(StandardNormal));
                } else {
                    current_bias.push(0.0);
                }
            }
            b.push(current_bias);
        }
        b
    }

    // each node has a number of weights equal to the number of nodes in the previous layer
    fn set_weights(sizes: &Vec<i32>, num_layers: usize, random: bool) -> Vec<Vec<Vec<f64>>> {
        let mut z = sizes.clone();
        z.remove(0);
        let dim = z.iter().zip(sizes.iter());
        let mut w: Vec<Vec<Vec<f64>>> = vec![];

        // each element of dim represents the dimensions of an array of weights
        // each tuple in dim is a layer and each layer contains an array
        for (x, y) in dim {
            let mut layer: Vec<Vec<f64>> = vec![];
            for _ in 0..*x {
                let mut array: Vec<f64> = vec![];
                for _ in 0..*y {
                    if random {
                        array.push(SmallRng::from_entropy().sample(StandardNormal));
                    } else {
                        array.push(0.0);
                    }
                }
                layer.push(array);
            }
            w.push(layer);
        }
        w
    }

    // I'm having a lot of fun using .iter() and stuff
    // here the network is given an input a which is dot producted with every weight for each node, summed with the biases and sigmoided through all the layers
    pub fn feed_forward(&mut self, mut a: Vec<f64>) -> Vec<f64> {
        let mut t: Vec<f64> = vec![];
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            for n in w.iter() {
                // .clone() is a big no no
                t.push(dot_product(n.to_vec(), a.clone()));
            }
            a = sigmoid(t.iter().zip(b.iter()).map(|(x,y)| x + y).collect());
            t = vec![];
        }
        println!("{:#?}", a);
        a
    }

    // stochastic gradient descent
    // training data gives a list of tuples of inputs and expected outputs
    pub fn sgd(&mut self, mut training_data: Vec<(Vec<f64>, Vec<f64>)>, epochs: i32, mini_batch_size: usize, eta: f64) {
        let n = training_data.len();
        let mut rng = thread_rng();
        for j in 0..epochs {
            training_data.shuffle(&mut rng);
            for mini_batch in training_data.chunks(mini_batch_size) {
                self.update_mini_batch(mini_batch, eta);
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: &[(Vec<f64>,Vec<f64>)], eta: f64) {
        // nablas start off as zero arrays of the shape of weights and biases
        // how do you get their shapes? you can use sizes in the same way that the constructor does
        // in fact, you can just call set weights and biases with a special boolean parameter
        let mut nabla_b = Network::set_biases(&self.sizes, self.num_layers, false);
        let mut nabla_w = Network::set_weights(&self.sizes, self.num_layers, false);


        // iterate over the mini batch, call back propogate to find delta nablas and then adjust nablas
        // finally update weights and biases
    }

    fn backprop(&mut self, x: Vec<f64>, y: Vec<f64>) {

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
