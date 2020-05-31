mod network;

fn main() {
    /*
    println!("Hello, world!");
    let num = sigmoid(0.0);
    println!("Sigmoid of 420: {}", num);
    */
    let mut my_network = network::Network::new(vec![3,10,5,10]);

    my_network.feed_forward(vec![0.0, 1.1, 2.2,]);

    //let mut data = vec![0.78,-0.53,0.21];
}

#[cfg(test)]
mod tests {
    use super::*;

    // test whether the dimensions of a network are as expected
    #[test]
    fn network_dimensions() {
    }

    // maybe should be separate tests, but testing dot product
    #[test]
    fn dot_product() {
        assert_eq!(network::Network::dot_product(vec![1.0, 2.0], vec![2.0, 3.0]), 8.0);
        assert_eq!(network::Network::dot_product(vec![5.0, 3.0], vec![-1.0, 3.0]), 4.0);
        assert_eq!(network::Network::dot_product(vec![5.0, 3.0, 2.0, 1.0], vec![-1.0, 3.0, 4.0]), 12.0);
    }
}
