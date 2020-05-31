mod netwerk;

fn main() {
    /*
    println!("Hello, world!");
    let num = sigmoid(0.0);
    println!("Sigmoid of 420: {}", num);
    */
    let mut my_network = netwerk::Network::new(vec![3,10,5,10]);

    println!("{:#?}", my_network);

    let mut data = vec![0.78,-0.53,0.21];
}

#[cfg(Test)]
mod tests {
    use super::*;

    // test whether the dimensions of a network are as expected 
    #[test]
    fn network_dimensions() {
    }
}
