mod network;

fn main() {
    /*
    println!("Hello, world!");
    let num = sigmoid(0.0);
    println!("Sigmoid of 420: {}", num);
    */
    let mut my_network = network::Network::new(vec![3,10,5,10]);
    let mut data = vec![0.78,-0.53,0.21];
    //my_network.display();
    let result = my_network.feed_forward(data);
    println!("{:?}", result);
}
