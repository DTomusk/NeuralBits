mod network;

const E: f64 = 2.718281828;

fn sigmoid(param: f64) -> f64 {
    1.0/(1.0+E.powf(-param))
}

fn dot_product(w: Vec<f64>, x: Vec<f64>) -> Result<f64, ()> {
    if w.len() != x.len() {
        return Err(())
    } else {
        let mut dprod = 0.0;
        for i in 0..w.len()-1 {
            dprod += w[i]*x[i];
        }
        return Ok(dprod)
    }
}

fn main() {
    println!("Hello, world!");
    let num = sigmoid(0.0);
    println!("Sigmoid of 420: {}", num);
    let mut my_network = network::Network::new(vec![2,3,1]);
    my_network.display();
}
