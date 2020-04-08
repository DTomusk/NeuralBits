const BIAS: i32 = 3;

fn add(x1: i32, x2: i32) -> (i32, i32) {
    let a = nand(x1, x2);
    let b = nand(x1, a);
    let c = nand(x2, a);
    let carry = nand(a, a);
    let sum = nand(b, c);
    (carry, sum)
}

fn nand(x1: i32, x2: i32) -> i32 {
    if (-2*x1)+(-2*x2)+BIAS>=0{
        1
    } else {
        0
    }
}

fn display_sum(sum: (i32, i32)) {
    println!("Sum: {}{}", sum.0, sum.1);
}

fn main() {
    let sum = add(1,1);
    display_sum(sum);
}
