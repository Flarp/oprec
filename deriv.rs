extern crate num;
use std::ops::*;
use std::rc::Rc;

extern crate petgraph;

type ConstNum = Option<f64>;

impl<T: Into<f64>> Add<T> for OpRec {
    type Output = f64;
    fn add(self, next: T) -> Self::Output {
        self.0.add_node() 
    }
}

struct OpRec(petgraph::graph::Graph<Ops, Ops>);

#[derive(Debug)]
enum Ops {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Recip,
    PowI,
    PowF,
    Exp,
    Exp2,
    Ln,
    Log,
    Log2,
    Log10,
    Sqrt,
    Cbrt,
    Hypot,
    Add,
    Sub,
    Mul,
    Div,
    Const(ConstNum)
}

/*
fn get_derivative(i: CalculusOperations) -> CalculusOperations {
    match i {
        CalculusOperations::Sin => CalculusOperations::Cos,
        CalculusOperations::Cos => CalculusOperations::Mul(Box::new(Derivable::new(-1 as f64).push(CalculusOperations::Sin))),
        CalculusOperations::Tan => CalculusOperations::PowI(Box::new(Derivable::new(2 as f64).push(CalculusOperations::Sec))),
        _ => CalculusOperations::Nop
    }
}

fn apply_derivative(chain: Derivable) {

}

#[derive(Debug)]
struct CalculusCons {
    curr_op: CalculusOperations,
    next_op: Option<Box<CalculusCons>>
}

impl CalculusCons {
    fn new() -> CalculusCons {
        CalculusCons { curr_op: CalculusOperations::Nop, next_op: None }
    }
    fn push(self, i: CalculusOperations) -> CalculusCons {
        CalculusCons { curr_op: i, next_op: Some(Box::new(self)) }
    }
}

#[derive(Debug)]
struct Derivable {
    num: Option<f64>,
    ops: CalculusCons
}

impl Derivable {
    fn new(x: f64) -> Derivable {
        Derivable { num: Some(x), ops: CalculusCons::new() }
    }
    fn push(self, i: CalculusOperations) -> Derivable {
        Derivable { num: self.num, ops: self.ops.push(i) }
    }
}

impl Add for Derivable {
    type Output = Derivable;
    fn add(self, next: Derivable) -> Self::Output {
        self.push(CalculusOperations::Add(Box::new(next)))

    }
}
*/
fn main() {
    let mut graph = petgraph::graph::Graph::<Ops, Ops>::new();
    graph.add_node(Ops::Sin);
    println!("can compile? {:?}", graph);
}