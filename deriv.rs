#![feature(trace_macros)]

#[macro_use]
extern crate lazy_static;
trace_macros!(true);

extern crate num;
use std::ops::*;

#[derive(Debug, Hash)]
enum CalculusOperations {
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
    Pow(Box<Derivable>),
    Exp,
    Exp2,
    Ln,
    Log(Box<Derivable>),
    Log2,
    Log10,
    Sqrt,
    Cbrt,
    Hypot,
    Add(Box<Derivable>),
    Sub(Box<Derivable>),
    Mul(Box<Derivable>),
    Div(Box<Derivable>),
    Nop
}

use std::collections::HashMap;

fn get_derivative(i: CalculusOperations) -> String {
    match i {
        CalculusOperations::Sin => "cos".to_string(),
        CalculusOperations::Cos => "negative sin".to_string()
    };
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
    num: f64,
    ops: CalculusCons
}

impl Derivable {
    fn new(x: f64) -> Derivable {
        Derivable { num: x, ops: CalculusCons::new() }
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

fn main() {
    let z = Derivable::new(14 as f64);
    let x = Derivable::new(15 as f64);
    println!("{:?}", z+x);
    println!("{:?}", DERIVATIVES.get(CalculusOperations::Sin));
}

