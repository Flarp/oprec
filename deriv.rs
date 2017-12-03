#![feature(trace_macros)]
trace_macros!(true);

extern crate num;
use std::ops::*;

trait Derive: Add + Sub + Neg + Mul + Div + std::marker::Sized {}

macro_rules! ugly_impl {
    ($($t:ty),*) => {
        $(impl Derive for $t {})*
    }
}

ugly_impl!(i32, i16, i8, i64, f32, f64);

#[derive(Debug)]
struct Derivable<T: Derive>(T);

impl<T: Derive> Add for Derivable<T> {
    type Output = Derivable<T>;
    
    fn add(self, other: Derivable<T>) -> Self::Output {
        let x: T = 2;
        Derivable::<T>(x)
    }
}

fn main() {

    println!("{:?}", 3);
}