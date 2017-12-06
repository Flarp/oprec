#![feature(trace_macros)]
trace_macros!(true);

extern crate petgraph;
use std::ops::*;
use petgraph::graph::*;

trait ArbitraryNumber: std::fmt::Debug {}

type ConstNum = Box<ArbitraryNumber>;

macro_rules! impl_oprec_op {
    ($lower:ident, $upper:ident) => {
        impl $upper for OpRec {
            type Output = OpRec;
            fn $lower(mut self, mut rhs: OpRec) -> Self::Output {
                let operation = self.graph.add_node(Ops::$upper);
                self.graph.add_edge(self.last, operation, Ops::$upper);
                self.graph.add_edge(rhs.last, operation, Ops::$upper);
                rhs.last = operation;
                self.last = operation;
                self
            }
        }
    }
}

macro_rules! impl_oprec_method {
    ($(($lower:ident, $upper:ident)),*) => {
    impl OpRec {
            $(
            fn $lower(mut self) -> OpRec {
                let operation = self.graph.add_node(Ops::$upper);
                self.graph.add_edge(self.last, operation, Ops::$upper);
                self
            }
            )*
        }
    }
}

macro_rules! impl_op {
    ($lower:ident, $upper:ident, $ty:ty) => {
        impl $upper<$ty> for OpRec {
            type Output = OpRec;
            fn $lower(mut self, rhs: $ty) -> Self::Output {
                let rh_node = self.graph.add_node(Ops::Const(Box::new(rhs)));
                let operation = self.graph.add_node(Ops::$upper);
                self.graph.add_edge(self.last, operation, Ops::$upper);
                self.graph.add_edge(rh_node, operation, Ops::$upper);
                self.last = operation;
                self
            }
        }
    }
}

macro_rules! impl_type {
    ($($ty:ty),*) => {
        $(
        impl ArbitraryNumber for $ty {}
        impl_op!(add, Add, $ty);
        impl_op!(sub, Sub, $ty);
        impl_op!(mul, Mul, $ty);
        impl_op!(div, Div, $ty);
        )*
        impl_oprec_op!(add, Add);
        impl_oprec_op!(sub, Sub);
        impl_oprec_op!(mul, Mul);
        impl_oprec_op!(div, Div);
    }
}

impl_type!(f64, f32, i8, i16, i32, i64);

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
    Atan2,
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
    Const(ConstNum),
    Root
}

#[derive(Debug)]
struct OpRec {
    root: NodeIndex,
    last: NodeIndex,
    graph: Graph<Ops, Ops>
}

impl OpRec {
    fn new() -> OpRec {
        let mut graph = petgraph::graph::Graph::<Ops, Ops>::new();
        let root = graph.add_node(Ops::Root);
        OpRec { graph: graph, root: root, last: root }
    }
    // atan2 and mul_add must be implemented seperate
    // from the impl_oprec_method macro because they both
    // take arguments
    fn atan2<T: ArbitraryNumber>(self, rhs: T) {
        let operation = self.graph.add_node(Ops::Atan2);
        let constant = self.graph.add_node(Ops::ConstNum(Box::new(rhs)));
        self.graph.add
    }
}

impl_oprec_method!(
    (sin, Sin), (cos, Cos), (tan, Tan), 
    (asin, Asin), (acos, Acos), (atan, Atan),
    (sinh, Sinh), (cosh, Cosh), (tanh, Tanh),
    (asinh, Asinh), (acosh, Acosh), (atanh, Atanh),
    (powf, PowF), (powi, PowI),
    (exp, Exp), (exp2, Exp2),
    (ln, Ln), (log, Log), (log10, Log10), (log2, Log2),
    (recip, Recip),
    (sqrt, Sqrt), (cbrt, Cbrt)
);


fn main() {
    let mut test = OpRec::new();
    //let mut test2 = OpRec::new();
    //test2 = test2-2;
    test = test+4;
    test = test-4;
    test = test.sin();
    //test = test*test2;
    println!("{:?}", petgraph::dot::Dot::with_config(&test.graph, &[petgraph::dot::Config::EdgeNoLabel]));
}