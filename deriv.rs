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
    fn sin(mut self) -> OpRec {
        let sin = self.graph.add_node(Ops::Sin);
        self.graph.add_edge(self.last, sin, Ops::Sin);
        self
    }
}

fn main() {
    let mut test = OpRec::new();
    let mut test2 = OpRec::new();
    test2 = test2-2;
    test = test+4;
    test = test-4;
    test = test.sin();
    test = test*test2;
    println!("{:?}", petgraph::dot::Dot::with_config(&test.graph, &[petgraph::dot::Config::EdgeNoLabel]));
}