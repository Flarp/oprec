#![feature(trace_macros, conservative_impl_trait)]
//trace_macros!(true);

extern crate petgraph;
use std::ops::*;
use std::collections::HashMap;
use petgraph::graph::*;
use petgraph::visit::EdgeRef;

macro_rules! impl_oprec_op {
    ($lower:ident, $upper:ident) => {
        impl $upper for OpRec {
            type Output = OpRec;
            fn $lower(self, rhs: OpRec) -> Self::Output {
                let mut notself = self.clone();
                let operation = notself.graph.add_node(Ops::$upper);
                let mut node_mappings: HashMap<NodeIndex, NodeIndex> = HashMap::new();
                rhs.graph.node_indices().map(|index| {
                    let clone_node = rhs.graph[index].clone();
                    let final_index = notself.graph.add_node(clone_node.clone());
                    if clone_node == Ops::Root {
                        let intersection = RootIntersection { root: final_index, intersection: Some(operation) };
                        notself.roots.push(intersection);
                    }
                    node_mappings.insert(index, final_index);
                    
                }).count(); // to consume it;
                rhs.graph.edge_references().map(|edge| {
                    notself.graph.add_edge(node_mappings.get(&edge.source()).unwrap().clone(), node_mappings.get(&edge.target()).unwrap().clone(), 0);
                }).count(); // to consume it
                notself.graph.add_edge(node_mappings.get(&rhs.last).unwrap().clone(), operation, 0);
                notself.graph.add_edge(notself.last, operation, 0);
                notself.last = operation;
                notself
            }
        }
    }
}

macro_rules! impl_oprec_op_mut {
    ($lower:ident, $upper:ident, $discriminant:ident) => {
        impl $upper for OpRec {
            fn $lower(&mut self, rhs: OpRec) {
                let operation = self.graph.add_node(Ops::$discriminant);
                let mut node_mappings: HashMap<NodeIndex, NodeIndex> = HashMap::new();
                rhs.graph.node_indices().map(|index| {
                    let clone_node = rhs.graph[index].clone();
                    let final_index = self.graph.add_node(clone_node.clone());
                    if clone_node == Ops::Root {
                        let intersection = RootIntersection { root: final_index, intersection: Some(operation) };
                        self.roots.push(intersection);
                    }
                    node_mappings.insert(index, final_index);
                    
                }).count(); // to consume it;
                rhs.graph.edge_references().map(|edge| {
                    self.graph.add_edge(node_mappings.get(&edge.source()).unwrap().clone(), node_mappings.get(&edge.target()).unwrap().clone(), 0);
                }).count(); // to consume it
                self.graph.add_edge(node_mappings.get(&rhs.last).unwrap().clone(), operation, 0);
                self.graph.add_edge(self.last, operation, 0);
                self.last = operation;
            }
        }
    }
}

macro_rules! impl_op_mut {
    ($lower:ident, $upper:ident, $ty:ty, $discriminant:ident) => {
        impl $upper<$ty> for OpRec {
            fn $lower(&mut self, rhs: $ty) {
                let rh_node = self.graph.add_node(Ops::Const(f64::from(rhs)));
                let operation = self.graph.add_node(Ops::$discriminant);
                self.graph.add_edge(self.last, operation, 0);
                self.graph.add_edge(rh_node, operation, 0);
                self.last = operation;
            }
        }
    }
}


macro_rules! impl_oprec_method {
    ($(($lower:ident, $upper:ident $(, $var:ident : $ty:ty)*)),*) => {
    impl OpRec {
            $(
            fn $lower(self $(, $var : $ty)*) -> OpRec {
                let mut notself = self.clone();
                let operation = notself.graph.add_node(Ops::$upper);
                notself.graph.add_edge(notself.last, operation, 0);
                $(
                let rh_node = notself.graph.add_node(Ops::Const(f64::from($var)));
                notself.graph.add_edge(rh_node, operation, 0);
                )*
                notself.last = operation;
                notself
            }
            )*
        }
    }
}

macro_rules! impl_op {
    ($lower:ident, $upper:ident, $ty:ty) => {
        impl $upper<$ty> for OpRec {
            type Output = OpRec;
            fn $lower(self, rhs: $ty) -> Self::Output {
                let mut notself = self.clone();
                let rh_node = notself.graph.add_node(Ops::Const(f64::from(rhs)));
                let operation = notself.graph.add_node(Ops::$upper);
                notself.graph.add_edge(notself.last, operation, 0);
                notself.graph.add_edge(rh_node, operation, 0);
                notself.last = operation;
                notself
            }
        }
        
        impl $upper<OpRec> for $ty {
            type Output = OpRec;
            fn $lower(self, rhs: OpRec) -> Self::Output {
                let mut notself = rhs.clone();
                let rh_node = notself.graph.add_node(Ops::Const(f64::from(self)));
                let operation = notself.graph.add_node(Ops::$upper);
                notself.graph.add_edge(notself.last, operation, 0);
                notself.graph.add_edge(rh_node, operation, 0);
                notself.last = operation;
                notself
            }
        }
    }
}

macro_rules! impl_type {
    ($($ty:ty),*) => {
        $(
        impl_op!(add, Add, $ty);
        impl_op!(sub, Sub, $ty);
        impl_op!(mul, Mul, $ty);
        impl_op!(div, Div, $ty);
        impl_op_mut!(add_assign, AddAssign, $ty, Add);
        impl_op_mut!(sub_assign, SubAssign, $ty, Sub);
        impl_op_mut!(mul_assign, MulAssign, $ty, Mul);
        impl_op_mut!(div_assign, DivAssign, $ty, Div);
        )*
        impl_oprec_op!(add, Add);
        impl_oprec_op!(sub, Sub);
        impl_oprec_op!(mul, Mul);
        impl_oprec_op!(div, Div);
        impl_oprec_op_mut!(add_assign, AddAssign, Add);
        impl_oprec_op_mut!(sub_assign, SubAssign, Sub);
        impl_oprec_op_mut!(mul_assign, MulAssign, Mul);
        impl_oprec_op_mut!(div_assign, DivAssign, Div);
    }
}

impl_type!(f64, f32, i8, i16, i32, u8, u16, u32);

#[derive(Debug, Clone, PartialEq)]
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
    Const(f64),
    Root
}

#[derive(Debug, Clone)]
struct RootIntersection {
    root: NodeIndex,
    intersection: Option<NodeIndex>
}

#[derive(Debug, Clone)]
struct OpRec {
    roots: Vec<RootIntersection>,
    last: NodeIndex,
    graph: Graph<Ops, u8>
}

impl OpRec {
    fn new() -> OpRec {
        let mut graph = petgraph::graph::Graph::<Ops, u8>::new();
        let root = graph.add_node(Ops::Root);
        let intersect = RootIntersection { root: root, intersection: None };
        OpRec { graph: graph, roots: vec![intersect], last: root }
    }
    // atan2 and mul_add must be implemented seperate
    // from the impl_oprec_method macro because they both
    // take arguments
    
    fn mul_add<T: Into<f64>>(self, mul: T, add: T) -> OpRec {
        let mut notself = self.clone();
        let mul_op = notself.graph.add_node(Ops::Mul);
        let mul_const = notself.graph.add_node(Ops::Const(mul.into()));
        let add_op = notself.graph.add_node(Ops::Add);
        let add_const = notself.graph.add_node(Ops::Const(add.into()));
        notself.graph.add_edge(notself.last, mul_op, 0);
        notself.graph.add_edge(mul_const, mul_op, 0);
        notself.graph.add_edge(mul_op, add_op, 0);
        notself.graph.add_edge(add_const, add_op, 0);
        notself.last = add_op;
        notself
    }
    
    fn functify<T: Into<f64>>(self, a: T) -> impl Fn(f64) -> f64 {
        let z = a.into();
        move |x| x + z
    }
    
}

impl_oprec_method!(
    (sin, Sin), (cos, Cos), (tan, Tan), 
    (asin, Asin), (acos, Acos), (atan, Atan), (atan2, Atan2, float: f64),
    (sinh, Sinh), (cosh, Cosh), (tanh, Tanh),
    (asinh, Asinh), (acosh, Acosh), (atanh, Atanh),
    (powf, PowF, float: f64), (powi, PowI, int: i32),
    (exp, Exp), (exp2, Exp2),
    (ln, Ln), (log, Log, base: f64), (log10, Log10), (log2, Log2),
    (recip, Recip),
    (sqrt, Sqrt), (cbrt, Cbrt)
);


fn main() {
    let mut test = OpRec::new();
    test = (8*test.sin())+3;
    let mut test2 = OpRec::new();
    test2 = test2.sin();
    test += test2;
    let mut test3 = OpRec::new();
    test3 = test3.sin().cos().tan().powi(3);
    test -= test3;
    println!("{:?}", petgraph::dot::Dot::with_config(&test.graph, &[petgraph::dot::Config::EdgeNoLabel]));
    let mut test4 = OpRec::new();
    println!("{:?}", test4.functify(6)(4.into()));
}
