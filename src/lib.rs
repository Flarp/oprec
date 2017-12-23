#![feature(trace_macros, conservative_impl_trait)]
//trace_macros!(true);

extern crate petgraph;
use std::ops::*;
use std::collections::HashMap;
use petgraph::prelude::*;

macro_rules! extend_with_edges {
    ($graph:expr, $(($first:expr, $second:expr, $weight:expr)),*) => {
        $($graph.add_edge($first, $second, $weight);)*
    }
}

macro_rules! impl_oprec_op {
    ($lower:ident, $upper:ident) => {
        impl $upper for OpRec {
            type Output = OpRec;
            fn $lower(self, rhs: OpRec) -> Self::Output {
                let mut notself = self.clone();
                let operation = notself.graph.add_node(Ops::$upper);
                merge_oprec_at(rhs, &mut notself, operation);
                notself
            }
        }
        impl $upper for OpRecArg {
            type Output = OpRecArg;
            fn $lower(self, rhs: OpRecArg) -> Self::Output {
                let mut notself = self.clone();
                notself = match notself.clone() {
                    OpRecArg::Const(x) => match rhs {
                        OpRecArg::Const(y) => OpRecArg::Const(x.$lower(y)),
                        OpRecArg::Rec(y) => OpRecArg::Rec(OpRec::from(x).$lower(y))
                    },
                    OpRecArg::Rec(x) => match rhs {
                        OpRecArg::Rec(y) => OpRecArg::Rec(x.$lower(y)),
                        OpRecArg::Const(y) => OpRecArg::Rec(x.$lower(y))
                    }
                };
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
                merge_oprec_at(rhs, self, operation);
            }
        }
        
        impl $upper for OpRecArg {
            fn $lower(&mut self, rhs: OpRecArg) {
                *self = match self.clone() {
                    OpRecArg::Const(mut x) => match rhs {
                        OpRecArg::Const(y) => {
                            x.$lower(y);
                            OpRecArg::Const(x)
                        },
                        OpRecArg::Rec(y) => {
                            let mut z = OpRec::from(x);
                            z.$lower(y);
                            OpRecArg::Rec(z)
                        }
                    },
                    OpRecArg::Rec(mut x) => match rhs {
                        OpRecArg::Rec(y) => {
                            x.$lower(y);
                            OpRecArg::Rec(x)
                        },
                        OpRecArg::Const(y) => {
                            x.$lower(y);
                            OpRecArg::Rec(x)
                        }
                    }
                };
            }
        }
    }
    
}

macro_rules! impl_op_mut {
    ($lower:ident, $upper:ident, $ty:ty, $discriminant:ident) => {
        impl $upper<$ty> for OpRec {
            fn $lower(&mut self, rhs: $ty) {
                impl_op_inner!($upper, self, $discriminant, rhs);
            }
        }
        
        impl $upper<$ty> for OpRecArg {
            fn $lower(&mut self, rhs: $ty) {
                *self = match self.clone() {
                    OpRecArg::Const(mut x) => {
                        x.$lower(f64::from(rhs));
                        OpRecArg::Const(x)
                    },
                    OpRecArg::Rec(mut x) => {
                        x.$lower(rhs);
                        OpRecArg::Rec(x)
                    }
                };
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
                notself.graph.add_edge(notself.last, operation, 1);
                $(
                let rh_node = notself.graph.add_node(Ops::Const(f64::from($var)));
                let root = RootIntersection { root: rh_node, intersection: Some(operation) };
                notself.roots.push(root);
                notself.graph.add_edge(rh_node, operation, 0);
                )*
                notself.last = operation;
                notself
            }
            )*
        }
        impl OpRecArg {
            $(
            fn $lower(self $(, $var : $ty)*) -> OpRecArg {
                match self {
                    OpRecArg::Rec(x) => OpRecArg::Rec(x.$lower($($var)*)),
                    OpRecArg::Const(x) => OpRecArg::Const(x.$lower($($var)*))
                }
            }
            )*
        }
    }
}

macro_rules! impl_op_inner {
    ($upper:ident, $selfi:ident, $discriminant:ident, $rhs:ident) => {
        let rh_node = $selfi.graph.add_node(Ops::Const(f64::from($rhs)));
        let operation = $selfi.graph.add_node(Ops::$discriminant);
        let root = RootIntersection { root: rh_node, intersection: Some(operation) };
        $selfi.roots.push(root);
        $selfi.graph.add_edge($selfi.last, operation, 1);
        $selfi.graph.add_edge(rh_node, operation, 0);
        $selfi.last = operation;
    }
}

macro_rules! impl_op {
    ($lower:ident, $upper:ident, $ty:ty) => {
        impl $upper<$ty> for OpRec {
            type Output = OpRec;
            fn $lower(self, rhs: $ty) -> Self::Output {
                let mut notself = self.clone();
                impl_op_inner!($upper, notself, $upper, rhs);
                notself
            }
        }
        
        impl $upper<$ty> for OpRecArg {
            type Output = OpRecArg;
            fn $lower(self, rhs: $ty) -> Self::Output {
                match self {
                    OpRecArg::Const(x) => OpRecArg::Const(x.$lower(f64::from(rhs))),
                    OpRecArg::Rec(x) => OpRecArg::Rec(x.$lower(rhs))
                }
            }
        }
        
        impl $upper<OpRec> for $ty {
            type Output = OpRec;
            fn $lower(self, rhs: OpRec) -> Self::Output {
                let mut notself = rhs.clone();
                let rh_node = notself.graph.add_node(Ops::Const(f64::from(self)));
                let operation = notself.graph.add_node(Ops::$upper);
                notself.graph.add_edge(notself.last, operation, 0);
                notself.graph.add_edge(rh_node, operation, 1);
                let root = RootIntersection { root: rh_node, intersection: Some(operation) };
                notself.roots.push(root);
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
    Var
}

#[derive(Debug, Clone)]
enum OpRecArg {
    Rec(OpRec),
    Const(f64)
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
    graph: Graph<Ops, u8>,
    limit_bound: u64
}

//#[derive(Debug, Clone)]
//struct PolynomialTerm {
//    exponent: f64,
//    coefficient: f64,
//    number: Option<f64>
//}

//type Polynomial = Vec<PolynomialTerm>;

impl OpRecArg {
    fn mul_add<T: Into<f64>>(self, mul: T, add: T) -> OpRecArg {
        match self {
            OpRecArg::Const(z) => OpRecArg::Const(z.mul_add(mul.into(), add.into())),
            OpRecArg::Rec(z) => OpRecArg::Rec(z.mul_add(mul.into(), add.into()))
        }
    }
    
    fn sin_cos(self) -> (OpRecArg, OpRecArg) {
        match self {
            OpRecArg::Const(z) => (OpRecArg::Const(z.sin()), OpRecArg::Const(z.cos())),
            OpRecArg::Rec(z) => (OpRecArg::Rec(z.clone().sin()), OpRecArg::Rec(z.cos())),
        }
    }
}

impl OpRec {
    fn new() -> OpRec {
        let mut graph = petgraph::graph::Graph::<Ops, u8>::new();
        let root = graph.add_node(Ops::Var);
        OpRec { graph: graph, roots: vec![RootIntersection { root: root, intersection: None }], last: root, limit_bound: 1_000_000u64 }
    }
    
    fn limit_bound(self, x: u64) -> OpRec {
        let mut notself = self.clone();
        notself.limit_bound = x;
        notself
    }
    // mul_add must be implemented separately because
    // it is two operations in one function
    
    fn mul_add<T: Into<f64>>(self, mul: T, add: T) -> OpRec {
        let mut notself = self.clone();
        let mul_op = notself.graph.add_node(Ops::Mul);
        let mul_const = notself.graph.add_node(Ops::Const(mul.into()));
        let add_op = notself.graph.add_node(Ops::Add);
        let add_const = notself.graph.add_node(Ops::Const(add.into()));
        extend_with_edges!(notself.graph, 
        (notself.last, mul_op, 0), (mul_const, mul_op, 0),
        (mul_op, add_op, 0), (add_const, add_op, 0)
        );
        notself.last = add_op;
        notself
    }
    
    // sin_cos must be implemented separately because it returns
    // a tuple
    
    fn sin_cos(self) -> (OpRec, OpRec) {
        (self.clone().sin(), self.cos())
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

type OpRecGraph = Graph<Ops, u8>;

impl From<f64> for OpRec {
    fn from(x: f64) -> OpRec {
        let mut graph = OpRecGraph::new();
        let constant = graph.add_node(Ops::Const(x));
        OpRec { graph: graph, roots: vec![], last: constant, limit_bound: 1_000_000u64 }
    }
}

/*
fn graph_from_branch(rec: &OpRec, start: NodeIndex) -> OpRec {
    let graph = &rec.graph;
    let mut next_nodes = Vec::new();
    let mut node_mapping = HashMap::new();
    let mut new_graph = OpRecGraph::new();
    next_nodes.push(start);
    node_mapping.insert(start, new_graph.add_node(graph[start].clone()));
    while next_nodes.len() > 0 {
        let mut next_nodes_temp = Vec::new();
        for node in next_nodes {
            for edge in graph.edges_directed(node, petgraph::Incoming) {
                let old_source = edge.source();
                let new_source = new_graph.add_node(graph[old_source].clone());
                let new_target = node_mapping[&node];
                node_mapping.insert(old_source, new_source);
                new_graph.add_edge(new_source, new_target, edge.weight().clone());
                next_nodes_temp.push(old_source);
            }
        }
        next_nodes = next_nodes_temp;
    }
    OpRec { graph: new_graph, last: node_mapping[&start], limit_bound: rec.limit_bound }
}
*/

fn merge_oprec_at(merger: OpRec, mergee: &mut OpRec, at: NodeIndex) {
    let mut node_mappings: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    merger.roots.iter().map(|root| mergee.roots.push(root.clone())).count();
    for index in merger.graph.node_indices() {
        let clone_node = merger.graph[index].clone();
        let final_index = mergee.graph.add_node(clone_node.clone());
        node_mappings.insert(index, final_index);
    }
    for edge in merger.graph.edge_references() {
        mergee.graph.add_edge(node_mappings[&edge.source()].clone(), node_mappings[&edge.target()].clone(), 0);
    }
    mergee.graph.add_edge(node_mappings[&merger.last].clone(), at, 0);
    mergee.graph.add_edge(mergee.last, at, 1);
}

fn get_derivative(rec: &OpRec) -> OpRec {
    // left, right, intersection
    //let mut temp = Vec::new();
    let intersections: Vec<(NodeIndex, NodeIndex, NodeIndex)> = Vec::new();
    unimplemented!();
}

fn main() {
    let mut test = OpRec::new();
    test *= 4;
    test = test.sin().cos().tan().mul_add(3, 5);
    //branch_from_index(&test.graph, &mut x, NodeIndex::new(3));
    //println!("{:?}", &test);
    println!("{:?}", petgraph::dot::Dot::with_config(&test.graph, &[petgraph::dot::Config::EdgeNoLabel]));
    //let g = graph_from_branch(&test, NodeIndex::new(3));
    //println!("{:?}", petgraph::dot::Dot::with_config(&g.graph, &[petgraph::dot::Config::EdgeNoLabel]));
}
