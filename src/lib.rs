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
                impl_oprec_op_inner!($upper, notself, rhs);
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
                impl_oprec_op_inner!($discriminant, self, rhs);
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
        /*
        impl $upper for OpRecArg {
            fn $lower(&mut self, rhs: OpRecArg) {
                impl_oparg_op_inner!($lower, self, rhs);
            }
        }
        */
    }
    
}

macro_rules! impl_oprec_op_inner {
    ($discriminant:ident, $selfi:ident, $rhs:ident) => {
        let operation = $selfi.graph.add_node(Ops::$discriminant);
        let mut node_mappings: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        $rhs.roots.iter().map(|root| $selfi.roots.push(root.clone())).count();
        $rhs.graph.node_indices().map(|index| {
            let clone_node = $rhs.graph[index].clone();
            let final_index = $selfi.graph.add_node(clone_node.clone());
            node_mappings.insert(index, final_index);
            
        }).count(); // to consume it;
        $rhs.graph.edge_references().map(|edge| {
            $selfi.graph.add_edge(node_mappings.get(&edge.source()).unwrap().clone(), node_mappings.get(&edge.target()).unwrap().clone(), 0);
        }).count(); // to consume it
        $selfi.graph.add_edge(node_mappings.get(&$rhs.last).unwrap().clone(), operation, 0);
        $selfi.graph.add_edge($selfi.last, operation, 1);
        $selfi.last = operation;
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
                notself.roots.push(rh_node);
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
        $selfi.roots.push(rh_node);
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
        /*
        
        */
        impl $upper<OpRec> for $ty {
            type Output = OpRec;
            fn $lower(self, rhs: OpRec) -> Self::Output {
                let mut notself = rhs.clone();
                let rh_node = notself.graph.add_node(Ops::Const(f64::from(self)));
                let operation = notself.graph.add_node(Ops::$upper);
                notself.graph.add_edge(notself.last, operation, 0);
                notself.graph.add_edge(rh_node, operation, 1);
                notself.roots.push(rh_node);
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
    roots: Vec<NodeIndex>,
    last: NodeIndex,
    graph: Graph<Ops, u8>,
    limit_bound: u64
}

#[derive(Debug, Clone)]
struct PolynomialTerm {
    exponent: f64,
    coefficient: f64,
    number: Option<f64>
}

type Polynomial = Vec<PolynomialTerm>;

impl OpRec {
    fn new() -> OpRec {
        let mut graph = petgraph::graph::Graph::<Ops, u8>::new();
        let root = graph.add_node(Ops::Var);
        OpRec { graph: graph, roots: vec![root], last: root, limit_bound: 1_000_000u64 }
    }
    
    fn limit_bound(self, x: u64) -> OpRec {
        let mut notself = self.clone();
        notself.limit_bound = x;
        notself
    }
    // mul_add must be implemented seperatly because
    // it is two operations in one function
    
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

impl From<f64> for OpRec {
    fn from(x: f64) -> OpRec {
        let mut graph = petgraph::graph::Graph::<Ops, u8>::new();
        let constant = graph.add_node(Ops::Const(x));
        OpRec { graph: graph, roots: vec![], last: constant, limit_bound: 1_000_000u64 }
    }
}

fn main() {
    let mut test = OpRec::new();
    test += 4;
    test -= 8;
    test = test.sin().cos().tan();
    let g = petgraph::algo::astar(&test.graph, test.roots[0], |finish| finish == test.last, |_| 0, |_| 0);
    println!("{:?}", g);
    println!("{:?}", petgraph::dot::Dot::with_config(&test.graph, &[petgraph::dot::Config::EdgeNoLabel]));
    println!("{:?}", test.roots);
}
