//#![feature(trace_macros, conservative_impl_trait)]
#![allow(dead_code, unreachable_patterns, non_camel_case_types, non_upper_case_globals)]

/*

#![feature(specialization, optin_builtin_traits)]

use std::ops::*;

auto trait notrec {}
impl<T> !notrec for lmao<T> {}

struct lmao<T>(T);

macro_rules! impl_op {
    ($(($lower:ident, $upper:ident)),*) => {
        $(
        impl<U: notrec, T: $upper<U>> $upper<U> for lmao<T> {
            type Output = lmao<T>;
            fn $lower(self, lhs: U) -> Self::Output {
                self
            }
        }
        
        impl<U: $upper<T>, T> $upper<lmao<T>> for lmao<U> {
            type Output = lmao<T>;
            fn $lower(self, lhs: lmao<T>) -> Self::Output {
                lhs
            }
        }
        )*
    }
}

impl_op!((add, Add), (sub, Sub), (div, Div), (mul, Mul));

fn main() {
    lmao(4) + 4i32;
}

*/

pub extern crate petgraph;
extern crate rand;
use std::ops::*;
use std::collections::HashMap;
pub use petgraph::prelude::*;

macro_rules! impl_oprec_op {
    ($lower:ident, $upper:ident) => {
    impl $upper for OpRec {
            type Output = OpRec;
            fn $lower(self, rhs: OpRec) -> Self::Output {
                let mut notself = self.clone();
                let operation = notself.graph.add_node(Ops::$upper);
                merge_oprec_at(rhs, &mut notself, operation);
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
                merge_oprec_at(rhs, self, operation);
                self.last = operation;
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
        
    }
}

macro_rules! impl_oprec_method_intermediate {
    ($(($str:expr, $lower:ident, $upper:ident $(, $var:ident : $ty:ty)*)),*) => {
        impl OpRec {
            $(
            // revolting hack to make already generic functions even
            // more generic
            #[doc = $str]
            pub fn $lower$(<$var: Into<OpRec>>)*(self $(, $var : $var)*) -> OpRec {
                let mut notself = self.clone();
                let operation = notself.graph.add_node(Ops::$upper);
                notself.graph.add_edge(notself.last, operation, 1);
                $(
                let rh_node = $var.into();
                //let rh_node = notself.graph.add_node(Ops::Const(f64::from($var)));
                merge_oprec_at(rh_node, &mut notself, operation);
                )*
                notself.last = operation;
                notself
            }
            )*
            #[inline(always)]
            fn is_oprec_method(g: &Ops) -> Result<Box<Fn(f64) -> f64>, Ops> {
                match g {
                    $(&Ops::$upper => is_oprec_method_macro!($lower $(,$var, $upper)*)),*
                    ,_ => Err(g.clone())
                }
            }
        }
    };

}

macro_rules! impl_oprec_method {
    ($(($lower:ident, $upper:ident $(, $var:ident : $ty:ty)*)),*) => {
        impl_oprec_method_intermediate!($((
            concat!("Performs [`", stringify!($lower), "`](https://doc.rust-lang.org/std/primitive.f64.html#method.",stringify!($lower),") on the tree."),
            $lower, 
            $upper 
            $(, $var:$ty)*)),*);
    };
}

macro_rules! is_oprec_method_macro {
    ($lower:ident, $var:ident, $upper:ident) => { Err(Ops::$upper) };
    ($lower:ident) => {
        Ok(Box::new(f64::$lower))
    };
}

macro_rules! impl_op_inner {
    ($upper:ident, $selfi:ident, $discriminant:ident, $rhs:ident) => {
        let rh_node = $selfi.graph.add_node(Ops::Const(f64::from($rhs)));
        let operation = $selfi.graph.add_node(Ops::$discriminant);
        let root = RootIntersection { root: rh_node, intersection: Some(operation) };
        if $selfi.roots[0].intersection.is_none() {
            $selfi.roots[0].intersection = Some(operation);
        }
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
        
        impl $upper<OpRec> for $ty {
            type Output = OpRec;
            fn $lower(self, rhs: OpRec) -> Self::Output {
                let mut notself = rhs.clone();
                let lh_node = notself.graph.add_node(Ops::Const(f64::from(self)));
                let operation = notself.graph.add_node(Ops::$upper);
                notself.graph.add_edge(notself.last, operation, 0);
                notself.graph.add_edge(lh_node, operation, 1);
                let root = RootIntersection { root: lh_node, intersection: Some(operation) };
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
        impl From<$ty> for OpRec {
            fn from(x: $ty) -> OpRec {
                let mut graph = OpRecGraph::new();
                let constant = graph.add_node(Ops::Const(f64::from(x)));
                let id = rand::random::<u64>();
                OpRec { vars: vec![id], id: id, graph: graph, roots: vec![], last: constant }
            }
        }
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

#[derive(Debug, Clone)]
pub enum Ops {
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
    Pow,
    Exp,
    Ln,
    Add,
    Sub,
    Mul,
    Div,
    Const(f64),
    Var(u64),
    Abs,
}

#[derive(Debug, Clone, PartialEq)]
struct RootIntersection {
    root: NodeIndex,
    intersection: Option<NodeIndex>
}

#[derive(Debug, Clone)]
pub struct OpRec {
    roots: Vec<RootIntersection>,
    last: NodeIndex,
    graph: Graph<Ops, u8>,
    id: u64,
    vars: Vec<u64>
}

//#[derive(Debug, Clone)]
//struct PolynomialTerm {
//    exponent: f64,
//    coefficient: f64,
//    number: Option<f64>
//}

//type Polynomial = Vec<PolynomialTerm>;

impl Default for OpRec {
    fn default() -> Self {
        OpRec::new()
    }
}

impl OpRec {
    /// Constructs a new `OpRec` with default parameters
    pub fn new() -> OpRec {
        let id = rand::random::<u64>();
        let mut graph = petgraph::graph::Graph::<Ops, u8>::new();
        let root = graph.add_node(Ops::Var(id));
        OpRec { vars: vec![id], id: id, graph: graph, roots: vec![RootIntersection { root: root, intersection: None }], last: root }
    }

    /// Converts the current OpRec into a function that will reproduce the operations
    /// applied to it to the functions arguments 
    pub fn functify(self) -> Box<Fn(HashMap<u64, f64>) -> Result<f64, u64>> {
        oprec_to_function_check(&self, self.last)
    }
    
    pub fn differentiate(self) -> OpRec {
        get_derivative(&self, self.last)
    }
    
    pub fn differentiate_wrt(self, respect: &OpRec) -> OpRec {
        let mut notself = self.clone();
        notself.id = respect.id;
        get_derivative(&notself, notself.last)
    }
    
    pub fn id(self, z: Option<u64>) -> OpRec {
        let mut notself = self.clone();
        notself.id = match z {
            Some(x) => x,
            None => rand::random::<u64>()
        };
        notself
    }
    
    #[doc = "Performs [`exp_m1`](https://doc.rust-lang.org/std/primitive.f64.html#method.exp_m1) on the tree."]
    pub fn exp_m1(self) -> OpRec {
        self.exp() - 1f64
    }

    #[doc = "Performs [`ln_1p`](https://doc.rust-lang.org/std/primitive.f64.html#method.ln_1p) on the tree."]
    pub fn ln_1p(self) -> OpRec {
        (self+1f64).ln()
    }
    
    #[doc = "Performs [`log2`](https://doc.rust-lang.org/std/primitive.f64.html#method.log2) on the tree."]
    pub fn log2(self) -> OpRec {
        (self.ln())/(2f64.ln())
    }

    #[doc = "Performs [`log10`](https://doc.rust-lang.org/std/primitive.f64.html#method.log10) on the tree."]
    pub fn log10(self) -> OpRec {
        (self.ln())/(10f64.ln())
    }

    #[doc = "Performs [`cbrt`](https://doc.rust-lang.org/std/primitive.f64.html#method.cbrt) on the tree."]
    pub fn cbrt(self) -> OpRec {
        self.powf((1/3) as f64)
    }

    #[doc = "Performs [`sqrt`](https://doc.rust-lang.org/std/primitive.f64.html#method.sqrt) on the tree."]
    pub fn sqrt(self) -> OpRec {
        self.powf((1/2) as f64)
    }
    // mul_add must be implemented separately because
    // it is two operations in one function
    
    #[doc = "Performs [`mul_add`](https://doc.rust-lang.org/std/primitive.f64.html#method.mul_add) on the tree."]
    pub fn mul_add<T: Into<OpRec>>(self, mul: T, add: T) -> OpRec {
        (self*mul.into())+add.into()
    }

    #[doc = "Performs [`powi`](https://doc.rust-lang.org/std/primitive.f64.html#method.powi) on the tree."]
    pub fn powi(self, i: i32) -> OpRec {
        self.powf(f64::from(i))
    }
    
    // sin_cos must be implemented separately because it returns
    // a tuple
   
    #[doc = "Performs [`sin_cos`](https://doc.rust-lang.org/std/primitive.f64.html#method.sin_cos) on the tree."] 
    pub fn sin_cos(self) -> (OpRec, OpRec) {
        (self.clone().sin(), self.cos())
    }

    #[doc = "Performs [`log`](https://doc.rust-lang.org/std/primitive.f64.html#method.log) on the tree."]
    pub fn log<T: Into<OpRec>>(self, base: T) -> OpRec {
        self.ln()/base.into().ln()
    }
    
    #[doc = "Performs [`hypot`](https://doc.rust-lang.org/std/primitive.f64.html#method.hypot) on the tree."]
    pub fn hypot<T: Into<OpRec>>(self, rhs: T) -> OpRec {
        (self.powi(2) + rhs.into().powi(2)).sqrt()
    }
    
    #[doc = "Performs [`exp2`](https://doc.rust-lang.org/std/primitive.f64.html#method.exp2) on the tree."]
    pub fn exp2(self) -> OpRec {
        OpRec::from(2).powf(self)
    }
   
   /// Returns a reference to the underlying graph
    pub fn graph<'a>(&'a self) -> &'a OpRecGraph {
        &self.graph
    }
    
}

impl_oprec_method!(
    (sin, Sin), (cos, Cos), (tan, Tan), 
    (asin, Asin), (acos, Acos), (atan, Atan),
    (sinh, Sinh), (cosh, Cosh), (tanh, Tanh),
    (asinh, Asinh), (acosh, Acosh), (atanh, Atanh),
    (powf, Pow, float: f64),
    (exp, Exp), (ln, Ln), (abs, Abs)
);

fn oprec_to_function_check(x: &OpRec, last: NodeIndex) -> Box<Fn(HashMap<u64, f64>) -> Result<f64, u64>> {
    let rec = x.clone();
    Box::new(move |x| {
        for var in rec.vars.iter() {
            if x.get(var).is_none() {
                return Err(*var)
            }
        }
        Ok(oprec_to_function(&rec, last)(x))
    })
}

fn oprec_to_function(rec: &OpRec, last: NodeIndex) -> Box<Fn(HashMap<u64, f64>) -> f64> {
    match OpRec::is_oprec_method(&rec.graph[last]) {
        Ok(func) => {
            let prev: NodeIndex = rec.graph.neighbors_directed(rec.last, petgraph::Incoming).next().unwrap();
            let graph = graph_from_branch(rec, prev);
            let inner_func = oprec_to_function(&graph, graph.last);
            Box::new(move |x| func(inner_func(x)))
        },
        Err(x) => match x {
            Ops::Const(z) => Box::new(move |_| z),
            Ops::Var(z) => Box::new(move |x| x[&z]),
            Ops::Pow => {
                let (left, right) = larger_parent_weight(&rec.graph, last);
                let (left_graph, right_graph) = (
                    graph_from_branch(&rec, left),
                    graph_from_branch(&rec, right)
                );
                let (left_func, right_func) = (
                    oprec_to_function(&left_graph, left_graph.last), 
                    oprec_to_function(&right_graph, right_graph.last)
                );
                Box::new(move |x| left_func(x.clone()).powf(right_func(x)))
            },
            Ops::Add => {
                let mut neighbors = rec.graph.neighbors_directed(last, petgraph::Incoming);
                let (left, right) = (neighbors.next().unwrap(), neighbors.next().unwrap());
                let (left_graph, right_graph) = (graph_from_branch(&rec, left), graph_from_branch(&rec, right));
                let (left_func, right_func) = (
                    oprec_to_function(&left_graph, left_graph.last), 
                    oprec_to_function(&right_graph, right_graph.last)
                );
                Box::new(move |x| left_func(x.clone()) + right_func(x))
            },
            Ops::Mul => {
                let mut neighbors = rec.graph.neighbors_directed(last, petgraph::Incoming);
                let (left, right) = (neighbors.next().unwrap(), neighbors.next().unwrap());
                let (left_graph, right_graph) = (graph_from_branch(&rec, left), graph_from_branch(&rec, right));
                let (left_func, right_func) = (
                    oprec_to_function(&left_graph, left_graph.last), 
                    oprec_to_function(&right_graph, right_graph.last)
                );
                Box::new(move |x| left_func(x.clone()) * right_func(x))
            },
            Ops::Div => {
                let (left, right) = larger_parent_weight(&rec.graph, last);
                let right_graph = graph_from_branch(&rec, right);
                let left_graph = graph_from_branch(&rec, left);
                let (left_func, right_func) = (
                    oprec_to_function(&left_graph, left_graph.last), 
                    oprec_to_function(&right_graph, right_graph.last)
                );
                Box::new(move |x| left_func(x.clone())/right_func(x))
            },
            Ops::Sub => {
                let (left, right) = larger_parent_weight(&rec.graph, last);
                let right_graph = graph_from_branch(&rec, right);
                let left_graph = graph_from_branch(&rec, left);
                let (left_func, right_func) = (
                    oprec_to_function(&left_graph, left_graph.last), 
                    oprec_to_function(&right_graph, right_graph.last)
                );
                Box::new(move |x| left_func(x.clone())-right_func(x))
            }
            _ => unreachable!()
        }
    }
}

pub type OpRecGraph = Graph<Ops, u8>;

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
    let mut roots = Vec::new();
    for root in &rec.roots {
        if node_mapping.get(&root.root).is_some() {
            if root.intersection.is_some() {
                roots.push(RootIntersection { root: node_mapping[&root.root], intersection: node_mapping.get(&root.intersection.unwrap()).cloned() });
            } else {
                roots.push(RootIntersection { root: node_mapping[&root.root], intersection: None })
            }
        }
    }
    OpRec { vars: rec.vars.clone(), id: rec.id, graph: new_graph, last: node_mapping[&start], roots: roots }
}

#[inline(always)]
fn larger_parent_weight(graph: &OpRecGraph, last: NodeIndex) -> (NodeIndex, NodeIndex) {
    let mut edges = graph.edges_directed(last, petgraph::Incoming);
    let temp1 = edges.next().unwrap();
    let temp2 = edges.next().unwrap();
    if temp1.weight() > temp2.weight() {
        (temp1.source(), temp2.source())
    } else {
        (temp2.source(), temp1.source())
    }
}

fn merge_oprec_at(merger: OpRec, mergee: &mut OpRec, at: NodeIndex) {
    let mut node_mappings: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    merger.roots.iter().map(|root| mergee.roots.push(root.clone())).count();
    mergee.vars.append(&mut merger.vars.clone());
    mergee.vars.sort_unstable();
    mergee.vars.dedup();
    for index in merger.graph.node_indices() {
        let clone_node = merger.graph[index].clone();
        let final_index = mergee.graph.add_node(clone_node.clone());
        node_mappings.insert(index, final_index);
    }
    for edge in merger.graph.edge_references() {
        mergee.graph.add_edge(node_mappings[&edge.source()].clone(), node_mappings[&edge.target()].clone(), 0);
    }
    mergee.graph.add_edge(node_mappings[&merger.last].clone(), at, 0);
    if mergee.graph.find_edge(mergee.last, at).is_none() {
        mergee.graph.add_edge(mergee.last, at, 1);
    };
}

fn get_derivative(rec: &OpRec, last: NodeIndex) -> OpRec {
    match rec.graph[last] {
        Ops::Sin => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            graph_from_branch(&rec, prev).cos() * get_derivative(rec, prev)
        },
        Ops::Var(z) => if z == rec.id {
            OpRec::from(1f64)
        } else {
            OpRec::from(0f64)
        },
        Ops::Const(_) => OpRec::from(0f64),
        Ops::Cos => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            (graph_from_branch(&rec, prev).sin() * OpRec::from(-1f64)) * get_derivative(rec, prev)
        },
        Ops::Mul => {
            let mut neighbors = rec.graph.neighbors_directed(last, petgraph::Incoming);
            let (left, right) = (neighbors.next().unwrap(), neighbors.next().unwrap());
            (graph_from_branch(&rec, left)*get_derivative(&rec, right)) + (graph_from_branch(&rec, right)*get_derivative(&rec, left))
        },
        Ops::Pow => {
            let (left, right) = larger_parent_weight(&rec.graph, last);
            let right_branch = graph_from_branch(&rec, right);
            match rec.graph[left] {
                //natural logarithm expansion
                Ops::Const(x) => {
                    OpRec::from(x).ln() * graph_from_branch(&rec, last) * get_derivative(&rec, right)
                },
                //power rule
                _ => {
                    (graph_from_branch(&rec, left).powf(graph_from_branch(&rec, right)-1f64)) * (right_branch*get_derivative(rec, left))
                }
            }
        },
        Ops::Tan => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            ((1f64/graph_from_branch(&rec, prev).cos()).powi(2)) * get_derivative(rec, prev)
        },
        Ops::Add => {
            let mut neighbors = rec.graph.neighbors_directed(last, petgraph::Incoming);
            let (left, right) = (neighbors.next().unwrap(), neighbors.next().unwrap());
            get_derivative(&rec, left) + get_derivative(rec, right)
        },
        Ops::Div => {
            let (left, right) = larger_parent_weight(&rec.graph, last);
            let right_branch = graph_from_branch(&rec, right);
            let left_branch = graph_from_branch(&rec, left);
            ((right_branch.clone()*get_derivative(&rec, left))-(left_branch*get_derivative(rec, right)))/(right_branch.powi(2))            
        },
        Ops::Sub => {
            let (left, right) = larger_parent_weight(&rec.graph, last);
            get_derivative(&rec, left) - get_derivative(rec, right)
        },
        Ops::Ln => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)/graph_from_branch(rec, prev)
        },
        Ops::Exp => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            graph_from_branch(&rec, last) * get_derivative(rec, prev)
        },
        Ops::Abs => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            (graph_from_branch(&rec, prev)*get_derivative(&rec, prev))/(graph_from_branch(rec, last))
        },
        Ops::Asin => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)/(1f64-graph_from_branch(&rec, last).powi(2)).sqrt()
        },
        Ops::Acos => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            (get_derivative(&rec, prev)/(1f64-graph_from_branch(&rec, last).powi(2)).sqrt()) * -1
        },
        Ops::Atan => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)/(1+graph_from_branch(rec, last).powi(2))
        },
        Ops::Sinh => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)*graph_from_branch(rec, prev).cosh()
        },
        Ops::Cosh => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)*graph_from_branch(rec, prev).sinh()
        },
        Ops::Tanh => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)/graph_from_branch(rec, prev).cosh().powi(2)
        },
        Ops::Asinh => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)/(graph_from_branch(&rec, prev).powi(2) + 1f64).sqrt()
        },
        Ops::Acosh => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)/(graph_from_branch(&rec, prev).powi(2) - 1f64).sqrt()
        },
        Ops::Atanh => {
            let prev: NodeIndex = rec.graph.neighbors_directed(last, petgraph::Incoming).next().unwrap();
            get_derivative(&rec, prev)/(1f64-graph_from_branch(&rec, prev).powi(2))
        }
    }
}
