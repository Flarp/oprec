# OpRec

A giant work in progress crate that can transform functions. It is planned to support inverses (solving for variables), derivatives using [operator overloading](https://en.wikipedia.org/wiki/Automatic_differentiation#Operator_overloading_(OO)). It can be only used in the latest Rust nightly due to https://github.com/rust-lang/rust/issues/44851.

```rust
extern crate oprec;
use oprec::*;
use std::f64::consts::PI;

fn main() {
  // sin(4x)
  let mut rec = OpRec::new();
  rec *= 4;
  rec = rec.sin();
  let id = rec.id();
  // 4cos(4x)
  let cos_4 = |x: f64| 4.0*((4.0*x).cos());
  let func = rec.differentiate().functify();
  let mut map = HashMap::new();
  map.insert(id, PI);
  
  assert_eq!(cos_4(PI), func(map).ok().unwrap());
}
```

There is currently no documentation due to docs.rs not using the latest nightly.
