# OpRec

A giant work in progress crate that can transform functions. It is planned to support inverses (solving for variables), derivatives using [operator overloading](https://en.wikipedia.org/wiki/Automatic_differentiation#Operator_overloading_(OO)), and integrals using the [Risch Algorithm](https://en.wikipedia.org/wiki/Risch_algorithm).

```rust
extern crate oprec;
use oprec::*;
use std::f64::consts::PI;

fn main() {
  // sin(4x)
  let rec = (OpRec::new()*4).sin();
  
  // 4cos(4x)
  let cos_4 = |x| 4*((4*x).cos());
  let func = rec.differentiate().functify();
  let mut map = HashMap::new();
  map.insert(rec.id(), PI);
  
  assert_eq!(cos_4(PI), func(map));
}
```
