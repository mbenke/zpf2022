# Template Haskell

Write a function such that `build_ps n` generates all projections for n-tuples, e.g.

``` haskell
{-# LANGUAGE TemplateHaskell #-}
import Language.Haskell.TH

import Build3

$(build_ps 8)

main = print $ p8_4 (1,2,3,4,5,6,7,8)  
```

should print 4.

Write a function `tupleFromList` such that

```
$(tupleFromList 8) [1..8] == (1,2,3,4,5,6,7,8) 
```

# Quasiquotation 1

write a `matrix` quasiquoter such that

```
*MatrixSplice> :{
*MatrixSplice| [matrix|
*MatrixSplice| 1 2
*MatrixSplice| 3 4
*MatrixSplice| |]
*MatrixSplice| :}
[[1,2],[3,4]]
```

be careful with blank lines!

# Quasiquotation
* Extend the expression simplifier with more rules.

* Extend the expression quasiquoter to handle metavariables for
  numeric constants, allowing to implement simplification rules like

```
simpl [expr|$int:n$ + $int:m$|] = [expr| $int:m+n$ |]
```

(you are welcome to invent your own syntax in place of `$int: ... $`)
