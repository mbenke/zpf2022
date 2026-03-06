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

# Reification

Extend the `fieldsOf` to produce JSON from records where each field can be a base type, record, or a list.

When using the State monad, we often write a lot of boilerplate of the form

``` haskell
getGenerateDefs :: TcM Bool
getGenerateDefs = gets generateDefs

setGenerateDefs :: Bool -> TcM ()
setGenerateDefs b =
  modify (\env -> env {generateDefs = b})
```

write a function `makeStateAccessors` such that

``` haskell
data State = { field1 :: T1, ..., fieldN :: TN }

makeStateAccessors ''State
```
generates `getFieldK/setFieldK` for all fields (field names can be arbitrary, not necessarily `fieldK`)

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

# Quasiquotation 2
* Extend the expression simplifier with more rules.

* Extend the expression quasiquoter to handle metavariables for
  numeric constants, allowing to implement simplification rules like

```
simpl [expr|$int:n$ + $int:m$|] = [expr| $int:m+n$ |]
```

(you are welcome to invent your own syntax in place of `$int: ... $`)

* generate expression quasiquoters using `bnfc-meta`