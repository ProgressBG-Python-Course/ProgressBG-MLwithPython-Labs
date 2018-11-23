import pandas as pd

### creation
# ds = pd.Series([1,2,3,4])


### indexing
# ds = pd.Series([1,2,3,4], index=['a', 'b', 'c', 'd'])

### Series as specialized dictionary
ds = pd.Series({
  "d":4,
  "a":1,
  "c":3,
  "b":2,
  "e":5
})
print(ds)

## get index object:
print(ds.index)
# Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

## numerical or keyword indexes
print(ds["a"])
print(ds[0])
#1
#1

## indexes as list:
print(ds[['a', 'b','c']])
#a    1
#b    2
#c    3
#dtype: int6


## slicing
print(ds["a":"d"])
#a    1
#b    2
#c    3
#d    4
#dtype: int64


