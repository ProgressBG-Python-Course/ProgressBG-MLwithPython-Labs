import pandas as pd

prices_dict = {
  "apples": 1.5,
  "oranges": 2,
  "bananas": 2.5,
  "strawberies": 3
}

suppliers_dict = {
  "apples":"supplier1",
  "oranges":"supplier2",
  "bananas":"supplier4",
  "strawberries":"supplier3",
}


### Constructing DataFrame
## from  a single Series object:
prices_ds = pd.Series([1.5, 2, 2.5, 3],
            index=["apples", "oranges", "bananas", "strawberries"])
prices_df = pd.DataFrame(prices_ds,columns=["prices"])



## from  a dictionary of Series objects
prices_ds = pd.Series([1.5, 2, 2.5, 3],
            index=["apples", "oranges", "bananas", "strawberries"])
suppliers_ds = pd.Series(["supplier1", "supplier2", "supplier4", "supplier3"],
            index=["apples", "oranges", "bananas", "strawberries"])

fruits_df = pd.DataFrame({
  "prices": prices_ds,
  "suppliers": suppliers_ds
})
print(fruits_df)

