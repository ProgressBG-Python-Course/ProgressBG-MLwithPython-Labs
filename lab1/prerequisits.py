# define function addManually, which returns the sum of variable number of arguments
def addManually(l):
    # addManually of list elements
    total=0
    for el in l:
        total += el

    return total

print( addManually([2,3]) )   # 5
print( addManually([2,3,5]) ) # 10

# the same with *args
def add(*args):
    total=0
    for el in args:
        total += el

    return total


print(add(1,2))     # 3
print(add(1,2,3))   # 6



