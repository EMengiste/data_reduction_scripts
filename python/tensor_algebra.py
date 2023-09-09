def levi_civita(i,j,k):
    order= [1,2,3]
    if i+j+k !=6:
        return 0 
    ind = order.index(i)
    if k!= order[ind-1]:
        return -1
    else:
        return 1
    
def dot_product_o1(tensor1,tensor2):
    compat = len(tensor1)==len(tensor2)
    if compat:
        sum = 0
        for i in range(len(tensor1)):
            sum+=tensor1[i]*tensor2[i]
        return sum
    
def cross_product_o1(tensor1,tensor2):
    compat = len(tensor1)==len(tensor2)
    if compat:
        value = []
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    sum+=tensor1[i]*tensor2[i]
        return sum
arr1= [0,0,1]
arr2= [0,1,0]
print(dot_product_o1(arr1, arr2))
exit(0)
def inner_product(tensor1,tensor2):
    return 0