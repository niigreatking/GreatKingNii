# import numpy as np
# import pandas as pd
# import random 
# import matplotlib.pyplot as plt
# a1 =np.array([1,2,3])
# a2=np.array([[1,2.0,3.3],[4,5,6.5]])
# a3=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])

# print(a1.shape, a1.ndim, a1.dtype, a1.size, type(a1))
# print(a2.shape, a2.ndim, a2.dtype, a2.size, type(a2))
# print(a3.shape, a3.ndim, a3.dtype, a3.size, type(a3))


# df =pd.DataFrame(np.random.randint (10, size= (5,3)), 
#                                     columns =['a','b','c'])
# print(df)

# print(a2)

# df2 = pd.DataFrame(a2)
# print(df2)

# simply_array = np.array([1,2,3])
# print(simply_array)
# print(simply_array, simply_array.dtype)

# ones = np.ones((10,3))
# print(ones)
# print(ones, ones.dtype)

# zeros = np.zeros((10,3,3))
# print(zeros)
# print(zeros, zeros.dtype)

# range_array = np.arange(0,12,2)
# print(range_array)

# random_array = np.random.randint(10,size=(5,3))
# print(random_array)
# print(np.random.random((5,3)))
# print(np.random.rand(5,3))

# np.random.seed(0)
# df = pd.DataFrame(np.random.randint(10, size=(5,3)))
# print(np.random.randint(10, size=(5,3)))
# print(df)

# print(a1.shape,a2.shape,a3.shape)
# print (a2)

# print(a1+a2)
# print(a2+2)


# massive_array =np.random.random(100000)
# print(massive_array.size, type(massive_array))


# mass_list = [random.randint(0,10) for i in range(100000)]
# print(len(mass_list), type(mass_list))

# print(mass_list[:10])

# print(np.mean(a2))
# print(np.min(a2))
# print(np.std(a2))

# high_var_array = np.array([1,100,200,300,4000,5000])
# low_var_array = np.array([2,4,6,8,10])

# print(np.var(high_var_array), np.var(low_var_array))
# print(np.std(high_var_array), np.std(low_var_array))
# print(np.sqrt(np.var(high_var_array)), np.sqrt(np.var(low_var_array)))

# plt.hist(high_var_array)
# plt.show()

# plt.hist(low_var_array)
# plt.show()
import random
import numpy as np
import pandas as pd

np.random.seed(0)
sales_amnt = np.random.randint(20, size =(5,3))
print(sales_amnt)

weekly_sales=pd.DataFrame(sales_amnt, index =['Mon','Tue','Wed','Thu','Fri'], columns =['Almond butter','Peanut butter','Cashew butter'])
print(weekly_sales)
prices = np.array([10,8,12])
print(prices)

butter_prices = pd.DataFrame(prices.reshape(1,3), index=['price'], columns =['Almond butter','Peanut butter','Cashew butter'])
print(butter_prices.shape)
print(weekly_sales.shape)

total_sales=prices.dot(sales_amnt.T)
print(total_sales)

print(butter_prices.shape, weekly_sales.shape)

daily_sales = butter_prices.dot(weekly_sales.T)
print(daily_sales)

weekly_sales['Total'] = daily_sales.T
print(weekly_sales)