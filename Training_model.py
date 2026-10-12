def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    for i in range(companies):
        total_error += (sales[i]-(weight*radio[i] + bias ))**2
    return total_error /companies

def update_weights(radio, sales, weight, bias, learning_rate):
    weight_deriv = 0 
    bias_deriv = 0
    companies = len(radio)
    
    for i in range(companies):
        # Calculating partial derivatives of weight [-2x(y-(mx+b))]
        weight_deriv += -2*radio[i] * (sales[i]-(weight*radio[i] + bias))
        
        # Calculating paritial derivatives for bias [-2(y-(mx+b))]
        bias_deriv += -2 * (sales[i] - (weight*radio[i] + bias))
        
    # Doing subtraction because the derivativespoints in upward direction
    weight -= (weight_deriv / companies) * learning_rate
    bias -= (bias_deriv / companies) * learning_rate

    return weight, bias 

def training (radio, sales, weight, bias, learning_rate, iterator):
    cost_history = [ ]
    
    for i in range(iterator):
        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)  
        
        cost =cost_function(radio, sales, weight, bias)
        cost_history.append(cost)
        
        if i %10 == 0:
            print( "iterator ={:d}   weight ={:.2f}  bias ={:.4f}    cost ={:.2}".format(i, weight, bias, cost)) 
    
    return weight, bias, cost_history

radio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sales = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

weight, bias, history = training(
    radio=radio,
    sales=sales,
    weight=0,
    bias=0,
    learning_rate=0.01,
    iterator=1000
)

print("\nFinal weight:", weight)
print("Final bias:", bias)
print("Final cost:", history[-1])