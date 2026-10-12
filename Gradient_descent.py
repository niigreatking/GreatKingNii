# Gradient Descent is usually use to reduce or minimuze 
# Mean Squared Error to calculate the gradient for cost function.
# To solve for gradient, iterate through the data points using 
# new weight and bias values and then take the average of the partial derivatives.

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
    weight -= (weight_deriv / companies) / learning_rate
    bias -= (bias_deriv / companies) / learning_rate

    return weight, bias 