def f(x):
    return x**2

epsilon = 1e-4
x = 1.5

numericGradient = (f(x+epsilon)-f(x-epsilon))/(2*epsilon)

print(numericGradient, x*2)