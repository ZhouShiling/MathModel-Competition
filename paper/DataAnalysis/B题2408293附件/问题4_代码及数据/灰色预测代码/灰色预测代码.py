import numpy as np
import matplotlib.pyplot as plt


def gm11(x0, predict_num):
    """
    Function to perform prediction using the traditional GM(1,1) model.
    """
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2

    y = x0[1:]
    x = z1

    k = ((n - 1) * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
        ((n - 1) * np.sum(x ** 2) - np.sum(x) ** 2)
    b = (np.sum(x ** 2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / \
        ((n - 1) * np.sum(x ** 2) - np.sum(x) ** 2)
    a = -k

    print('Now performing GM(1,1) prediction on the original data:')
    print(x0)
    print(
        f'The development coefficient obtained by least squares fitting is {a:.2f}, and the gray action quantity is {b:.2f}')
    print('----------------------------------------')

    x0_hat = np.zeros(n)
    x0_hat[0] = x0[0]
    for m in range(1, n):
        x0_hat[m] = (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * m)

    result = np.zeros(predict_num)
    for i in range(predict_num):
        result[i] = (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * (n + i))

    # Calculate absolute residuals and relative residuals
    absolute_residuals = x0[1:] - x0_hat[1:]
    relative_residuals = np.abs(absolute_residuals) / x0[1:]

    # Calculate ratio and ratio deviation
    class_ratio = x0[1:] / x0[:-1]
    eta = np.abs(1 - (1 - 0.5 * a) / (1 + 0.5 * a) * (1 / class_ratio))

    return result, x0_hat, relative_residuals, eta


def metabolism_gm11(x0, predict_num):
    """
    Function to perform prediction using the metabolism GM(1,1) model.
    """
    result = np.zeros(predict_num)
    for i in range(predict_num):
        res = gm11(x0, 1)[0][0]
        result[i] = res
        x0 = np.concatenate((x0[1:], [res]))  # Fix here

    return result


def new_gm11(x0, predict_num):
    """
    Function to perform prediction using the new information GM(1,1) model.
    """
    result = np.zeros(predict_num)
    for i in range(predict_num):
        res = gm11(x0, 1)[0][0]
        result[i] = res
        x0 = np.append(x0, res)

    return result


# Clear the workspace
plt.close('all')

# Initial data
data = np.array([
    1012690, 1003769, 1001310, 1000531, 1005271, 994409,
    1078131, 1048657, 1042885, 1017673, 1034229, 1029314,
    1005004, 806243, 849140, 882206, 918097, 912263,
    891480, 916571, 936927, 954389, 950297, 974398,
    981432, 983528, 983394, 986649, 997433, 1007579,
    1023000, 1031000, 1033000, 1071000
])

# Check data validity
ERROR = 0
if np.any(data < 0):
    print("The time series for grey prediction cannot use negative numbers.")
    ERROR = 1

n = len(data)
print(f"The length of the original data is {n}")
if n <= 3:
    print("The data volume is too small to perform grey prediction.")
    ERROR = 1
elif n > 10:
    print("The data volume is too large for grey prediction to provide accurate predictions.")

# Exponential regularity check
if ERROR == 0:
    print("----------------------------")
    print("Exponential Regularity Check")
    x1 = np.cumsum(data)
    rho = data[1:] / x1[:-1]

    fig, ax = plt.subplots()
    ax.plot(range(1, n), rho, 'o-')
    ax.axhline(y=0.5, color='r', linestyle='-')
    ax.text(n - 2, 0.55, 'Critical Line')
    ax.set_xticks(range(1, n))
    ax.set_xlabel('Year Sequence')
    ax.set_ylabel('Smoothness of Original Data')

    print(f"Indicator 1: The percentage of smoothness ratios less than 0.5 is {100 * np.sum(rho < 0.5) / (n - 1):.2f}%")
    print(
        f"Indicator 2: Excluding the first two periods, the percentage of smoothness ratios less than 0.5 is {100 * np.sum(rho[2:] < 0.5) / (n - 3):.2f}%")
    print("Reference standards: Indicator 1 should be greater than 60%, and Indicator 2 should be greater than 90%.")
    judge = int(input("Do you think this data passes the exponential regularity test? Enter 1 for yes, 0 for no: "))
    if judge == 0:
        print("Grey prediction model is not suitable for this data.")
        ERROR = 1

# Split the data into training and testing sets
if ERROR == 0 and n > 4:
    print(
        "Because the number of periods in the original data is greater than 4, we can divide the data into training and testing sets.")
    if n > 7:
        test_num = 3
    else:
        test_num = 2
    train_data = data[:-test_num]
    test_data = data[-test_num:]

    print("Training data:")
    print(train_data)
    print("Testing data:")
    print(test_data)
    print("----------------------------")

    # Traditional GM(1,1) model
    print("Traditional GM(1,1) model prediction details:")
    result1 = gm11(train_data, test_num)[0]

    # New information GM(1,1) model
    print("New information GM(1,1) model prediction details:")
    result2 = new_gm11(train_data, test_num)

    # Metabolism GM(1,1) model
    print("Metabolism GM(1,1) model prediction details:")
    result3 = metabolism_gm11(train_data, test_num)

    fig, ax = plt.subplots()
    ax.plot(range(n - test_num + 1, n + 1), test_data, 'o-', label='Actual Test Data')
    ax.plot(range(n - test_num + 1, n + 1), result1, '*-', label='Traditional GM(1,1)')
    ax.plot(range(n - test_num + 1, n + 1), result2, '+-', label='New Information GM(1,1)')
    ax.plot(range(n - test_num + 1, n + 1), result3, 'x-', label='Metabolism GM(1,1)')
    ax.set_xticks(range(n - test_num + 1, n + 1))
    ax.legend()
    ax.set_xlabel('Year Sequence')
    ax.set_ylabel('Sales')

    SSE_1 = np.sum((test_data - result1) ** 2)
    SSE_2 = np.sum((test_data - result2) ** 2)
    SSE_3 = np.sum((test_data - result3) ** 2)

    print(f"SSE for the traditional GM(1,1) model on the test set is {SSE_1:.2f}")
    print(f"SSE for the new information GM(1,1) model on the test set is {SSE_2:.2f}")
    print(f"SSE for the metabolism GM(1,1) model on the test set is {SSE_3:.2f}")

    if SSE_1 < SSE_2:
        if SSE_1 < SSE_3:
            choose = 1
        else:
            choose = 3
    elif SSE_2 < SSE_3:
        choose = 2
    else:
        choose = 3

    models = ['Traditional GM(1,1) Model', 'New Information GM(1,1) Model', 'Metabolism GM(1,1) Model']
    print(f"Since the SSE of {models[choose - 1]} is the smallest, we should choose it for prediction.")
    print("----------------------------")

    predict_num = int(input("Enter the number of periods you want to forecast: "))
    result, data_hat, relative_residuals, eta = gm11(data, predict_num)

    if choose == 2:
        result = new_gm11(data, predict_num)
    elif choose == 3:
        result = metabolism_gm11(data, predict_num)

    print("Fit results for the original data:")
    for i in range(n):
        print(f"{i + 1} : {data_hat[i]:.2f}")
    print(f"Forecast results for the next {predict_num} periods:")
    for i in range(predict_num):
        print(f"{n + i + 1} : {result[i]:.2f}")

else:
    predict_num = int(input("Enter the number of periods you want to forecast: "))
    print("Due to the small number of data points, we will simply average the results from all three methods.")

    print("Traditional GM(1,1) model prediction details:")
    result1, data_hat, relative_residuals, eta = gm11(data, predict_num)

    print("New information GM(1,1) model prediction details:")
    result2 = new_gm11(data, predict_num)

    print("Metabolism GM(1,1) model prediction details:")
    result3 = metabolism_gm11(data, predict_num)

    result = (result1 + result2 + result3) / 3

    print("Fit results for the original data:")
    for i in range(n):
        print(f"{i + 1} : {data_hat[i]:.2f}")
    print(f"Forecast results for the next {predict_num} periods using the Traditional GM(1,1) model:")
    for i in range(predict_num):
        print(f"{n + i + 1} : {result1[i]:.2f}")
    print(f"Forecast results for the next {predict_num} periods using the New Information GM(1,1) model:")
    for i in range(predict_num):
        print(f"{n + i + 1} : {result2[i]:.2f}")
    print(f"Forecast results for the next {predict_num} periods using the Metabolism GM(1,1) model:")
    for i in range(predict_num):
        print(f"{n + i + 1} : {result3[i]:.2f}")
    print(f"Forecast results averaged over the three methods for the next {predict_num} periods:")
    for i in range(predict_num):
        print(f"{n + i + 1} : {result[i]:.2f}")

# Plot relative residuals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(range(1, n), relative_residuals, '*-')
ax1.legend(['Relative Residuals'])
ax1.set_xlabel('Year Sequence')
ax1.set_xticks(range(1, n))

# Plot ratio deviation
ax2.plot(range(1, n), eta, 'o-')
ax2.legend(['Ratio Deviation'])
ax2.set_xlabel('Year Sequence')
ax2.set_xticks(range(1, n))

# Evaluation of the fit to the original data
average_relative_residuals = np.mean(relative_residuals)
print(f"Average relative residual: {average_relative_residuals:.2f}")
if average_relative_residuals < 0.1:
    print("Residual test indicates that the model fits the original data very well.")
elif average_relative_residuals < 0.2:
    print("Residual test indicates that the model fits the original data to an acceptable level.")
else:
    print("Residual test indicates that the model does not fit the original data well.")

average_eta = np.mean(eta)
print(f"Average ratio deviation: {average_eta:.2f}")
if average_eta < 0.1:
    print("Ratio deviation test indicates that the model fits the original data very well.")
elif average_eta < 0.2:
    print("Ratio deviation test indicates that the model fits the original data to an acceptable level.")
else:
    print("Ratio deviation test indicates that the model does not fit the original data well.")

# Plot the original data, fitted data, and forecasted data
fig, ax = plt.subplots()
ax.plot(range(1, n + 1), data, '-o', label='Original Data')
ax.plot(range(1, n + 1), data_hat, '-*m', label='Fitted Data')
ax.plot(range(n + 1, n + 1 + predict_num), result, '-*b', label='Forecasted Data')
ax.plot([n, n + 1], [data[-1], result[0]], '-*b')
ax.set_xticks(range(1, n + 1 + predict_num))
ax.legend()
ax.set_xlabel('Year Sequence')
ax.set_ylabel('Sales')

plt.show()
