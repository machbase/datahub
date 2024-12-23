import matplotlib.pyplot as plt

def compare_graph(real_values, real_pred_values, x_size, y_size):
    # Compare the test results: Actual values vs Predicted values
    plt.figure(figsize=(x_size, y_size))
    plt.plot(real_values, label='Actual', color='blue')
    plt.plot(real_pred_values, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()