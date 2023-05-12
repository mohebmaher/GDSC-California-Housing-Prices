def plot_learning_curve(train_sizes, train_scores, test_scores):
    
    """Plots the learning curve."""
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create means and standard deviations of training set scores:
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores:
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines:
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands:
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create the plot:
    plt.xlim(left=np.min(train_sizes), right=np.max(train_sizes))
    plt.ylim(bottom=0, top=1.0)
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Score"),
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
