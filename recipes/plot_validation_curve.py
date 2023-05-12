def plot_validation_curve(param_range, train_scores, test_scores):
    
    """Plots the validation curve."""
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calculate mean and standard deviation for training set scores:
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Calculate mean and standard deviation for test set scores:
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot mean accuracy scores for training and test sets:
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
    
    # Plot accurancy bands for training and test sets:
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
    
    # Create plot:
    plt.xlim(left=np.min(param_range), right=np.max(param_range))
    plt.ylim(bottom=0, top=1.0)
    plt.title("Validation Curve")
    plt.xlabel("Parameter range")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()
