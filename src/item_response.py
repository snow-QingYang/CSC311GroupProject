from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    # We will accumulate the negative log likelihood for each student-question pair
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
        
        theta_i = theta[user_id] 
        beta_j = beta[question_id]  
        
        # compute p(c_ij|theta_i, beta_j)
        probability = sigmoid(theta_i - beta_j)
        
        # compute negative log-likelihood
        if is_correct:
            log_lklihood += np.log(probability)
        else:
            log_lklihood += np.log(1 - probability)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # We will update theta and beta for each student-question pair
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
    
        theta_i = theta[user_id]
        beta_j = beta[question_id]
        
        # Compute prediction probability
        probability = sigmoid(theta_i - beta_j)
        
        # Compute gradient
        theta_grad = is_correct - probability
        beta_grad = probability - is_correct
        # Update parameters
        theta[user_id] += lr * theta_grad
        beta[question_id] += lr * beta_grad
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.random.normal(0, 1, len(data["user_id"]))
    beta = np.random.normal(0, 1, len(data["question_id"]))

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Try different learning rates and iterations
    learning_rates = [0.001, 0.002, 0.005]
    iterations_list = [10, 20, 30]
    
    best_val_acc = 0
    best_lr = 0
    best_iter = 0
    best_theta = None 
    best_beta = None
    
    # Grid search for best parameters
    for lr in learning_rates:
        for iteration in iterations_list:
            print(f"\nLearning rate={lr}, Iterations={iteration}:")
            theta, beta, val_acc_lst = irt(train_data, val_data, lr, iteration)
            
            # Record best results
            if val_acc_lst[-1] > best_val_acc:
                best_val_acc = val_acc_lst[-1]
                best_lr = lr
                best_iter = iteration
                best_theta = theta
                best_beta = beta
                
    print(f"\nBest parameters:")
    print(f"Learning rate: {best_lr}")
    print(f"Number of iterations: {best_iter}")
    print(f"Validation accuracy: {best_val_acc}")
    
    # Evaluate on test set using best parameters
    test_acc = evaluate(test_data, best_theta, best_beta)
    print(f"Test accuracy: {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
