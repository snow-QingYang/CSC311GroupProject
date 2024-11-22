import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Transpose matrix to group by questions instead of users
    trans_matrix = matrix.T
    
    # Use imputer on transposed matrix
    nbrs = KNNImputer(n_neighbors=k)
    imputed_matrix = nbrs.fit_transform(trans_matrix)
    
    imputed_matrix = imputed_matrix.T
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    print("Validation Accuracy: {}".format(acc))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc

def plot_result(k_values, val_accuracies, name ):
    plt.plot(k_values, val_accuracies, marker='o')
    plt.xlabel("k_value")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs K")
    plt.grid(True)
    plt.legend()
    plt.savefig(name)
def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Try different k values and record validation accuracy
    user_based_k_values = [1, 6, 11, 16, 21, 26]
    user_based_val_accuracy = []
    item_based_k_values = [1, 6, 11, 16, 21, 26]
    item_based_val_accuracy = []
    # user_based 
    for k in user_based_k_values:
        print("\nk = {}:".format(k))
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_based_val_accuracy.append(acc)
    
    best_k = user_based_k_values[np.argmax(user_based_val_accuracy)]
    print("\nUser-based best k : {}".format(best_k))
    # Evaluate on test set with the best k value
    print("\nUsing k = {} on test set:".format(best_k))
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("Test Accuracy: {}".format(test_acc))


    # item_based 
    for k in item_based_k_values:
        print("\nk = {}:".format(k))
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_based_val_accuracy.append(acc)
    
    best_k = item_based_k_values[np.argmax(item_based_val_accuracy)]
    print("\nItem-based best k : {}".format(best_k))
    plot_result(item_based_k_values, item_based_val_accuracy, "item_based")

    # Evaluate on test set with the best k value
    print("\nUsing k = {} on test set:".format(best_k))
    test_acc = knn_impute_by_item(sparse_matrix, test_data, best_k)
    print("Test Accuracy: {}".format(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
