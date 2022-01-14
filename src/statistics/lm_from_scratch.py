""" Building blocks of a linear regression:
1. equation ( y = mx + b )
2. features ( 1D )
3. target
4. model ( with weights and intercept )
5. cost function ( with gradient )
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


class LinearRegression:
    def __init__(self,
                 x_train: np.ndarray, y_train: np.ndarray, lr=0.01, num_epochs=1000,
                 cutoff=0.999, old_cost=1000, intercept=0) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.n_dim, self.m_dim = self.x_train.shape
        self.vector_dim = self.x_train.shape[1]
        self.weights: np.ndarray = np.random.randn(self.vector_dim)

        self.intercept: int = intercept
        self.old_cost: int = old_cost
        self.cutoff: float = cutoff
        self.lr: float = lr
        self.num_epochs: int = num_epochs
        self.callback_dict: Dict[str, List[int]] = {
            'epoch': [],
            'cost': [],
            'delta_y': [],
            'y_predicted': [], }
        self.r2: float = 0.

    def run_epoch(self) -> np.ndarray:
        """ Gradient equation:
        y^ = w0x0 + b with e = (y - y^)^2
        (gradient of cost / y^) * ( gradient of y^ / weight )
        dy^/dw = x0
        de/dy^ = -2(y - y^)
        de/dw = -2(y - y^2)x0
        """
        y_predicted = self.predict()
        current_cost = self.cost(y_predicted)
        # print(y_predicted)

        # Calculate error
        delta_y = y_predicted - self.y_train

        # Compute gradients
        # Missing: (1 / self.n_dim), (1 / self.m_dim)
        weights_grad = np.dot(self.x_train.ravel(), delta_y)
        intercept_grad = np.sum(delta_y)

        # Update parameters
        self.weights -= self.lr * weights_grad
        self.intercept -= self.lr * intercept_grad
        return current_cost, delta_y, y_predicted

    def train(self) -> None:
        for i in range(self.num_epochs):
            _current_cost, _delta_y, _y_predicted = self.run_epoch()

            # Stop training if cost does not improve
            if (_current_cost / self.old_cost) > self.cutoff:
                print("In epoch #{} the cost improved less than {:.5}.".format(i, 1 - self.cutoff))
                break

            # Update break condition
            self.old_cost = _current_cost

            # Update dict
            self.callback_dict['epoch'].append(i)
            self.callback_dict['cost'].append(_current_cost)
            self.callback_dict['delta_y'].append(_delta_y)
            self.callback_dict['y_predicted'].append(_y_predicted)

        print(self.callback_dict)

    def cost(self, y_predicted) -> np.ndarray:
        """ cost = ( prediction - actual value ) ^ 2 """
        cost: np.ndarray = (y_predicted - self.y_train) ** 2
        # print("Cost vector: ", cost)
        # print("Calculated cost: ", np.sum(cost))
        return np.sum(cost)

    def predict(self) -> np.ndarray:
        y_pred: np.ndarray = np.dot(self.x_train, self.weights) + self.intercept
        return y_pred

    def r_squared(self) -> float:
        """ Calculates the r2 adj. value of the final prediction
        :return: Rounded r2 adj.
        """
        n = len(self.x_train)
        m = 1  # number of predicators / parameters
        self.r2 = np.round(
            1 - ((n - 1) / (n - m - 1)) * (1 - np.cov(self.predict()) / np.cov(self.y_train)), 3)
        print(self.r2)
        return self.r2

    def visualize(self) -> None:
        """ Plots the lm line on the data-points
        :return: None
        """
        # Plot real data
        plt.scatter(self.x_train, self.y_train, label="Original values")

        # Plot lm
        axs = plt.gca()
        x_values = np.array(axs.get_xlim())
        y_values = self.intercept + self.weights * x_values
        plt.plot(x_values, y_values, 'g--', label="Linear regression")
        plt.title("Linear regression for floor area by price")
        plt.xlabel("Floor area")
        plt.ylabel("Price")
        plt.text(0, 0, f"R2: {self.r2}")
        plt.legend(loc='best')
        plt.show()

        # Plot loss
        plt.plot(self.callback_dict['epoch'], self.callback_dict['cost'])
        plt.title(f"Stopped at epoch #{self.callback_dict['epoch'][-1] + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


if __name__ == '__main__':
    x_train_ = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
    y_train_ = np.array([1, 4, 7, 9, 11])

    lm = LinearRegression(x_train_, y_train_)
    lm.train()
    lm.r_squared()
    lm.visualize()
