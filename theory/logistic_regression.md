1.  **Sigmoid Function:** The `_sigmoid` method implements the logistic sigmoid function, which squashes the linear output into a probability between 0 and 1.
2.  **Initialization:** Weights are initialized to zeros, and bias to zero, which is a common starting point for gradient descent.
3.  **Forward Pass:** Inside the `fit` method, `linear_model = np.dot(X, self.weights) + self.bias` calculates the linear combination of inputs and weights, and `y_predicted = self._sigmoid(linear_model)` transforms this into a predicted probability. This is the core of logistic regression's prediction.
4.  **Gradient Descent Update:** The updates `self.weights -= self.lr * dw` and `self.bias -= self.lr * db` correctly apply the gradient descent rule to minimize the cost function.
5.  **Prediction:** The `predict` method calculates probabilities and then thresholds them at 0.5 to classify into 0 or 1, which is standard for binary logistic regression.

### Why in the derivatives there is not logistic function?

The apparent absence of the sigmoid function in the derivative calculation of `dw` and `db` is a beautiful result of the derivative of the logistic loss function. Let's derive it.

**1. Logistic Regression Model:**

The predicted probability $\hat{y}$ is given by the sigmoid function:
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = \mathbf{w}^T \mathbf{x} + b$.

**2. Loss Function (Binary Cross-Entropy):**

For a single training example $(\mathbf{x}, y)$, the binary cross-entropy loss function is:
$$L(\hat{y}, y) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$
For $n$ samples, the average cost function $J$ is:
$$J(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$$

**3. Derivative of the Sigmoid Function:**

Before we proceed, let's recall the derivative of the sigmoid function, which is crucial:
$$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$

**4. Derivatives of the Cost Function with respect to Weights ($\mathbf{w}$) and Bias ($b$):**

We need to calculate $\frac{\partial J}{\partial \mathbf{w}}$ and $\frac{\partial J}{\partial b}$. We'll use the chain rule. Let's focus on a single sample first and then generalize for the sum.

**Derivative with respect to $z$:**
$$\frac{\partial L}{\partial \hat{y}} = - \left[ \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right] = - \frac{y(1 - \hat{y}) - \hat{y}(1 - y)}{\hat{y}(1 - \hat{y})} = - \frac{y - y\hat{y} - \hat{y} + y\hat{y}}{\hat{y}(1 - \hat{y})} = - \frac{y - \hat{y}}{\hat{y}(1 - \hat{y})} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

Now, using the chain rule:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$
We know $\hat{y} = \sigma(z)$ and $\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$.

Substitute these into the chain rule expression:
$$\frac{\partial L}{\partial z} = \left( \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right) \cdot (\hat{y}(1 - \hat{y}))$$
Notice how the $\hat{y}(1 - \hat{y})$ terms cancel out:
$$\frac{\partial L}{\partial z} = \hat{y} - y$$

This is the elegant simplification! The term $\hat{y} - y$ is the "error" or "difference between predicted probability and true label."

**Derivative with respect to Weights ($\mathbf{w}$):**
Recall $z = \mathbf{w}^T \mathbf{x} + b$. So, $\frac{\partial z}{\partial \mathbf{w}} = \mathbf{x}$.
Applying the chain rule:
$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}} = (\hat{y} - y) \mathbf{x}$$

For $n$ samples, the average gradient for $\mathbf{w}$ is:
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)}) \mathbf{x}^{(i)}$$
In matrix form, this is equivalent to:
$$\mathbf{dw} = \frac{1}{n} \mathbf{X}^T (\mathbf{\hat{Y}} - \mathbf{Y})$$
This matches the code's `dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))`.

**Derivative with respect to Bias ($b$):**
Recall $z = \mathbf{w}^T \mathbf{x} + b$. So, $\frac{\partial z}{\partial b} = 1$.
Applying the chain rule:
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = (\hat{y} - y) \cdot 1 = \hat{y} - y$$

For $n$ samples, the average gradient for $b$ is:
$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})$$
In code, this is equivalent to:
$$\mathbf{db} = \frac{1}{n} \sum (\mathbf{\hat{Y}} - \mathbf{Y})$$
This matches the code's `db = (1 / n_samples) * np.sum(y_predicted - y)`.
