import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Generate synthetic binary classification data
X, y = make_classification(n_samples=2000, n_features=12, n_informative=10, n_redundant=2, n_classes=2, class_sep=0.9, n_clusters_per_class=2, flip_y=0.05, random_state=7)
y = y.reshape(-1, 1)
X = np.hstack((np.ones((X.shape[0], 1)), X))  #Add bias

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=None, optimizer="gd", beta=0.9):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.losses = []
        self.beta = beta
        self.velocity = None
        if optimizer=="sgd":
            self.learning_rate=learning_rate/10 #Lower learning rate for SGD
    #Sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    #Loss function
    def compute_loss(self, y_true, y_pred):
        clip = 1e-30  #clip to avoid log(0)
        y_pred = np.clip(y_pred, clip, 1 - clip)

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, X, y):
        #Get number of samples and features and initialize weights
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))

        #Initialize velocity when using momentum based batch gradient descent
        if self.optimizer == "mbbgd" and self.velocity is None:
            self.velocity = np.zeros((n_features, 1))

        for epoch in range(self.epochs):
            #For Batch Gradient Descent
            if self.optimizer == "gd" or self.optimizer == "bgd":
                predictions = self.sigmoid(X @ self.weights)
                gradient = (X.T @ (predictions - y)) / n_samples
                self.weights -= self.learning_rate * gradient

            #For Stochastic Gradient Descent
            elif self.optimizer == "sgd":
                indices = np.random.permutation(n_samples) #Shuffle Dataset
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                for i in range(n_samples): #Loop over shuffled dataset one by one
                    xi = X_shuffled[i:i+1]
                    yi = y_shuffled[i:i+1]
                    prediction = self.sigmoid(xi @ self.weights)
                    gradient = xi.T @ (prediction - yi)
                    self.weights -= self.learning_rate * gradient
                

            #For Mini Batch Gradient Descent
            elif self.optimizer == "mbgd":
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                for i in range(0, n_samples, self.batch_size):
                    X_batch = X_shuffled[i:i+self.batch_size] #Perform operations on batch only every iteration
                    y_batch = y_shuffled[i:i+self.batch_size]
                    predictions = self.sigmoid(X_batch @ self.weights)
                    gradient = (X_batch.T @ (predictions - y_batch)) / X_batch.shape[0]
                    self.weights -= self.learning_rate * gradient

            #For Momentum Based Batch Gradient Descent
            elif self.optimizer == "mbbgd":
                predictions = self.sigmoid(X @ self.weights)
                gradient = (X.T @ (predictions - y)) / n_samples

                #Update Velocity
                self.velocity = self.beta * self.velocity + gradient
                self.weights -= self.learning_rate * self.velocity

            #Log loss
            y_pred = self.sigmoid(X @ self.weights)
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            #print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X):
        probs = self.sigmoid(X @ self.weights)
        return (probs >= 0.5).astype(int)

def evaluate_model(name, y_true, y_pred): #Prints performance metrics
    print(f"\n------- {name} -------")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Final Loss:", model.losses[len(model.losses)-1])
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

#Different optimizers to be used
optimizers = {
    "bgd": "Batch Gradient Descent",
    "sgd": "Stochastic Gradient Descent",
    "mbgd": "Mini-Batch Gradient Descent",
    "mbbgd": "Momentum-Based Batch Gradient Descent"
}

for key, name in optimizers.items():
    print(f"\nTraining with {name}")
    model = LogisticRegressionCustom(learning_rate=0.01, epochs=500, batch_size=32 if key == "mbgd" else None, optimizer=key, beta=0.9)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(name, y_test, y_pred)

    #Plot loss for each
    plt.plot(model.losses, label=name)

plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
