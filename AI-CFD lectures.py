#!/usr/bin/env python
# coding: utf-8

# # Test case

# In[1]:


import numpy as np

# define two vectors
vector1= np.array ([1, 3, 3])
vector2= np.array ([4, 5, 6])

# Vector Addition
result_addition = vector1 + vector2
print("Vector Addition Result;", result_addition)

# Vector Subtraction
result_subtraction = vector1 - vector2
print("Vector Subtraction Result;", result_subtraction)

# Scalar Multiplication
scalar = 2
result_scalar_multiplication = vector1 * scalar
print("Scalar Multiplication Result;", result_scalar_multiplication)

# Dot Product
dot_product = np.dot(vector1, vector2)
print("Dot Product Result;", dot_product)


# # Derivative

# In[2]:


import sympy as sp
# Define a variable and a function
x = sp. Symbol('x')
func = x**2 + 3*x + 2
# Calculate the derivative of the function with respect to x
derivative = sp.diff (func, x)

# print the result
print("Derivative of the function:", derivative)


# # Derivative1

# In[3]:


import sympy as sp
# Define a variable and a function
x = sp. Symbol('x')
func = x**3 + 3*x**2 + 2*x+5
# Calculate the derivative of the function with respect to x
derivative = sp.diff (func, x)

# print the result
print("Derivative of the function:", derivative)


# # Gradient

# In[4]:


import numpy as np
# definea function to be optimized (e.g., a simple quadratic function)
def loss_function(x):
    return x**2 + 3*x + 2
# define the derivative of a loss function
def derivative_loss_function(x):
    return 2*x + 3
#gradient descent
learning_rate = 0.1
iterations = 100
initial_x = 0
for i in range(iterations):
    gradient = derivative_loss_function(initial_x)
    initial_x = initial_x - learning_rate * gradient
# Prient the result
print(" Optimized value of x:", initial_x)


# # Gradient Descent2

# In[10]:


import numpy as np
import sympy as sp
# Define a variable and a function
x = sp. Symbol('x')
func = x**2 + 3*x + 2
# Calculate the derivative of the function with respect to x
derivative = sp.diff (func, x)

# print the result
print("Derivative of the function:", derivative)
#convert the derivative expression into a Python function
derivative_function = sp.lambdify(x, derivative)
#gradient descent
learning_rate = 0.1
iterations = 100
initial_x = 0
for i in range(iterations):
    gradient = derivative_function(initial_x)
    initial_x = initial_x - learning_rate * gradient
# Prient the result
print(" Optimized value of x:", initial_x)


# # Adam

# In[8]:


import numpy as np
import tensorflow as tf

# Define a simple loss function for illustration purpose
def loss_function(x):
    return x**2 + 2*x + 1
#create a variable to be optimized
x = tf.Variable(0.0)

#adam optimization
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

iterations=400
for i in range(iterations):
    with tf.GradientTape() as tape:
        loss = loss_function(x)
        gradients = tape.gradient(loss, [x])
        optimizer.apply_gradients(zip(gradients, [x]))
# print the optimized value
print("optimized value of x:", x.numpy())


# # Probability

# In[9]:


import numpy as np
# simulating a fair coin toss
num_tosses = 1000
# 0 represents tails, and 1 represents heads
coin_tosses = np.random.randint(0, 2, num_tosses)
#print(coin_tosses)
# calculate the probability of getting heads
num_heads = np.sum(coin_tosses == 1)
probability_heads = num_heads / num_tosses
# Print the result
print("Probability of getting heads:", probability_heads)


# #Python basics

# In[14]:


#python variable and dara Types
name = "Junaid"
age = 39 
height = 5.11
is_student = True
hobbies = {'Reading', 'coding', 'travelling'}
address = {'city': 'eidgah', 'zipcode': '190002'}
# output
# Print the result
print("name:", name)
print("age:", age)
print("height:", height)
print("is student?", is_student)
print("hobbies:", hobbies)
print("address:", address)


# # Control Flow and Function

# In[1]:


# control flow -if else
num = 10

if num > 0:
    print ("positive number")
elif num < 0:
    print ("negative number")
else:
    print ("zero")
# control flow for loop
for i in range(10):
    print ("iterations:", i)
# control flow while loop
count = 0
while count <= 100:
    print ("count:", count)
    count += 1
# function
def add_numbers(a, b):
    return a+b
result = add_numbers(5,8)
print ("result of addition:", result)


# # class

# In[ ]:


#example of python class


# In[9]:


class employee:
    def __init__(self,name,age,department):
        self.name = name
        self.age = age
        self.department = department
    def get_info(self):
        return f"{self.name},{self.age} years old, works in {self.department} department."
    def print_employee_info(employee_obj):
        print("employee information:")
        print(employee_obj.get_info())
#creating class instances
employee1 = employee("junaid", 39, "engineering")
employee1 = employee("iqra", 38, "biochemistry")


# # example of python class tutorial

# In[15]:


import numpy as np
class MathOperations:
    def vector_addition(self, vector1, vector2):
        return np.add(vector1, vector2)
    def vector_subtraction(self, vector1, vector2):
        return np.subtract(vector1, vector2)
    def matrix_multiplication(self, matrix1, matrix2):
        return np.dot(matrix1, matrix2)
    def dot_product(self, vector1, vector2):
        return np.dot(vector1, vector2)  
    def define_function(self, x):
        return x**2 + 3*x +2
    def calculate_derivative(self, func, x):
        h = 1e-5
        return (func(x + h)-func(x))/ h
    def gradient_descent(self,learning_rate, iterations, initial_x):
        x = initial_x
        for _ in range(iterations):
            gradient = self.calculate_derivative(self.define_function, x)
            x = x - learning_rate * gradient 
            return x
    def calculate_mean(self, data):
        return np.mean(data)
    def calculate_median(self, data):
        return np.median(data)
    def calculate_std_deviation(self, data):
        return np.std(data)
    def calculate_variance(self, data):
        return np.var(data)
math_ops = MathOperations()

vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])
print("Vector Addition:", math_ops.vector_addition(vector1, vector2))
print("Vector Subtraction:", math_ops.vector_subtraction(vector1, vector2))
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print("Matrix Multiplication:", math_ops.matrix_multiplication(matrix1, matrix2))
print("Dot Product:", math_ops.dot_product(matrix1, matrix2))
x_value = 2
print("Function value at x =", x_value, ":", math_ops.define_function(x_value))
print("derivative at x =", x_value, ":", math_ops.calculate_derivative(math_ops.define_function, x_value))
learning_rate = 0.1
iterations = 100
initial_x = 0
print("optimized value of x:", math_ops.gradient_descent(learning_rate, iterations, initial_x))
data = np.array([1, 2, 3, 4, 5])
print("mean:", math_ops.calculate_mean(data))
print("median:", math_ops.calculate_median(data))
print("standard dev:", math_ops.calculate_std_deviation(data))
print("var:", math_ops.calculate_variance(data))


# # creating NumPy Array

# In[16]:


import numpy as np
#create a 1D Numpy array
arr1d = np.array([1, 2, 3, 4, 5])
print("1D Array:",arr1d)
#create a 2D Numpy array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:",arr2d)


# # mathematical Operations

# In[18]:


arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
addition_result = arr1 + arr2
subtraction_result = arr1 - arr2
multiplication_result = arr1 * arr2
division_result = arr2 / arr1
print("addition:", addition_result)
print("sub:", subtraction_result)
print("mul:", multiplication_result)
print("div:", division_result)


# # broadcasting

# In[20]:


#broadcasting with scalar
arr = np.array([1, 2, 3])
scalar_value = 2
result = arr + scalar_value
print("broadcasting with scalar:", result)
#broadcasting with different shaped arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([1, 2])
result = arr1 + arr2
print("broadcasting with  different shaped arrays:", result)


# # Linear Algebra Operations

# In[22]:


# Matrix Multiplication
matrix1 = np.array([[1, 2],[3,4]])
matrix2 = np.array([[5, 6],[7,8]])
result = np.dot(matrix1, matrix2)
print("Matrix Multiplications:", result)
#Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix1)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)


# # Fourier Transform

# In[24]:


#Fourier Transform
arr = np.array([1, 2, 3, 4, 5])
fft_result = np.fft.fft(arr)
print("Original array:",arr)
print("Fourier Transform:",fft_result)


# # pandas data frame

# In[32]:


# create pandas dataframe
import pandas as pd
data = {
    'name': ['junaid', 'iqra', 'asad', 'amal'],
    'age': [39, 38, 5, 3],
    'education': ['Phd', 'Phd', 'LKG', 'preschool']
}

df = pd.DataFrame(data)
print("pandas dataframe", df)
average_age = df['age'].mean()
max_age = df['age'].max()
print("average age:", average_age)
print("max age:", max_age)


# # logistic regression

# In[24]:


import numpy as np
from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
#load the iris datasheet
iris = load_iris()
print("X.shape", X.shape, "Y.shape", y.shape)
X, y = iris.data, iris.target
#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("X_train.shape", X_train.shape)
#create a Logistic regression classifier
classifier = LogisticRegression()
# train the model on the training data
classifier.fit(X_train, y_train)
# make predictions on the test data
y_pred = classifier.predict(X_test)
#evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
# Create a KNeighborsClassifier with a specified value for k (n_neighbors)
k_neighbors = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
k_neighbors.fit(X_train, y_train)
print(f"Accuracy: {accuracy}")


# # KNeighborsClassifier

# In[26]:


import numpy as np
from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#load the iris datasheet
iris = load_iris()
X = iris.data
Y = iris.data
#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize the K-nearesr neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
# train the classifier on the training data
knn_classifier.fit(X_train, y_train)
# make predictions on the test data
y_pred = knn_classifier.predict(X_test)

#calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # decision Treeclassifier

# In[28]:


import numpy as np
from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#load the iris datasheet
iris = load_iris()
X = iris.data
Y = iris.data
#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# create a decision tree classifier
clf = DecisionTreeClassifier(random_state=50)
# train the classifier on the training date
clf.fit(X_train, y_train)
# make prediction on the training data
y_pred = clf.predict(X_test)
#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # RandomForestClassifier

# In[33]:


import numpy as np
from sklearn.datasets import  load_iris
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#load the iris datasheet
iris = load_iris()
X = iris.data
Y = iris.data
#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# train the classifier on the training date
clf.fit(X_train, y_train)
# make prediction on the testing data
y_pred = clf.predict(X_test)
#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # Boston Dataset

# In[2]:


import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# load the boston dataset
boston = load_boston()
X, y = boston.data, boston.target
# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# create a linear regression model
regressor = LinearRegression()
# train the model on the training data
regressor.fit(X_train, y_train)
# make prediction on the testing data
y_pred = regressor.predict(X_test)
#evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Square Error:", mse)
print("R-squared:", r2)


# # Neural Network Regression

# In[9]:


import numpy as np
import h5py
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
# load the boston dataset
# boston = load_boston()
# X, y = boston.data, boston.target

# read from h5 file
with h5py.File('boston_dataset.h5', 'r') as hf:
   X = hf['X'][:]
   y = hf['y'][:]
print("X.shape", X.shape, "y.shape", y.shape)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale the features using standardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# create a neural network regression model
regressor = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=5000, random_state=42)
# train the model on the scaled training data
regressor.fit(X_train_scaled, y_train)
# make predictions on ythe scaled test data
y_pred = regressor.predict(X_test)
#evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Square Error:", mse)
print("R-squared:", r2)


# # scikit learn for clustring

# In[28]:


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
#create a K Means clustering model
kmeans =KMeans(n_clusters=4, random_state=42)
#fit te model to use data
kmeans.fit(X)
# get cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
# plot the data points and cluster centers
plt.scatter(X[:, 0], X[:, 0],c=labels, cmap='viridis', edgecolors='k')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red',
           marker='X', s=200, label='cluster centers')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.show()


# # Tensor Flow

# In[40]:


import tensorflow as tf
import numpy as np
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
# lets make this as float tensor
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
# lets make this as float tensor
rank_2_tensor = tf.constant([[2, 3],
                            [3, 4],
                            [3, 6]], dtype=tf.float16)
print(rank_2_tensor)
print(rank_2_tensor.dtype)


# # 3 axis tensors

# In[43]:


import tensorflow as tf
import numpy as np
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
print(rank_3_tensor.shape)
print(rank_3_tensor.dtype)


# # 4 axis tensors

# In[44]:


rank_4_tensor =tf.zeros([3, 2, 4, 5])
print(rank_4_tensor.dtype)
print(rank_4_tensor.ndim)
print(rank_4_tensor.shape)
print(tf.size(rank_4_tensor).numpy())


# # indexing

# In[55]:


rank_1_tensor = tf.constant([0, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
print("First", rank_1_tensor[0].numpy())
print("Second", rank_1_tensor[1].numpy())
print("last", rank_1_tensor[-1].numpy())
print("everything", rank_1_tensor[:].numpy())
print("before 4", rank_1_tensor[:4].numpy())
print("from 4 to end", rank_1_tensor[4:].numpy())
print("from 2 before 7", rank_1_tensor[2:7].numpy())
print("reversed", rank_1_tensor[::-1].numpy())


# In[8]:


print(rank_2_tensor)
# get row and column tensors
print("second row", rank_2_tensor[1, :].numpy())
print("last row", rank_2_tensor[-1, :].numpy())
print("first item in last column", rank_2_tensor[0, -1].numpy())


# # Manuplating the shape

# In[11]:


import tensorflow as tf
import numpy as np
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
# -1 passed in the shape argument says 
print(tf.reshape(rank_3_tensor, [-1]))
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, 1]))


# # Gradient tape

# In[21]:


import tensorflow as tf
# define some var
x = tf.Variable(2.0)
y = tf.Variable(3.0)

# define a function
def loss_function(x, y):
    return x** 2 + y**3
#create a gradient tape context
with tf.GradientTape() as tape:
        # compute the loss
        loss = loss_function(x, y)
# calculate gradients of the loass with respect tpo x and y
gradient = tape.gradient(loss, [x, y])
# print the gradients 
print("Gradient with respect to x:", gradient[0].numpy())
print("Gradient with respect to y:", gradient[1].numpy())


# In[50]:


import tensorflow as tf
# define function
def matrix_multiply(x, w):
    return tf.matmul(x, w)
def add_bias(x, b):
    return x + b
def relu(x):
    return tf.maximum(x, 0)

# sample data
inputs = tf.constant([[1.0, 2.0]])
weights = tf.constant([[0.5], [0.8]])
bias = tf.constant([.1])
targets = tf.constant([[0.0]])

# initialize variables for weights and bias
w = tf.Variable(weights)
b = tf.Variable(bias)

# create tensorflow var (inputs)
x = tf.Variable(inputs)

#forward pass
with tf.GradientTape(persistent=True) as tape:
    # chain function
    z1 = matrix_multiply(x, w)
    z2 = add_bias(z1, b)
    z3 = relu(z2)
    loss = tf.reduce_mean(tf.square(z3 - targets))
    
# compute gradients
gradients=  tape.gradient(loss, [w, b, x])
# print the gradient
print("Gradient of loss with respect to weight", gradients[0].numpy())
print("Gradient of loss with respect to bias", gradients[1].numpy())
print("Gradient of loss with respect to inputs", gradients[2].numpy())


# In[72]:


import tensorflow as tf
# define functions
def func1(x):
    return x**2 + 3*x
def func2(y):
    return y**3 + 2*y

#create tensorflow variable
x = tf.Variable(2.0)
y = tf.Variable(3.0)

# use tf.Gradienttape to compute gradients
with tf.GradientTape(persistent=True) as tape:
    z1 = func1(x)
    z2 = func2(y)
# calculate gradient
grad_x = tape.gradient(z1, x)
grad_y = tape.gradient(z2, y)
# print the cimputed gradients
print("gradient of func1 with respect to x", grad_x.numpy())
print("gradient of func2 with respect to y", grad_y.numpy())


# In[52]:


import tensorflow as tf
# define the function
def matrix_multiply(x, w):
    return tf.matmul(x, w)
def add_bias(x, b):
    return x + b
def relu(x):
    return tf.nn.relu(x)
# sample data
inputs = tf.constant([[1.0, 2.0],
                     [3.0, 2.0],
                     [6.0, 8.0]])
weights = tf.constant([[5.5], [3.8]])
print(weights)
bias = tf.constant([.1, 5.0])
targets = tf.constant([[0.0]])
print(inputs)

# initialize variables for weights and bias
w = tf.Variable(weights)
b = tf.Variable(bias)

# create tensorflow var (inputs)
x = tf.Variable(inputs)
                     #forward pass
with tf.GradientTape(persistent=True) as tape:
    # chain function
    z1 = matrix_multiply(x, w)
    z2 = add_bias(z1, b)
    z3 = relu(z2)
weights2 = w2 = tf.constant([[7.5], [8.8]])
bias2 = b2 = tf.constant([2.1, 8.0])
def matrix_multiply(z3, w2):
    return tf.matmul(z3, w2)
def add_bias(z3, b2):
    return z3 + b2
    j1 = matrix_multiply(z3, w2)
    j2 = add_bias(z3, b2)
    j3 = relu(j2)
    loss = tf.reduce_mean(tf.square(z3 - targets))
    loss = tf.reduce_mean(tf.square(j3 - targets))
    # compute gradients
gradients=  tape.gradient(loss, [w, b, x])
# print the gradient
print("Gradient of loss with respect to weight", gradients[0].numpy())
print("Gradient of loss with respect to bias", gradients[1].numpy())
print("Gradient of loss with respect to inputs", gradients[2].numpy())
# compute gradients
gradients=  tape.gradient(loss, [w2, b2, z3])
# print the gradient
print("Gradient of loss with respect to weight", gradients[0].numpy())
print("Gradient of loss with respect to bias", gradients[1].numpy())
print("Gradient of loss with respect to inputs", gradients[2].numpy())


# # Keras

# In[17]:


# lets say we expect our inputs to be GB image of arbitrary size 
from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input(shape=(None, None, 3))
# centre crop images to 150x150
class CenterCrop(layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(CenterCrop, self).__init__(**kwargs)
        self.height = height
        self.width = width

    def build(self, input_shape):
        super(CenterCrop, self).build(input_shape)

    def call(self, inputs):
        # Your implementation for center crop goes here
        # This could involve cropping the input tensor to the desired height and width
        # Example: cropped_inputs = custom_crop_function(inputs, self.height, self.width)
        return cropped_inputs
x = CenterCrop(height=150, width=150)(inputs)
# rescale images to (0, 1)
x = Rescaling(scale=1.0 / 255)(x)
# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))()
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))()
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)
# = add a dense classifier on top
num_classes = 10
output = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
processed_data = model(data)
print(processed_data,shape)


# # keras Layer

# In[19]:


class Linear(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(seld, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
        initial_value=w_init(shape=(input_dim, units), dtypes="float32"),
        trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
        initial_value=b_init(shape=(units,), dtypes="floats32"), trainable=True
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
 


# # layer gradients

# In[50]:


# prepare a detaset
import tensorflow as tf
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

#instantiate  linear layer with 10 unts
linear_layer = Linear(10)
#instantiate a logical loss function that expects integer tarets
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# instantiate an optimizer
optimzer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# iterate over the batches of the dataset
for step, (x, y) in enumerate(dataset):
# open a gradienttape
    with tf.GradientTape() as tape:
        # forward pass
        logits = linear_layer(x)
        
        # loss value for this batch
        loss = loss_fn(y, logits)
    # get gradients of the loss wrt the weights
    gradients = tape .gradient(loss, linear_layer.trainable_weights)
    
    # update the weights of our linear layer
    optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))
    if step % 100 == 0:
        print("steo:", step, "loss", float(loss))


# # Example 1: Simple Feedforward Neutral Network

# In[68]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# generate sysnthetic data for binary classification
np.random.seed(42)
X = np.random.randn(1000, 10) # 1000 samples with one features each
print (X)
y = (X[:, 0] + X[:, 1] > 0).astype(int) #Binary labels (i for sum of first two features > 0, else o)
# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# create a simple feedforward neural network using keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=10),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=10),
])
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# create a model summary
model.summary()
# train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# evaluate the model o the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


# # Example 1: Multi input and Multi output model

# In[84]:


import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, concatenate
from keras.models import Model

# sample data for imput layer1
X_train_input1 = np.random.rand(100, 10)  # 100 samples with 10 features each

# sample data for imput layer1
X_train_input2 = np.random.rand(100, 5)  # 100 samples with 10 features each

# sample labels for the output ayer (one-hot encoder)
y_train = np.random.randint(2, size=100) #100 binary labels (0 or 1)

# define input dimensions
input_dim1 = 10
input_dim2 = 5
# define output demension (number of classes)
output_dim = 2
# define two different input layers
input_layers1 = Input(shape=(input_dim1,))
input_layers2 = Input(shape=(input_dim2,))
# define hidden layers for each layers
hidden_layers1 = Dense(64, activation='relu')(input_layers1)
hidden_layers2 = Dense(32, activation='relu')(input_layers2)
# combine the output from both hidden layers
combined = concatenate([hidden_layers1, input_layers2])
# define the output layers
output_layer = Dense(output_dim, activation='softmax')(combined) # output_dim is the number of classes
# create the model with multiple inputs and one outputs
model = Model(inputs=[input_layers1, input_layers2], outputs=output_layer)
#create the model summary
model.summary()
# compile the modoel
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# convert the labels to one hot encoding (required for categorical_crossentropy loss)
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=output_dim)

#train the model
model.fit([X_train_input1, X_train_input2], y_train_onehot, epochs=10, batch_size=32)


# # three input layers

# In[86]:


import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, concatenate
from keras.models import Model

# sample data for imput layer1
X_train_input1 = np.random.rand(100, 10)  # 100 samples with 10 features each

# sample data for imput layer2
X_train_input2 = np.random.rand(100, 5)  # 100 samples with 5 features each

# sample data for imput layer3
X_train_input3 = np.random.rand(100, 7)  # 100 samples with 7 features each

# sample labels for the output ayer (one-hot encoder)
y_train = np.random.randint(2, size=100) #100 binary labels (0 or 1)

# define input dimensions
input_dim1 = 10
input_dim2 = 5
input_dim3 = 7
# define output demension (number of classes)
output_dim = 2
# define three different input layers
input_layers1 = Input(shape=(input_dim1,))
input_layers2 = Input(shape=(input_dim2,))
input_layers3 = Input(shape=(input_dim3,))
# define hidden layers for each layers
hidden_layers1 = Dense(64, activation='relu')(input_layers1)
hidden_layers2 = Dense(32, activation='relu')(input_layers2)
hidden_layers3 = Dense(32, activation='relu')(input_layers3)
# combine the output from both hidden layers
combined = concatenate([hidden_layers1, input_layers2, input_layers3])
# define the output layers
output_layer = Dense(output_dim, activation='softmax')(combined) # output_dim is the number of classes
# create the model with multiple inputs and one outputs
model = Model(inputs=[input_layers1, input_layers2, input_layers3], outputs=output_layer)
#create the model summary
model.summary()
# compile the modoel
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# convert the labels to one hot encoding (required for categorical_crossentropy loss)
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=output_dim)

#train the model
model.fit([X_train_input1, X_train_input2, X_train_input3], y_train_onehot, epochs=10, batch_size=32)


# In[ ]:




