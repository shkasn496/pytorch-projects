import numpy as np

np.random.seed(3)

#############################################################
####################### INSTRUCTIONS ########################
#############################################################
"""
Goal: Implement a simple Neural Network to for binary classification over
2-dimensional input data. The network must have 1 hidden layer containing 4 units.

The network should utilize mini-batch training, a sigmoid activation function, and
MSE loss function. The batch size is set in the 
"Parameters" section.

Please fill in the 4 functions in the section labeled "Fill In These Functions."
The function signatures must not change, and must return appropriate outputs based 
on the in-line comments within them. You may add additional functions as you see fit.
You may leverage the functions in the "Utilities" section if you find it necessary.
You may change N, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS if it helps you train your 
network. Do not modify the code in the section labeled "Do Not Modify Below."
Code in this section will call your functions, so make sure your implementation
is compatible.

Your code must run (you can test it by clicking "Run" button in the top-left).
The "train" method will train your network over NUM_EPOCHS epochs, and print a 
mean-squared error over the hold-out set after each epoch.

Please feel free to add extra print statements if it helps you debug your code.

This exercise is open-book. You may leverage resources you find on the Internet, 
such as syntax references, mathematical formulae, etc., but you should not adapt 
or otherwise use existing implementation code.

"""

#############################################################
######################### PARAMETERS ########################
#############################################################
N = 1000
LEARNING_RATE = 1
BATCH_SIZE = 5
NUM_EPOCHS = 10
INPUT_WIDTH = 2
HIDDEN_LAYER_WIDTH = 4
OUTPUT_LAYER_WIDTH = 1
HIDDEN_LAYER_WEIGHTS_SHAPE = (HIDDEN_LAYER_WIDTH, INPUT_WIDTH)
HIDDEN_LAYER_BIASES_SHAPE = (HIDDEN_LAYER_WIDTH, 1)
OUTPUT_LAYER_WEIGHTS_SHAPE = (OUTPUT_LAYER_WIDTH, HIDDEN_LAYER_WIDTH)
OUTPUT_LAYER_BIASES_SHAPE = (OUTPUT_LAYER_WIDTH, 1)
INITIAL_HIDDEN_LAYER_WEIGHTS = np.random.random(HIDDEN_LAYER_WEIGHTS_SHAPE)
INITIAL_HIDDEN_LAYER_BIASES = np.random.random(HIDDEN_LAYER_BIASES_SHAPE)
INITIAL_OUTPUT_LAYER_WEIGHTS = np.random.random(OUTPUT_LAYER_WEIGHTS_SHAPE)
INITIAL_OUTPUT_LAYER_BIASES = np.random.random(OUTPUT_LAYER_BIASES_SHAPE)

#############################################################
######################### UTILITIES #########################
#############################################################
def sigmoid(z):
    # activation function
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # derivative of activation function
    return sigmoid(z) * (1 - sigmoid(z))


def assert_has_shape(val, expected_shape):
    assert val.shape == expected_shape, f"Expected {val} to have shape {expected_shape}, but was {val.shape}."


#############################################################
################## FILL IN THESE FUNCTIONS ##################
#############################################################
def compute_hidden_layer_weighted_input(x, hidden_layer_weights, hidden_layer_biases):
    # return the weighted inputs (before applying sigmoid) for layer 1 as a 4x1 matrix
    assert_has_shape(x, (2, 1))
    assert_has_shape(hidden_layer_weights, HIDDEN_LAYER_WEIGHTS_SHAPE)
    assert_has_shape(hidden_layer_biases, HIDDEN_LAYER_BIASES_SHAPE)

    # fill in
    ''' return np.random.random((HIDDEN_LAYER_WIDTH, 1)) '''
    return np.dot(hidden_layer_weights, x) + hidden_layer_biases # 4x1 matrix


def compute_output_layer_weighted_input(
    hidden_layer_activation, output_layer_weights, output_layer_biases
):
    # return the weighted inputs (before applying sigmoid) for output layer as a 1x1 matrix
    assert_has_shape(hidden_layer_activation, (HIDDEN_LAYER_WIDTH, 1))
    assert_has_shape(output_layer_weights, OUTPUT_LAYER_WEIGHTS_SHAPE)
    assert_has_shape(output_layer_biases, OUTPUT_LAYER_BIASES_SHAPE)

    # fill in
    ''' return np.random.random((OUTPUT_LAYER_WIDTH, 1)) '''
    return np.dot(output_layer_weights, hidden_layer_activation) + output_layer_biases #1x1 matrix


def compute_gradients(
    x,
    y,
    hidden_layer_weights,
    hidden_layer_biases,
    hidden_layer_weighted_input,
    output_layer_weights,
    output_layer_biases,
    output_layer_weighted_input,
):
    # x, y is a single training example
    # for a single training example, return the gradient of loss with respect to each layer's weights and biases
    # return value should be a tuple of lists, where the first element is the list of weight gradients,
    # and the second is the list of bias gradients. the shape of each "gradient" should correspond to the shape of the
    # weight/bias matrix it will be used to update.
    assert_has_shape(x, (2, 1))
    assert_has_shape(hidden_layer_weights, HIDDEN_LAYER_WEIGHTS_SHAPE)
    assert_has_shape(hidden_layer_biases, HIDDEN_LAYER_BIASES_SHAPE)
    assert_has_shape(output_layer_weights, OUTPUT_LAYER_WEIGHTS_SHAPE)
    assert_has_shape(output_layer_biases, OUTPUT_LAYER_BIASES_SHAPE)

    # fill in
    '''
    weight_gradients = [
        np.zeros((HIDDEN_LAYER_WIDTH, INPUT_WIDTH)),
        np.zeros(OUTPUT_LAYER_WIDTH, HIDDEN_LAYER_WIDTH),
    ]
    bias_gradients = [
        np.zeros((HIDDEN_LAYER_WIDTH, 1)),
        np.zeros((OUTPUT_LAYER_WIDTH, 1)),
    ]
    '''

    # Forward propagation
    W2 = output_layer_weights
    Z1 = hidden_layer_weighted_input # 4x1 matrix H1 * X + B1
    A1 = sigmoid(Z1) #4x1
    Z2 = output_layer_weighted_input # 1x1 matrix O1 * A1 + B2
    y_hat = sigmoid(Z2)

    # Gradients for output layer weights and bias
    '''
    dl/dw2 = dl/dyhat * dyhat/dz2 * dz2/dw2
    dl/db2 = dl/dyhat * dyhat/dz2 * dz2/db2

    dz2_db2 = 1
    '''
    dl_dyhat = y_hat - y # 1x1
    dyhat_dz2 = np.dot(y_hat, (1 - y_hat)) # 1x1
    dz2_dw2 = A1 # 4x1
    
    common_gradient = np.dot(dl_dyhat, dyhat_dz2) # 1x1

    dl_dw2 = np.dot(common_gradient, dz2_dw2.T) # 1x4
    dl_db2 = common_gradient # 1x1

    # Gradients for hidden layer weights and bias
    '''
    dl/dw1 = (dl/dyhat * dyhat/dz2) * dz2/da1 * da1/dz1 * dz1/dw1
    dl/db1 = (dl/dyhat * dyhat/dz2) * dz2/da1 * da1/dz1 * dz1/db1

    (dl/dyhat * dyhat/dz2) = common_gradient
    dz1_db1 = 1
    '''
    dz2_da1 = W2 # 1x4
    da1_dz1 = np.dot(A1, (1 - A1).T) # (4, 4)
    dz1_dw1 = x # 2x1
    
    temp1 = np.dot(common_gradient, dz2_da1) #1x4
    temp2 = np.dot(temp1, da1_dz1) # 1x4

    dl_dw1 = np.dot(temp2.T, dz1_dw1.T) # 4x2
    dl_db1 = temp2.T #4x1

    weight_gradients = [dl_dw1, dl_dw2] # 4x2, 1x4
    bias_gradients = [dl_db1, dl_db2] # 4x1, 1x1
    return weight_gradients, bias_gradients

def get_new_weights_and_biases(
    training_batch,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    # training_batch is a list of (x, y) training examples
    # return the new weights and biases after processing this batch of data, and according to LEARNING_RATE

    # fill in
    new_weights = [hidden_layer_weights, output_layer_weights] #4x2, 1x4
    new_biases = [hidden_layer_biases, output_layer_biases] #4x1, 1x1

    batch_size = len(training_batch)
    batch_loss = 0.0
    # stochastic gradient descent as single samples are being passed to model
    for x, y in training_batch:
        # forward pass
        z1 = compute_hidden_layer_weighted_input(x, hidden_layer_weights, hidden_layer_biases)
        a1 = sigmoid(z1)
        z2 = compute_output_layer_weighted_input(a1, output_layer_weights, output_layer_biases)
        y_hat = sigmoid(z2)

        # loss calculation
        batch_loss+=(y_hat - y)**2

        # backward pass
        weight_gradients, bias_gradients = compute_gradients(
            x=x,
            y=y,
            hidden_layer_weights=hidden_layer_weights,
            hidden_layer_biases=hidden_layer_biases,
            hidden_layer_weighted_input=z1,
            output_layer_weights=output_layer_weights,
            output_layer_biases=output_layer_biases,
            output_layer_weighted_input=y_hat,
        )

        #update gradients
        # 1. Hidden layer weights
        new_weights[0] -= LEARNING_RATE * weight_gradients[0]
        # 2. Hidden layer bias
        new_biases[0] -= LEARNING_RATE * bias_gradients[0]
        # 3. Output layer weights
        new_weights[1] -= LEARNING_RATE * weight_gradients[1]
        # 4. output layer bias
        new_biases[1] -= LEARNING_RATE * bias_gradients[1]


    print("Batch loss: ", batch_loss/batch_size)    

    return new_weights, new_biases

#############################################################
#################### DO NOT MODIFY BELOW ####################
#############################################################
def predict(
    x,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    hidden_layer_activation = sigmoid(compute_hidden_layer_weighted_input(x, hidden_layer_weights, hidden_layer_biases))
    output_layer_activation = sigmoid(compute_output_layer_weighted_input(hidden_layer_activation, output_layer_weights, output_layer_biases))
    return output_layer_activation[0][0]

def train(
    X,
    Y,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    # X is an array of (2 x 1) input instances
    # Y is an array of scalar targets
    for batch_start in range(0, len(X), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        x_batch = X[batch_start:batch_end]
        y_batch = Y[batch_start:batch_end]
        batch = list(zip(x_batch, y_batch))

        new_weights, new_biases = get_new_weights_and_biases(
            batch,
            hidden_layer_weights,
            hidden_layer_biases,
            output_layer_weights,
            output_layer_biases,
        )
        hidden_layer_weights, output_layer_weights = new_weights
        hidden_layer_biases, output_layer_biases = new_biases

    # return the final weights and biases
    return (
        [hidden_layer_weights, output_layer_weights],
        [hidden_layer_biases, output_layer_biases],
    )


def compute_mse(
    X_test,
    Y_test,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    predictions = []
    for x in X_test:
        predictions.append(
            predict(
                x,
                hidden_layer_weights,
                hidden_layer_biases,
                output_layer_weights,
                output_layer_biases,
            )
        )
    y_hat = np.array(predictions)
    return np.mean((y_hat - Y_test) ** 2)


# prepare input data
X = np.random.choice([0, 1], (N, 2))
Y = np.logical_xor(X[:, 0], X[:, 1]) * 1
X = X + 0.1 * np.random.random((N, 2))
X = [np.array([x]).T for x in X]

# split into train and test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# initialize weigths, biases
hidden_layer_weights, hidden_layer_biases = (
    INITIAL_HIDDEN_LAYER_WEIGHTS,
    INITIAL_HIDDEN_LAYER_BIASES,
)
output_layer_weights, output_layer_biases = (
    INITIAL_OUTPUT_LAYER_WEIGHTS,
    INITIAL_OUTPUT_LAYER_BIASES,
)

# train over epochs, calculate MSE at each epoch
for epoch in range(NUM_EPOCHS):
    weights, biases = train(
        X_train,
        Y_train,
        hidden_layer_weights,
        hidden_layer_biases,
        output_layer_weights,
        output_layer_biases,
    )
    hidden_layer_weights, output_layer_weights = weights
    hidden_layer_biases, output_layer_biases = biases
    epoch_mse = compute_mse(
        X_test,
        Y_test,
        hidden_layer_weights,
        hidden_layer_biases,
        output_layer_weights,
        output_layer_biases,
    )
    print(f"MSE (epoch {epoch}):", epoch_mse)

print("done")
