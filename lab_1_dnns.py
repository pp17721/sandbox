
## Building Your First Fully Connected Network

Now you've got to know the basics of pytorch, we can implement a 2-layer fully connected network (a.k.a MultiLayer Percepton) and train it using gradient descent.

First we need to load in our data. We will use [scikit-learn](https://scikit-learn.org/stable/) for this as it bundles the iris dataset.
"""

from sklearn import datasets
iris = datasets.load_iris()  # datasets are stored in a dictionary containing an array of features and targets
iris.keys()

"""The data is stored in a float64 numpy array with 150 rows of 4 columns. Each row is a data sample, in this case a flower, and each column is a feature of that data sample."""

iris['data'].shape, iris['data'].dtype

"""What do the first 15 examples look like?"""

iris['data'][:15]

"""What do each of the columns correspond to?"""

iris['feature_names']

"""The labels for the data are in a separate array called *target*.

How many classes do we have?
"""

np.unique(iris['target'])

"""What do the labels correspond to?"""

iris['target_names']

"""Let's visualise the data to see what it looks like."""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import seaborn as sns
import pandas as pd

features_df = pd.DataFrame(
    iris['data'],
    columns=iris['feature_names']
)
features_df['label'] = iris['target_names'][iris['target']]
sns.pairplot(features_df, hue='label')

"""Typically we normalise features input to networks as this helps speed learning up as the loss landscape becomes easier to traverse (for more details see slides 11-13 in [lecture 6 of Geoff Hinton's course](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf))."""

preprocessed_features = (iris['data'] - iris['data'].mean(axis=0)) / iris['data'].std(axis=0)

"""Next we want to split this into a training and testing split.

**Question**: Why do we want to do this?

First we need to shuffle the data

**Question**: Why do we need to shuffle before splitting it? Hint: take a look at the entire dataset array.
"""

from sklearn.model_selection import train_test_split

labels = iris['target']
# train_test_split takes care of the shuffling and splitting process
train_features, test_features, train_labels, test_labels = train_test_split(preprocessed_features, labels, test_size=1/3)

"""Finally, we need to take our numpy arrays and put them into tensors for processing by PyTorch."""

features = {
    'train': torch.tensor(train_features, dtype=torch.float32),
    'test': torch.tensor(test_features, dtype=torch.float32),
}
labels = {
    'train': torch.tensor(train_labels, dtype=torch.long),
    'test': torch.tensor(test_labels, dtype=torch.long),
}

"""Now we need to create a fully connected layer that takes an input $x$, and trainable weights $W$ and biases $b$ and computes

$$Wx + b$$

PyTorch has a library of common layer types including a fully connected layer, its class name is `Linear` as the layer produces a linear transformation of the input data.

We have a single fully connected layer, but we want to stack these to produce a neural network composed of two layers (a.k.a Multi-layer Perceptron or MLP):

* Input size: 4 features
* Hidden layer size: 100 units
* Output size: 3 classes

We need to put a non-linear function in between these two layers as otherwise the transformation is just as powerful in representational capacity as a linear classifier. We want to produce non-linear decision boundaries as these will better fit our data.

Now we can define a MLP class that brings together 2 fully connected layers with a ReLU on the output of the first layer.
"""

from torch import nn
from torch.nn import functional as F
from typing import Callable


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        return x

"""Let's deconstruct the signature of the `forward` method

```python
def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    ...
```

`inputs: torch.Tensor` says that the parameter `inputs` is of type `torch.Tensor`. The return type of the method is denoted by `-> torch.Tensor` stating that the method returns a tensor.

We use a generic type `Callable` which defines the type of a function, it has the format `Callable[[args], return_type]`. `activation_fn: Callable[[torch.Tensor], torch.Tensor]` means that `activation_fn` should be a function that takes in a single argument of type `torch.Tensor` and returns a `torch.Tensor`. We've also defined the default value of this parameter to be `F.relu` which is the functional implementation of a rectified linear unit in PyTorch.

Now we can instantiate the MLP class for our problem:
"""

feature_count = 4
hidden_layer_size = 100
class_count = 3
model = MLP(feature_count, hidden_layer_size, class_count)

"""Our model produces a scalar value for each class for each example propagated through the network. We need to squash these values into a pseudo-probability distribution. We can do that with the softmax distribution $\mathrm{softmax} : \mathbb{R}^n \rightarrow [0, 1]^n$. It is defined as follows

$$\mathrm{softmax}(\mathbf{x})_i = \frac{e^{\mathbf{x}_i}}{\sum_j e^{\mathbf{x}_j}}$$

This definition results in the output vector summing to one: $$\sum_i\mathrm{softmax}(\mathbf{x})_i = 1$$

To train our network we need some way to measure the error between the output of the network $\mathbf{\hat{p}} : [0, 1]^C$ where $C$ is the number of classes and the label encoded into a one-hot representation $y: \{0, 1\}^C$. We measure the cross-entropy between them:

$$\mathrm{CE}(\mathbf{p}, y) = - \sum_c^C y_c \log(p_c)$$

We'll run a forward pass through the network to compute its predictions which we can then use to compute the loss function.
"""

logits = model.forward(features['train'])
logits.shape

"""As you can see `logits` has a shape of (100,3). For each of the 100 data samples we have 3 outputs, one per class. A higher output value (relative to the other values for that data sample) indicates that the model is predicting that class as being more likely.

**Task:** Compute the loss of the `logits` against the training labels `labels['train']` using the [`nn.CrossEntropyLoss`](https://pytorch.org/docs/1.2.0/nn.html#torch.nn.CrossEntropyLoss) class which combines the softmax and cross entopy functions into a single operation. Save this in a variable called `loss`. 

Note that you will have to instantiate the class before you can call it on your logits and labels, like so:
"""

loss_function = nn.CrossEntropyLoss()
loss=loss_function(logits,labels['train'])

"""Then you need to call `loss_function` with the `logits` and `labels['train']` to compute the loss

We can now compute the model parameters' gradients by calling `backward()` on the loss.
"""

loss.backward()

"""The gradients will be computed and propagated back through the network.

We want to evaluate the quality of our networks predictions, accuracy is an informative metric for a classification task on a balanced dataset.

**Task:** Implement a function to compute accuracy with the following signature

```python
def accuracy(probs: torch.FloatTensor, targets: torch.LongTensor) -> float:
    '''
    Args:
        probs: A float32 tensor of shape ``(batch_size, class_count)`` where each value 
            at index ``i`` in a row represents the score of class ``i``.
        targets: A long tensor of shape ``(batch_size,)`` containing the batch examples'
            labels.
    '''
    ## First work out which class has been predicted for each data sample. Hint: use argmax
    ## Second count how many of these are correctly predicted
    ## Finally return the accuracy, i.e. the percentage of samples correctly predicted
```    

Your implementation should *not* use any `for` loops, instead you should use the operations defined on tensors like `argmax` and `sum`.

We've also provided some test cases below to verify the correctness of your implementation of `accuracy`.
"""

def accuracy(probs, labels):
  tot=0
  for id in range(len(probs)):
    if probs[id].argmax()==labels[id]:
      tot+=1
  return tot/len(probs)
def check_accuracy(probs: torch.FloatTensor,
                   labels: torch.LongTensor,
                   expected_accuracy: float):
    actual_accuracy = float(accuracy(probs, labels))
    assert actual_accuracy == expected_accuracy, f"Expected accuracy to be {expected_accuracy} but was {actual_accuracy}"

check_accuracy(torch.tensor([[0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1]]),
               torch.ones(5, dtype=torch.long),
               1.0)
check_accuracy(torch.tensor([[1, 0],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1]]),
               torch.ones(5, dtype=torch.long),
               0.8)
check_accuracy(torch.tensor([[1, 0],
                             [1, 0],
                             [0, 1],
                             [0, 1],
                             [0, 1]]),
               torch.ones(5, dtype=torch.long),
               0.6)
check_accuracy(torch.tensor([[1, 0],
                             [1, 0],
                             [1, 0],
                             [1, 0],
                             [1, 0]]),
               torch.ones(5, dtype=torch.long),
               0.0)
print("All test cases passed")

"""We have our network and a way of computing the error of its output with respect to labels. Now we just need something to optimize the network's weights. We can use stochastic gradient descent (SGD) for this purpose. It is a simple hill descending algorithm, taking a step in the steepest downhill direction (the negative of the gradient) in order to reduce the loss.

We now implement the training that optimizes the network's parameters over the dataset repeatedly. Each iteration through the dataset is known as an *epoch*. It is typical to train networks for anywhere between tens to thousands of epochs.
"""

from torch import optim


# Define the model to optimze
model = MLP(feature_count, hidden_layer_size, class_count)

# The optimizer we'll use to update the model parameters
optimizer = optim.SGD(model.parameters(), lr=0.04)

# Now we define the loss function.
criterion = nn.CrossEntropyLoss() 

# Now we iterate over the dataset a number of times. Each iteration of the entire dataset 
# is called an epoch.
for epoch in range(0, 100):
    # We compute the forward pass of the network
    logits = model.forward(features['train'])
    # Then the value of loss function 
    loss = criterion(logits,  labels['train'])
    
    # How well the network does on the batch is an indication of how well training is 
    # progressing
    print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
        epoch,
        accuracy(logits, labels['train']) * 100,
        loss.item()
    ))
    
    # Now we compute the backward pass, which populates the `.grad` attributes of the parameters
    loss.backward()
    # Now we update the model parameters using those gradients
    optimizer.step()
    # Now we need to zero out the `.grad` buffers as otherwise on the next backward pass we'll add the 
    # new gradients to the old ones.
    optimizer.zero_grad()
    
# Finally we can test our model on the test set and get an unbiased estimate of its performance.    
logits = model.forward(features['test'])    
test_accuracy = accuracy(logits, labels['test']) * 100
print("test accuracy: {:2.2f}".format(test_accuracy))

