
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

"""**Task:** Try changing the learning rate `lr` passed in to the `optimizer` and see how this effects accuracy on the test set.

---
## Porting Your Network To BC4

Up until this point you've been running your code in a Jupyter notebook on Colaboratory. Whilst this is great for exploration and fiddling with things, it doesn't work well with the way HPC systems work, and also is hard to keep track of changes. We'll take what we've developed so far, and produce a Python script that can train a network and evaluate it.

Copy and paste all the code in the notebook cells needed to run the previous code cell, where we defined the training loop, into a script in your favourite editor and save it with the name `train_fully_connected.py`.

Let's make a folder on BC4 to house all these files rather than putting them into your home directory. 

```console
$ ssh bc4-external
[bc4] $ mkdir -p ~/adl/lab-1
[bc4] $ exit
$ scp train_fully_connected.py bc4-external:~/adl/lab-1/
```

Now that you've copied your script to BC4 we can run an interactive session to gain access to a compute node with a GPU

```console
$ ssh bc4-external
[bc4] $ srun --partition gpu --gres gpu:1 --account comsm0045 --time 0-00:15 --mem=64GB --reservation comsm0045-lab1 --pty bash
[bc4-compute-node] $
```

Now let's run our code, to do so we'll have to ensure we have the software set up:

```console
[bc4-compute-node] $ module load languages/anaconda3/2019.07-3.6.5-tflow-1.14
```

And now run the code

```
[bc4-compute-node] $ cd ~/adl/lab-1
[bc4-compute-node] $ python train_fully_connected.py
```

And remember to be a good HPC citizen and give up the compute node as soon as you're finished with it so others can use it:

```
[bc4-compute-node] $ exit   # Exit interactive session on BC4 compute node
[bc4] $ 
```

Up until this point, we have been executing all our code on a CPU. It's time to actually take advantage of those GPUs! PyTorch has a concept of a _device_, this is some piece of hardware that has both memory and compute capabilities. You can transfer tensors and models onto devices using the `to(device)` method defined on `torch.Tensor` and `nn.Module` objects.

**Task:** Modify your code so that

1. Define a device for the computation `device = torch.device('cuda')` at the start of your code
2. Move model to the GPU: `model = model.to(device)` after you define `model = MLP(...)`
3. Move train features to the gpu: `features["train"] = features["train"].to(device)` after defining `features`
4. Move test features to the gpu: `features["test"] = features["test"].to(device)`
5. Move labels to the gpu: `labels["train"] = labels["train"].to(device)` and `labels["train"] = labels["test"].to(device)`

Now rerun the code and see that it runs a little faster. The speed is not that noticeable for such a small network, but when we move to larger networks you'll find the GPU orders of magnitude quicker.

If you get an error similar to `Expected object of device type cuda but got device type cpu` then either the model or the features/labels were not copied over to the gpu. Below we list the common errors you might encounter at this stage and the reasons for these errors. Feel free to ask a TA for help understanding the error you are getting.

`RuntimeError: Expected object of device type cuda but got device type cpu for argument #1 'self'...`\
Your model may still be on the cpu. Make sure you aren't redefining `model` after moving it to the gpu.

`RuntimeError: Expected object of device type cuda but got device type cpu for argument #2 'mat1'...`\
This means `features['train']` or `features['test']` is likely still on the cpu.

`RuntimeError: Expected object of device type cuda but got device type cpu for argument #2 'target'...`\
This means `labels['train']` is likely still on the cpu.

`RuntimeError: expected device cuda:0 but got device cpu`\
Depending on your implementation, this may mean `labels['test']` is still on the cpu

`NameError: name 'device' is not defined`\
Make sure `device = torch.device('cuda')` is before any of the other lines which use `device`

---

## Logging Performance Metrics

To monitor training, use Tensorboard, a real-time graphing tool to emerge from the TensorFlow project. With tensorboard you can log metrics in real time from a python script and visualise them in a web browser.

PyTorch has native support for tensorboard (as of 1.1.0). There are two parts to tensorboard
1. A `SummaryWriter`, this is an object you'll instantiate in your code, you can use it to log tensors, scalars, images, audio, and more. It will serialise and write these objects to disk into a log directory.
2. The `tensorboard` executable which launches a web server that provides an interface graphing the data that you have logged.

Let's log the training accuracy and loss.

First we need to import `SummaryWriter` and instantiate it.

```python
from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter('logs', flush_secs=5)
```

Now **within the training loop**, log the scalar accuracy value and training loss:

```python
train_accuracy = accuracy(logits, labels['train']) * 100
summary_writer.add_scalar('accuracy/train', train_accuracy, epoch)
summary_writer.add_scalar('loss/train', loss.item(), epoch)
```

Close the writer **outside** the training loop

```python
summary_writer.close()
```

Now run your code again, you should have a `logs` directory in your working directory. This contains the values written by the summary writer. We can visualise these using `tensorboard`. 

To do so, we'll run `tensorboard` on a BC4, but we'll need to forward the TCP port to your own computer. 

```console
[bc4-compute-node] $ PORT=$((($UID-6025) % 65274))
[bc4-compute-node] $ echo $PORT
<PORT>
[bc4-compute-node] $ hostname -s
<HOSTNAME>
[bc4-compute-node] $ tensorboard --logdir logs --port $PORT
TensorBoard 1.14.0 at http://<HOSTNAME>.acrc.bris.ac.uk:<PORT>/ (Press CTRL+C to quit)
```


**Note** If you get an issue with `tensorboard: command not found`, this is because you haven't loaded the unit's module. Simply run
```console
[bc4] $ module load languages/anaconda3/2019.07-3.6.5-tflow-1.14  
```
to load it.

To avoid having to do this every time you log in to BC4, append it to your `~/.bashrc` file by running

```console
[bc4] $ echo "module load languages/anaconda3/2019.07-3.6.5-tflow-1.14" > ~/.bashrc
```

Open up a new console window on your laptop and run the following, replacing `<HOSTNAME>` and `<PORT>` with those obtained above.

```console
$ ssh -N -L 6006:<HOSTNAME>:<PORT> bc4-external
```

- `-N` tells SSH not to create a new terminal session on the remote host
- `-L <local-port>:<remote-address>:<remote-port>` tells SSH to forward traffic on port `<local-port>` on your machine to port `<remote-port>` on the machine at `<remote-address>`.

Now visit http://localhost:6006, you should see something similar to the screen below

![Tensorboard landing page](./media/tensorboard-landing-page.png)

The x-axis is the number of steps we've trained for.

By default tensorboard smoothes your data by computing a running average. You can adjust this smoothing using the slider in the left side bar. We'd recommend turning this off to begin with as the smoothing can be deceptive and hide issues with training.

**Congratulations.** 

You've trained your first NN model. This concludes the first lab.

# END of Lab 1

If you would like to learn more, we offer additional extensions below.

---

# Optional Extension: Implementing library functions

Read on if you wish to open the black box and better understand how the library functions you used are implemented.

## Implementing ReLU 
We used the ReLU function as the non linearity in our network.

$$\mathrm{ReLU}(\mathbf{x}) = \max(\mathbf{x}, \mathbf{0})$$

How is this implemented?

First we compute a binary tensor indicating which elements are less than 0 by doing an element-wise comparison. We then use this as a mask to set those elements of the vector to 0.
"""

def relu(inputs: torch.Tensor) -> torch.Tensor:
    # We take a copy of the input as otherwise we'll be modifying it in
    # place which makes it harder to debug.
    outputs = inputs.clone()  
    outputs[inputs < 0] = 0
    return outputs

"""**Task:** Test the `relu` function with a randomly generated tensor to check that all the values that are less than 0 are set to 0. Additionally, check that the input to the `relu` function wasn't modified (i.e. a new tensor was produced). This is a common implementation bug where you pass a tensor to a function expecting it to be idempotent, but it actually modifies its input. It is more memory efficient to compute ReLU in place but you have to be careful that you don't mind updating the tensor input to the function.

## Implementing Softmax and Cross entropy
We used the softmax function to squash our raw network outputs from the range $\mathbb{R}^n \rightarrow [0, 1]^n$. Implementing softmax is quite interesting as the naive implementation that is a direct translation of the equation suffers from numerical stability problems.

**Task:** Implement softmax and cross entropy exactly as they are defined below with the signatures `softmax(logits: torch.Tensor) -> torch.Tensor` and `cross_entropy(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor`


$$\mathrm{softmax}(\mathbf{x})_i = \frac{e^{\mathbf{x}_i}}{\sum_j e^{\mathbf{x}_j}}$$

$$\mathrm{CE}(\mathbf{p}, y) = - \sum_c^C y_c \log(p_c)$$

Replace the definition of `criterion` in the training loop with 
```python
criterion = lambda logits, ys: cross_entropy(softmax(logits), ys)
```

### Supplementary material

The following notebooks walk through *optional* material.

1. [Local Jupyter Lab setup](../misc/local-environment-setup.ipynb) - How to install Jupyter and set up your environment for exploring pytorch on your own machine.
2. [Autodifferentiation in pytorch](../misc/autograd-explanation.ipynb) - An explanation of how PyTorch performs auto differentiation.
"""