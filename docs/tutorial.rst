Tutorial
========

.. note::
    To get through the tutorial below, we assume basic familiarity with Deep Learning and `PyTorch <https://pytorch.org/>`__. Additional familiarity with `Scikit-Learn <https://scikit-learn.org/stable/>`__ would also be helpful.

.. _model-definition:

Define the base model
---------------------

Suppose that our task is to build a CNN model for classification on the CIFAR-10 dataset. At the begining, we would like to try some traditional CNN models. For example, below is the PyTorch implementation of a modified version of LeNet-5, which is used for the "Hello, World!" program in Deep Learning: MNIST.

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()

            self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(576, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, X):
            # CONV layers
            output = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
            output = F.max_pool2d(F.relu(self.conv2(output)), (2, 2))
            output = output.view(-1, self.num_flat_features(output))

            # FC layers
            output = F.relu(self.fc1(output))
            output = F.dropout(output)
            output = F.relu(self.fc2(output))
            output = F.dropout(output)
            output = self.fc3(output)

            return output

        def num_flat_features(self, X):
            size = X.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s

            return num_features

Load the dataset
----------------

After defining the model architecture, we can now load the CIFAR-10 dataset and use it for training and evaluating. As a kind reminder, we have used commonly-used techniques on data augmentation in the code snippet below, which is quite helpful on improving the performance of our model.

.. code-block:: python

    batch_size = 128
    data_dir = "./cifar"  # the directory of the CIFAR-10 dataset

    transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=True, download=True,
                         transform=transformer),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

.. _train-and-evaluate:

Train and Evaluate
------------------

With the ``train_loader`` and ``test_loader``, now let us follow the workflow in PyTorch and train the model first.

.. code-block:: python

    epochs = 100  # training epochs

    CNN = LeNet5()
    # parameter optimizer
    optimizer = torch.optim.Adam(CNN.parameters(),
                                 lr=1e-3,  # learning rate
                                 weight_decay=5e-4)  # weight decay
    criterion = nn.CrossEntropyLoss()  # loss function

    CNN.train()
    for e in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = CNN(X_train)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

It may take a while for the model to finish training because it is iteratively trained for 100 epochs to ensure the convergence. If you like, additional printing functions could be added in the loop to report intermediate training information such as training loss.

After the training stage, we can then use ``test_loader`` to evaluate the performance of our trained CNN model.

.. code-block:: python

    CNN.eval()
    correct = 0.
    for batch_idx, (X_test, y_test) in enumerate(test_loader):
        output = F.softmax(CNN(X_test), dim=1)
        y_pred = output.data.max(1)[1]
        correct += y_pred.eq(y_test.view(-1).data).sum()

    accuracy = 100. * float(correct) / len(test_loader.dataset)

The accuracy on the ``test_loader`` is around **73%**. It looks like there is a large gap between the performance of our model and the state-of-the-arts. For example, a 18-layer ResNet could easily achieve a testing accuracy of over **93%** on CIFAR-10 dataset.

Boosting using torchensemble
----------------------------

Now, let us turn to Ensemble-PyTorch and see how far we can go. For starters, we will use a classic ensemble method implemented in Ensemble-PyTorch: **Voting**. The idea of **Voting** is quite simple:

Concretely, you can imagine each LetNet-5 model as a voter, and now we have many individual voters. Each voter in our problem will output a probability distribution on 10 classes in CIFAR-10 dataset, and all we need to do is to take the average over these probability distributions, and return the class label with the highest probability. To achieve this, let us use the API provided by Ensemble-PyTorch ``VotingClassifier``:

.. code-block:: python

    from torchensemble.voting import VotingClassifier

    model = VotingClassifier(
        estimator=LeNet5,
        n_estimators=10,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs
    )

In the code snippet above, we have used **10** LetNet-5 models (i.e., voters), and the input parameter ``LeNet5`` is simply the class of LetNet-5 defined in Section :ref:`model-definition`.

Once again, we use the ``train_loader`` and ``test_loader`` to train and evaluate the model. However, unlike the workflow in Section :ref:`train-and-evaluate`, Ensemble-PyTorch provides high-leval APIs on the training and evaluating stage, which free us from writing the loops on training and evaluating.

.. code-block:: python

    # training
    model.fit(train_loader)

    # evaluating
    accuracy = model.predict(test_loader)

The accuracy of ``VotingClassifier`` is over **80%**! In other words, we have improved the performance of LetNet-5 by a large margin without any pain on tunning the model parameters. **The only thing needs to do is to wrap your model with APIs in Ensemble-PyTorch.**

This is not the only magic can be achieved with Ensemble-PyTorch. If you are interested, please refer to `Introduction <./introduction.html>`__ for more details on Ensemble-PyTorch.

.. note::
    The running script on this tutorial is available at `examples <https://github.com/AaronX121/Ensemble-Pytorch/tree/master/examples>`__.