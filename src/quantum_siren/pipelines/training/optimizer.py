import torch
import pennylane as qml


class QNG(qml.QNGOptimizer, torch.optim.Optimizer):
    """Implementation of the Quantum Natural Gradient Optimizer

    Args:
        params: Parameters of the torch model
        qnode: QNode instance that is being used in the model
        lr: Step size/ learning rate of the optimizer
        dampening: Float for metric tensor regularization
    """

    def __init__(
        self, params, qnode, argnum, lr=0.01, dampening=0, approx="block-diag"
    ):
        # Initialize a default dictionary, we utilize this to store any optimizer related hyperparameters
        # Note that this just follows the torch optimizer approach and is not mandatory
        defaults = dict(stepsize=0.01, dampening=0)

        # Initialize the QNG optimizer
        qml.QNGOptimizer.__init__(self, lr, dampening)

        # Initialize the Torch Optimizer Base Class
        torch.optim.Optimizer.__init__(self, params, defaults)

        self.argnum = argnum

        # Initialize the metric tensor
        # Note the argnum argument which specifies the parameter indices of which we want to calculate the metric tensor
        # Also important is hybrid=False as it forces the returned metric tensor being only calculated w.r.t. the gate arguments and not w.r.t. the QNode arguments (which would include the input of the classical layer)
        self.metric_tensor_fn = qml.metric_tensor(
            qnode, approx=approx, argnum=argnum, hybrid=False
        )


        self.requires_closure = True

    def step(self, closure=None):
        """Step method implementation. We call this to update the parameter values

        Args:
            closure (Callable, optional): The closure is a helper fn that is being called by the optimizer to get the metric tensor for the current parameter configuration. Defaults to None.

        Returns:
            Tensor: Updated parameters
        """

        # Iteratate the parameter groups (i.e. parts of the model (just one in this case))
        # We obtain the param_groups variable after the torch optimizer instantiation
        for pg in self.param_groups:
            # Each group is a dictionary where the actual parameters can be accessed using the "params" key
            for p in pg["params"]:
                # p is now a set of parameters (i.e. the weights of the VQC)
                if p.grad is None:
                    continue
                # we can get the gradients of those parameters using the following line
                g = p.grad.data

                # now we call the closure (which is in fact the metric tensor function) to obtain the actual metric tensor, given the current parameter configuration.
                # note that this will cause the circuit function being calles with inputs=None and therefore the inputs from the previous forward path will be used.
                _metric_tensor = self.metric_tensor_fn(p)

                # Reshape metric tensor to be square
                shape = qml.math.shape(_metric_tensor)
                size = qml.math.prod(shape[: len(shape) // 2])
                self.metric_tensor = qml.math.reshape(_metric_tensor, (size, size))
                # Add regularization
                self.metric_tensor = self.metric_tensor + self.lam * qml.math.eye(
                    size, like=_metric_tensor
                )

                # the apply_grad method requires to have only numpy arrays, that's why we have to call .detach().numpy() on our torch tensors here and in the following lines
                self.metric_tensor = self.metric_tensor.detach().numpy()

                # this cuts the metric tensor such that it results in the size of n_weights x n_weights
                # therefore, we assume that the first rows and cols are reserved for the input which is reasonable as they are all-zero
                if self.argnum is not None:
                    self.metric_tensor = self.metric_tensor[
                        self.argnum[0] :, self.argnum[0] :
                    ]

                # with the current parameters p and their gradients g, we can call the apply_grad method from the Pennylane optimizer which returns the updated parameter configuration. This will overwrite the current parameter configuration which the model will use in the next iteration. Remember, the parameters here are just references to the actual parameters within our model.
                # again, we have to first convert to numpy and then back to torch tensors due to the incompability with Pennylane's implementation. This should be fixed in future.
                p.data = torch.tensor(
                    self.apply_grad(g.detach().numpy(), p.detach().numpy()),
                    requires_grad=True,
                )

        # unwrap from list if one argument, cleaner return
        if len(p) == 1:
            return p[0]

        return p
