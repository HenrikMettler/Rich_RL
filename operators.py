from cgp.node import OperatorNode


class Const05Node(OperatorNode):
    """Node with a 0.5 as output."""

    _arity = 0
    _def_output = "0.5"
    _def_numpy_output = "np.ones(len(x[0])) * 0.5"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * 0.5"


class Const2Node(OperatorNode):
    """Node with a 2.0 as output."""

    _arity = 0
    _def_output = "2.0"
    _def_numpy_output = "np.ones(len(x[0])) * 2.0"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * 2.0"


