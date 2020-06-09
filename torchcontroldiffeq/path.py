import abc
import math
import torch


# Just a wrapper to note to Path that we want gradients wrt this tensor.
class ComputedParameter:
    """Registers a non-leaf tensor as requiring gradients when used with adjoint backpropagation. Must be assigned to
    an instance of a subclass of torchcontroldiffeq.Path.
    """
    def __init__(self, tensor):
        super(ComputedParameter, self).__init__()
        self.tensor = tensor


# Metaclass used for post-init finalisation.
class _PathMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        self = super(_PathMeta, cls).__call__(*args, **kwargs)

        for key, value in list(self._controldiffeq_computed_parameters.items()):
            # Ensures that no ComputedParameter is in the graph of another ComputedParameter. Otherwise when we
            # differentiate wrt both then we end up double-counting the one that is earlier in the chain, as it
            # eventually gets the gradient that its successor has as well.
            value = value.view(value.shape)
            setattr(self, key, ComputedParameter(value))
        self._controldiffeq_finalised = True
        return self


class Path(torch.nn.Module, metaclass=_PathMeta):
    def __init__(self, *args, **kwargs):
        super(Path, self).__init__(*args, **kwargs)
        self._controldiffeq_computed_parameters = {}
        self._controldiffeq_finalised = False

    def _del_computed_parameter(self, key):
        try:
            del self._controldiffeq_computed_parameters[key]
        except KeyError:
            pass

    def __setattr__(self, key, value):
        pre_init = not hasattr(self, '_controldiffeq_finalised')
        if isinstance(value, ComputedParameter):
            computed_parameter = True
            value = value.tensor
            if pre_init:
                # Before __init__. Just error in this case; torch.nn.Module doesn't like this case either.
                raise RuntimeError("Cannot assign ComputedParameters before super().__init__() has been called.")
            elif self._controldiffeq_finalised:
                # After both __init__ and the metaclass' __call__, at some point later in the program.

                # We prohibit changes after we've taken the views of our ComputedParameters. Otherwise it's possible
                # that one ComputedParameter is derived from another and then assigned to this instance, which will then
                # give incorrect gradients. Note that it's fine to derive one ComputedParameter from another, just as
                # long as they're not both on the same instance, as then they can't both be detected at the same time on
                # any one call to cdeint. For example this may occur when doing a multi-layer Neural CDE.
                raise RuntimeError("Cannot assign ComputedParameters after initialisation.")
        else:
            computed_parameter = False

        super(Path, self).__setattr__(key, value)

        # Done after the super() call in case the superclass wants to throw an error about assigning things, in which
        # case we shouldn't modify our state until now.
        if computed_parameter and value.grad_fn is not None:
            self._controldiffeq_computed_parameters[key] = value
        else:
            if not pre_init:
                self._del_computed_parameter(key)

    def __delattr__(self, key):
        self._del_computed_parameter(key)
        super(Path, self).__delattr__(key)

    # Don't necessarily have to implement this one to make cdeint work.
    def evaluate(self, t):
        """Evaluates the natural cubic spline interpolation at a point t. t should be a scalar tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, t):
        """Evaluates the derivative of the natural cubic spline at a point t. t should be a scalar tensor."""
        raise NotImplementedError


def validate_input(t, X):
    if not t.is_floating_point():
        raise ValueError("t must both be floating point.")
    if not X.is_floating_point():
        raise ValueError("X must both be floating point.")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional. It instead has shape {}.".format(tuple(t.shape)))
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")

    if X.ndimension() < 2:
        raise ValueError("X must have at least two dimensions, corresponding to time and channels. It instead has "
                         "shape {}.".format(tuple(X.shape)))

    if X.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t. X has shape {} and t has shape {}, "
                         "corresponding to time dimensions of {} and {} respectively."
                         .format(tuple(X.shape), tuple(t.shape), X.size(-2), t.size(0)))

    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2. It instead has shape {}, corresponding to a "
                         "time dimension of size {}.".format(tuple(t.shape), t.size(0)))
