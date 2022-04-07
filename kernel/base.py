"""Provide the base class for logging and compatibility with scikit-learn API.
"""
import abc
import collections
import inspect
import logging
import pickle
import sys
from typing import Dict, Optional


def get_logger(
        name: str, level: str,
        formatter: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        file: Optional[str] = None
) -> logging.Logger:
    """Configures and returns a logger sending messages to standard error

    Parameters
    ----------
    name : str
        Name of the created logger, to be displayed in the header of
        log messages.
    level : str
        The minimum log level handled by the logger (any message above
        this level will be ignored). Must be 'debug', 'info',
        'warning' or 'error'. Default to 'warning'.
    formatter : str
        A string to format the log messages, see
        https://docs.python.org/3/library/logging.html#formatter-objects. By
        default display level and message. Use '%(asctime)s -
        %(levelname)s - %(name)s - %(message)s' to display time,
        level, name and message.
    file : str, optional
        If provided, the name of the logging file associated.

    Returns
    -------
    logging.Logger
        A configured logging instance displaying messages to the
        standard error stream.

    Raises
    ------
    ValueError
        If the logging `level` is not 'debug', 'info', 'warning' or
        'error'.
    """
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR}

    if file is not None:
        handler = logging.FileHandler(file)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(formatter))

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.addHandler(handler)

    try:
        logger.setLevel(levels[level])
    except KeyError:
        raise ValueError(
            'invalid logging level "{}", must be in {}'.format(
                level, ', '.join(levels.keys())))
    return logger


class Base(abc.ABC):
    def __init__(self) -> None:
        """Base class. Provide logging and save methods.
        Also provide method to ensure compatibility with the
        scikit-learn API.
        """
        self._logger = get_logger(self.name, level='info')

    def __repr__(self) -> str:
        return self.__class__.__name__

    @abc.abstractproperty
    def name(self) -> str:
        """Class name"""

    @property
    def log(self) -> logging.Logger:
        """Access the logger"""
        return self._logger

    def save(self, out: str) -> None:
        """Save the object using pickle.

        Parameters
        ----------
        out : str
            Output file.
        """
        with open(out, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.log.info(f'Saved {self.name} at {out}.')

    def set_logger(self, level: str,
                   formatter: str = '%(levelname)s - %(name)s - %(message)s',
                   file: Optional[str] = None) -> None:
        """Change level and/or format of the logger

        Parameters
        ----------
        level : str
            The minimum log level handled by the logger (any message above this
            level will be ignored). Must be 'debug', 'info', 'warning' or
            'error'.
        formatter : str, optional
            A string to format the log messages.
        file : str, optional
            If provided, the name of the logging file associated.

        """
        self._logger = get_logger(self.name, level, formatter, file)

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the object"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:  # pragma: nocover
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for param in parameters:
            if param.kind == param.VAR_POSITIONAL:
                raise RuntimeError(
                    f'Using `Base` you should always '
                    f'specify the parameters in the signature '
                    f'of the __init__ (no varargs). '
                    f'{cls} with constructor {init_signature} does not '
                    f'follow this convention.')

        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this processor and
            contained subobjects that are processors. Default to True.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If any given parameter in ``params`` is invalid.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = collections.defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError(
                    f'invalid parameter {key} for processor {self}, '
                    f'check the list of available parameters '
                    f'with `processor.get_params().keys()`.')

            if delim:
                nested_params[key][sub_key] = value
            else:
                try:
                    setattr(self, key, value)
                except AttributeError:
                    raise ValueError(f'cannot set attribute {key} for {self}')
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
