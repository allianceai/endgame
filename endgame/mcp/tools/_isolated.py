"""Run GPU foundation models in a worker process, so the server gives their GPU memory back when it moves on.

A long MCP session keeps every trained model. In-process, each foundation model would keep its weights on the GPU (and
some packages also cache their network for the life of the process), so a handful of them fill an 8 GB card. Instead,
on a CUDA machine:

- ``train_model``'s cross-validation of a foundation model runs in a child process that exits when it is done;
- the session stores an ``IsolatedClassifier`` / ``IsolatedRegressor``; its first ``predict``/``predict_proba``
  starts a worker that fits the model once and then serves calls on it;
- at most one worker lives at a time: training anything, or using a different isolated model, stops it first
  (``release``), and its process exit returns all of its GPU memory.

The cost is a refit when a stored model is used again after another one ran: seconds for in-context models, a minute
or two for the fine-tuned ones (Mitra, iLTM). Refits use the same parameters and seeds.
"""

from __future__ import annotations

import atexit
import importlib.util
import multiprocessing as mp
import os

import numpy as np

_WORKER: dict = {}      # the one live worker: owner, proc, conn


def should_isolate(info) -> bool:
    """Foundation models on a CUDA machine run in a worker process."""
    if info.family != "foundation" or importlib.util.find_spec("torch") is None:
        return False
    import torch
    return torch.cuda.is_available()


def _reply(conn, status, value):
    try:
        conn.send((status, value))
    except Exception as exc:    # an unpicklable result or exception
        conn.send(("err", RuntimeError(f"{type(value).__name__}: {value}" if status == "err" else f"unpicklable: {exc}")))


def _child(conn, fn, args, serve):
    """Reply with fn(*args); with ``serve``, keep the result (a fitted model) and answer (method, X) calls on it."""
    try:
        value = fn(*args)
    except BaseException as exc:
        return _reply(conn, "err", exc)
    _reply(conn, "ok", None if serve else value)
    while serve:
        try:
            msg = conn.recv()
        except EOFError:        # the server went away
            return
        if msg is None:
            return
        try:
            _reply(conn, "ok", getattr(value, msg[0])(msg[1]))
        except BaseException as exc:
            _reply(conn, "err", exc)


def _start(fn, args, serve):
    release()
    ctx = mp.get_context("spawn")       # a fresh interpreter: no CUDA state inherited
    conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child, args=(child_conn, fn, args, serve))
    # The child inherits fd 1, the MCP JSON-RPC stream; model packages print while importing, so hand it stderr.
    stdout = os.dup(1)
    os.dup2(2, 1)
    try:
        proc.start()
    finally:
        os.dup2(stdout, 1)
        os.close(stdout)
    child_conn.close()
    return proc, conn


def _receive(proc, conn):
    try:
        status, value = conn.recv()
    except EOFError:
        proc.join(5)
        raise RuntimeError(f"model worker process died (exit code {proc.exitcode})") from None
    if status == "err":
        raise value
    return value


def _stop(proc, conn):
    try:
        conn.send(None)
    except (OSError, ValueError):
        pass
    proc.join(10)
    if proc.is_alive():
        proc.kill()
        proc.join()
    conn.close()


def release():
    """Stop the live model worker, if any; its exit frees all the GPU memory it held."""
    worker = dict(_WORKER)
    _WORKER.clear()
    if worker:
        _stop(worker["proc"], worker["conn"])


atexit.register(release)    # before multiprocessing's exit handler, which would wait on a live worker forever


def run_isolated(fn, *args):
    """``fn(*args)`` in a fresh child process (``fn`` importable at module level); the child exits before this returns."""
    proc, conn = _start(fn, args, serve=False)
    try:
        return _receive(proc, conn)
    finally:
        _stop(proc, conn)


def _fit(model_name, task_type, params, X, y):
    from endgame.automl.model_registry import instantiate_model
    return instantiate_model(model_name, task_type=task_type, **params).fit(X, y)


def _fit_and_save(model_name, task_type, params, X, y, path):
    from endgame.persistence import save
    return save(_fit(model_name, task_type, params, X, y), path)


class IsolatedModel:
    """A trained registry model that is fitted and called in the worker process (see the module docstring)."""

    def __init__(self, model_name, task_type, params, X, y):
        self.model_name, self.task_type, self.params, self.X, self.y = model_name, task_type, params, X, y

    def _call(self, method, X):
        try:
            if _WORKER.get("owner") is not self:
                proc, conn = _start(_fit, (self.model_name, self.task_type, self.params, self.X, self.y), serve=True)
                _WORKER.update(owner=self, proc=proc, conn=conn)
                _receive(proc, conn)        # the fit
            _WORKER["conn"].send((method, X))
            return _receive(_WORKER["proc"], _WORKER["conn"])
        except BaseException:
            release()                       # a timeout or failure leaves the worker in an unknown state
            raise

    def fit(self, X, y):
        """Replace the training data; the next call refits (sklearn tools such as permutation_importance want ``fit``)."""
        if _WORKER.get("owner") is self:
            release()
        self.X, self.y = X, y
        return self

    def predict(self, X):
        return self._call("predict", X)

    def save(self, path):
        """Fit in a child process and save there with ``endgame.persistence.save``; returns the saved path."""
        return run_isolated(_fit_and_save, self.model_name, self.task_type, self.params, self.X, self.y, path)


class IsolatedClassifier(IsolatedModel):
    _estimator_type = "classifier"

    def __init__(self, model_name, task_type, params, X, y):
        super().__init__(model_name, task_type, params, X, y)
        self.classes_ = np.unique(y)

    def predict_proba(self, X):
        return self._call("predict_proba", X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class IsolatedRegressor(IsolatedModel):
    _estimator_type = "regressor"

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
