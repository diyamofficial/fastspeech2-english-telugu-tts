# Adapted from ming024/FastSpeech2 (MIT) model/optimizer.py
# Noam (Vaswani et al., 2017) schedule with warm-up, plus step-wise annealing,
# as in Appendix B of the FastSpeech 2 paper / the reference implementation.
import numpy as np
import torch


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=float(train_config["optimizer"]["eps"]),
            weight_decay=float(train_config["optimizer"]["weight_decay"]),
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad(set_to_none=True)

    def load_state_dict(self, path_or_state):
        self._optimizer.load_state_dict(path_or_state)

    def state_dict(self):
        return self._optimizer.state_dict()

    def current_lr(self):
        return self.init_lr * self._get_lr_scale(max(self.current_step, 1))

    def _get_lr_scale(self, step):
        lr = np.min(
            [
                np.power(step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * step,
            ]
        )
        for s in self.anneal_steps:
            if step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale(self.current_step)

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
