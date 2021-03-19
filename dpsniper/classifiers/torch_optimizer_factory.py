from typing import Optional

import torch


class TorchOptimizerFactory:
    """
    A factory for torch optimizers and schedulers to be used when training classifiers.
    """

    def __init__(self, optimizer_clazz: type,
                 optimizer_args: dict,
                 scheduler_class: Optional[type],
                 scheduler_args: dict):
        """
        Construct a factory which creates instances of a fixed type using fixed arguments.

        Args:
            optimizer_clazz: the type of the optimizer
            optimizer_args: the arguments to be passed when constructing the optimizer
            scheduler_class: (optional) the type of the learning rate scheduler
            scheduler_args: the arguments to be passed when constructing the scheduler
        """
        self.optimizer_clazz = optimizer_clazz
        self.scheduler_class = scheduler_class
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def create_optimizer_with_scheduler(self, model_params)\
            -> (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler):
        """
        Creates an optimizer and learning rate scheduler.

        Args:
            model_params: the torch model parameters

        Returns:
            a pair of optimizer and associated learning rate scheduler
        """
        optimizer = self.optimizer_clazz(model_params, **self.optimizer_args)
        scheduler = None
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_args)
        return optimizer, scheduler


class AdamOptimizerFactory(TorchOptimizerFactory):

    def __init__(self,
                 learning_rate=0.1,
                 step_size=500):
        super().__init__(torch.optim.Adam, {"lr": learning_rate},
                         torch.optim.lr_scheduler.StepLR, {"step_size": step_size})


class LBFGSOptimizerFactory(TorchOptimizerFactory):

    def __init__(self,
                 learning_rate=1.0,
                 max_iter=20):
        super().__init__(torch.optim.LBFGS, {"lr": learning_rate, "max_iter": max_iter},
                         None, {})


class SGDOptimizerFactory(TorchOptimizerFactory):
    def __init__(self,
                 learning_rate=0.3,
                 momentum=0.3,
                 step_size=500):
        super().__init__(torch.optim.SGD, {"lr": learning_rate, "momentum": momentum},
                         torch.optim.lr_scheduler.StepLR, {"step_size": step_size})
