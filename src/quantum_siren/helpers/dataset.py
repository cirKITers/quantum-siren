""" Kedro Torch Model IO
Models need to be imported and added to the dictionary
as shown with the ExampleModel
Example of catalog entry:
modo:
  type: kedro_example.io.torch_model.TorchLocalModel
  filepath: modo.pt
  model: ExampleModel
"""
from os.path import isfile
from typing import Any, Dict
import torch
from kedro.io import AbstractDataSet

from quantum_siren.pipelines.training.models import Model

models = {
    "model": Model,
}


class TorchLocalModel(AbstractDataSet):
    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            model=self._model,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def __init__(
        self,
        filepath: str,
        model: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        self._filepath = filepath
        self._model = model
        if model in models:
            self._Model = Model
        else:
            raise KeyError("Add model to models.")
        default_save_args = {}
        default_load_args = {}

        self._load_args = (
            {**default_load_args, **load_args}
            if load_args is not None
            else default_load_args
        )
        self._save_args = (
            {**default_save_args, **save_args}
            if save_args is not None
            else default_save_args
        )

    def _load(self):
        state_dict = torch.load(self._filepath)
        model = self._Model(**self._load_args)
        model.load_state_dict(state_dict)
        return model

    def _save(self, model) -> None:
        torch.save(model.state_dict(), self._filepath, **self._save_args)

    def _exists(self) -> bool:
        return isfile(self._filepath)
