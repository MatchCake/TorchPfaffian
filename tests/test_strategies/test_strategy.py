import torch

from torch_pfaffian.strategies.strategy import PfaffianStrategy


class _RecordingContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class TestPfaffianStrategy:
    def test_class_attributes(self):
        assert PfaffianStrategy.NAME == "PfaffianStrategy"
        assert PfaffianStrategy.EPSILON == 1e-30

    def test_setup_context_saves_inputs_and_output(self):
        matrix = torch.eye(2, dtype=torch.float64)
        pfaffian = torch.tensor(1.0, dtype=torch.float64)
        context = _RecordingContext()
        PfaffianStrategy.setup_context(context, (matrix,), pfaffian)
        assert context.saved_tensors[0] is matrix
        assert context.saved_tensors[1] is pfaffian

    def test_base_forward_is_not_implemented(self):
        assert PfaffianStrategy.forward(torch.eye(2, dtype=torch.float64)) is None

    def test_base_backward_is_not_implemented(self):
        assert PfaffianStrategy.backward(None, None) is None
