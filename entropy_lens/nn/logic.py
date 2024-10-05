import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .concepts import Conceptizator


class EntropyLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, temperature: float = 0.6,
                 bias: bool = True, conceptizator: str = 'identity_bool') -> None:
        super(EntropyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.conceptizator = Conceptizator(conceptizator)
        self.alpha = None
        self.weight = nn.Parameter(torch.Tensor(n_classes, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_classes, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, torch.Tensor):
            # Find the maximum size in each dimension
            max_size = [max(tensor.shape[i] for tensor in input) for i in range(len(input[0].shape))]

            # Pad all tensors to the same size
            padded_tensors = []
            for tensor in input:
                # Calculate the padding for each dimension
                pad = []
                for i in range(len(tensor.shape) - 1, -1, -1):
                    pad.extend([0, max_size[i] - tensor.shape[i]])
                padded_tensor = F.pad(tensor, pad, "constant", 0)  # Pad with zeros
                padded_tensors.append(padded_tensor)
            
            # Stack the padded tensors into a single tensor
            input = torch.stack(padded_tensors)

        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        self.conceptizator.concepts = input
        # compute concept-awareness scores
        gamma = self.weight.norm(dim=1, p=1)
        self.alpha = torch.exp(gamma/self.temperature) / torch.sum(torch.exp(gamma/self.temperature), dim=1, keepdim=True)

        # weight the input concepts by awareness scores
        self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        self.concept_mask = self.alpha_norm > 0.5
        x = input.multiply(self.alpha_norm.unsqueeze(1))

        # compute linear map
        x = x.matmul(self.weight.permute(0, 2, 1)) + self.bias
        return x.permute(1, 0, 2)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}'.format(
            self.in_features, self.out_features, self.n_classes
        )


if __name__ == '__main__':
    data = torch.rand((10, 5))
    layer = EntropyLinear(5, 4, 2)
    out = layer(data)
    print(out.shape)
