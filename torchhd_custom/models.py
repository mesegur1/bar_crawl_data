#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import copy
import math
from typing import Type, Union, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F

import torch.utils.data as data
from tqdm import tqdm
import torchmetrics
import torchhd
import torchhd.functional as functional
import torchhd.datasets as datasets
import torchhd.embeddings as embeddings


__all__ = ["Centroid", "IntRVFL", "BHDC"]


class Centroid(nn.Module):
    r"""Implements the centroid classification model using class prototypes.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of the output, typically the number of classes.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Shape:
        - Input: :math:`(*, d)` where :math:`*` means any number of
          dimensions including none and ``d = in_features``.
        - Output: :math:`(*, n)` where all but the last dimension
          are the same shape as the input and ``n = out_features``.

    Attributes:
        weight: the trainable weights, or class prototypes, of the module of shape
            :math:`(n, d)`. The values are initialized as all zeros.

    Examples::

        >>> m = Centroid(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> output.size()
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        method=None,
        device=None,
        dtype=None,
        requires_grad=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device
        super(Centroid, self).__init__()

        self.in_features = in_features  # dimensions
        self.method = method
        self.out_features = out_features
        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

        self.sim = 0
        self.sim_count = 0
        self.aux = 0

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)

        # QuantHD
        if method == "quant_iterative":
            weight_quant = torch.empty((out_features, in_features), **factory_kwargs)
            self.weight_quant = Parameter(weight_quant, requires_grad=requires_grad)

        # SparseHD
        if method == "sparse_iterative":
            weight_sparse = torch.empty((out_features, in_features), **factory_kwargs)
            self.weight_sparse = Parameter(weight_sparse, requires_grad=requires_grad)

        # DistHD
        if method == "dist_iterative":
            self.n_disthd = torch.empty((0, in_features))
            self.m_disthd = torch.empty((0, in_features))

        # CompHD
        if method == "comp":
            self.comp_weight = None
            self.position_vectors = None

        # MultiCentroidHD
        if method == "multicentroid":
            multi_weight = [torch.empty(1, in_features) for i in range(out_features)]
            self.multi_weight = [Parameter(tensor) for tensor in multi_weight]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)
        if self.method == "quant_iterative":
            init.zeros_(self.weight_quant)
        if self.method == "sparse_iterative":
            init.zeros_(self.weight_sparse)
        if self.method == "multicentroid":
            for i in self.multi_weight:
                init.zeros_(i)

    def forward(self, input: Tensor, dot: bool = False) -> Tensor:
        if dot:
            return functional.dot_similarity(input, self.weight)
        return functional.cosine_similarity(input, self.weight)

    @torch.no_grad()
    def add(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.weight.index_add_(0, target, input, alpha=lr)

    @torch.no_grad()
    def add_noise(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        self.sim_count += 1
        if torch.sum(abs(self.weight[target])) != 0:
            s = (
                torch.sum(input * self.weight[target])
                / torch.sum(abs(self.weight[target])).item()
            )
            self.sim += s
            if s < self.sim * 0.9:
                self.weight.index_add_(0, target, input, alpha=lr)
        else:
            # print(self.sim/self.sim_count, s)
            # print(s.item(), self.sim/self.sim_count, torch.sum(input*self.weight[target])/torch.sum(abs(self.weight[target])), torch.sum(input*self.weight[target]), torch.sum(abs(self.weight[target])))
            self.weight.index_add_(0, target, input, alpha=lr)

    @torch.no_grad()
    def add_adapt(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, input * lr)
        self.weight.index_add_(0, pred, -input * lr)

    @torch.no_grad()
    def add_online(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0

        self.weight.index_add_(0, target, lr * alpha1 * input)
        self.weight.index_add_(0, pred, lr * alpha2 * input)

    @torch.no_grad()
    def adjust_reset(self):
        self.similarity_sum = 0
        self.count = 0
        self.error_similarity_sum = 0
        self.error_count = 0

    @torch.no_grad()
    def add_adjustSemi(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred
        self.sim_count += 1
        if is_wrong.sum().item() == 0:
            self.sim += logit.max(1).values
            if logit.max(1).values <= self.sim / self.sim_count:
                self.weight.index_add_(0, target, input, alpha=lr)
            return False, logit.gather(1, target.unsqueeze(1))
        else:
            return True, logit.gather(1, pred.unsqueeze(1))

    @torch.no_grad()
    def add_adjust(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        logit = self(input)
        # pred = logit.argmax(1)
        predx = torch.topk(logit, 2)
        pred = torch.tensor([predx.indices[0][0]], device=self.device)

        is_wrong = target != pred

        alpha = 1 - (abs(predx[0][0][0]) - abs(predx[0][0][1]))

        self.similarity_sum += logit.max(1).values.item()
        self.count += 1
        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count
        if is_wrong.sum().item() == 0:
            if logit.max(1).values.item() < val:
                self.weight.index_add_(0, target, lr * alpha * input)
                # self.weight.index_add_(0, target, input)
            # print("RIGHT", 1 / (predx[0][0][0] - predx[0][0][1]))

            return

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]
        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1

        # print(target, predx.indices)
        # print(predx.values)
        # print(alpha1,alpha2)
        # print("WRONG", 1/(predx[0][0][0]-predx[0][0][1]))

        self.weight.index_add_(0, target, lr * alpha1 * alpha * input)
        self.weight.index_add_(0, pred, lr * alpha2 * alpha * input)

    @torch.no_grad()
    def add_adjust_iterative(
        self, input: Tensor, target: Tensor, lr: float = 1.0
    ) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        self.similarity_sum += logit.max(1).values.item()
        self.count += 1
        if self.error_count == 0:
            val = self.similarity_sum / self.count
        else:
            val = self.error_similarity_sum / self.error_count
        if is_wrong.sum().item() == 0:
            if logit.max(1).values.item() < val:
                alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
                self.weight.index_add_(0, target, lr * alpha1 * input)
                # self.weight.index_add_(0, target, input)
            return

        self.error_count += 1
        self.error_similarity_sum += logit.max(1).values.item()

        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        self.weight.index_add_(0, target, lr * alpha1 * input)
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1
        self.weight.index_add_(0, pred, lr * alpha2 * input)

    def quantized_similarity(self, input, model):
        if model == "binary":
            return functional.hamming_similarity(input, self.weight_quant).float()
        elif model == "ternary":
            return functional.dot_similarity(input, self.weight_quant)

    def add_quantize(
        self, input: Tensor, target: Tensor, lr: float = 1.0, model="binary"
    ) -> None:
        logit = self.quantized_similarity(input, model)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, lr * input)
        self.weight.index_add_(0, pred, lr * -input)

    def binarize_model(self, model, device):
        if model == "binary":
            self.weight_quant.data = torch.sign(self.weight.data)
        elif model == "ternary":
            self.weight_quant.data = torch.where(
                self.weight.data > 0,
                torch.tensor(1.0),
                torch.where(
                    self.weight.data < 0, torch.tensor(-1.0), torch.tensor(0.0)
                ).to(device),
            )

    def add_sparse(
        self, input: Tensor, target: Tensor, lr: float = 1.0, iter=0
    ) -> None:
        if iter == 0:
            logit = self(input)
        else:
            logit = self.sparse_similarity(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        if iter == 0:
            self.weight.index_add_(0, target, lr * input)
            self.weight.index_add_(0, pred, lr * -input)
        else:
            self.weight_sparse.index_add_(0, target, lr * input)
            self.weight_sparse.index_add_(0, pred, lr * -input)

    def sparse_similarity(self, input):
        return functional.dot_similarity(input, self.weight_sparse)

    def sparsify_model(self, model, s, iter):
        if model == "dimension":
            if iter == 0:
                max_vals, _ = torch.max(self.weight.data, dim=0)
                min_vals, _ = torch.min(self.weight.data, dim=0)
            else:
                max_vals, _ = torch.max(self.weight_sparse.data, dim=0)
                min_vals, _ = torch.min(self.weight_sparse.data, dim=0)
            variation = max_vals - min_vals
            _, dropped_indices = variation.topk(s, largest=False)

            if iter == 0:
                self.weight_sparse.data = self.weight.data.clone()
            self.weight_sparse.data[:, dropped_indices] = 0
        if model == "class":
            if iter == 0:
                print(s)
                _, dropped_indices = torch.topk(
                    self.weight.abs(), k=s, dim=1, largest=False, sorted=True
                )
            else:
                _, dropped_indices = torch.topk(
                    self.weight_sparse.abs(), k=s, dim=1, largest=False, sorted=True
                )
            if iter == 0:
                self.weight_sparse.data = self.weight.data.clone()
            self.weight_sparse.data[:, dropped_indices] = 0

    def add_neural(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        self.weight.index_add_(0, target, lr * input)
        self.weight.index_add_(0, pred, lr * -input)

    def neural_regenerate(self, r, encode, device):
        max_vals, _ = torch.max(self.weight.data, dim=0)
        min_vals, _ = torch.min(self.weight.data, dim=0)

        variation = max_vals - min_vals
        _, dropped_indices = variation.topk(r, largest=False)

        self.weight.data[:, dropped_indices] = (
            torch.randn(self.weight.size(0)).unsqueeze(1).to(device)
        )

        if hasattr(encode.embed, "flocet_encoding"):
            encode.embed.flocet_encoding.weight[:, dropped_indices] = (
                torch.randn(encode.embed.flocet_encoding.weight.size(0))
                .unsqueeze(1)
                .to(device)
            )
        elif hasattr(encode.embed, "density_encoding"):
            encode.embed.density_encoding.weight[:, dropped_indices] = (
                torch.randn(encode.embed.density_encoding.weight.size(0))
                .unsqueeze(1)
                .to(device)
            )
        else:
            encode.embed.weight[:, dropped_indices] = (
                torch.randn(encode.embed.weight.size(0)).unsqueeze(1).to(device)
            )

    @torch.no_grad()
    def add_dist(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = 1.0 - logit.gather(1, pred.unsqueeze(1))

        self.weight.index_add_(0, target, lr * alpha1 * input)
        self.weight.index_add_(0, pred, lr * alpha2 * -input)

    @torch.no_grad()
    def eval_dist(
        self,
        input: Tensor,
        target: Tensor,
        device,
        lr: float = 1.0,
        alpha=1.0,
        beta=1.0,
        theta=1.0,
    ) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        logit = self(input)
        _, top_2 = torch.topk(logit, k=2)
        pred1 = top_2[0][0]
        pred2 = top_2[0][1]

        if pred2 == target:
            m1 = torch.abs(input - self.weight[pred1]).to(device)
            m2 = torch.abs(input - self.weight[pred2]).to(device)
            self.m_disthd = torch.cat(
                (self.m_disthd.to(device), (alpha * m1 - beta * m2).to(device)), dim=0
            )
        if pred1 != target and pred2 != target:
            n1 = torch.abs(input - self.weight[pred2]).to(device)
            n2 = torch.abs(input - self.weight[pred1]).to(device)
            n3 = torch.abs(input - self.weight[target]).to(device)
            self.n_disthd = torch.cat(
                (
                    self.m_disthd.to(device),
                    (alpha * n1 + beta * n2 - theta * n3).to(device),
                ),
                dim=0,
            ).to(device)

    @torch.no_grad()
    def reset_n_m(self):
        self.n_disthd = torch.empty((0, self.in_features))
        self.m_disthd = torch.empty((0, self.in_features))

    def regenerate_dist(self, r, encode, device, eps=1e-12):
        norms = self.m_disthd.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.m_disthd.div_(norms)

        norms = self.n_disthd.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.n_disthd.div_(norms)

        m_ = torch.sum(self.m_disthd, dim=0)
        n_ = torch.sum(self.n_disthd, dim=0)

        _, top_m_ = m_.topk(r, largest=True)
        _, top_n_ = n_.topk(r, largest=True)

        # get the intersecting elements
        intersect = list(set(top_m_).intersection(set(top_n_)))

        # convert intersect back to a tensor
        dimensions_regenerated = torch.tensor(intersect).long().to(device)

        self.weight.data[:, dimensions_regenerated] = (
            torch.randn(self.weight.size(0)).unsqueeze(1).to(device)
        )

        if hasattr(encode.embed, "flocet_encoding"):
            encode.embed.flocet_encoding.weight[:, dimensions_regenerated] = (
                torch.randn(encode.embed.flocet_encoding.weight.size(0))
                .unsqueeze(1)
                .to(device)
            )
        elif hasattr(encode.embed, "density_encoding"):
            encode.embed.density_encoding.weight[:, dimensions_regenerated] = (
                torch.randn(encode.embed.density_encoding.weight.size(0))
                .unsqueeze(1)
                .to(device)
            )
        else:
            encode.embed.weight[:, dimensions_regenerated] = (
                torch.randn(encode.embed.weight.size(0)).unsqueeze(1).to(device)
            )

        self.reset_n_m()

    def multi_similarity(self, input, device):
        return torch.cat(
            [
                functional.dot_similarity(input, i.to(device))[0]
                for i in self.multi_weight
            ],
            dim=0,
        )

    @torch.no_grad()
    def add_multi(self, input: Tensor, target: Tensor, device, lr: float = 1.0) -> None:
        """Adds the input vectors scaled by the lr to the target prototype vectors."""
        logit = self.multi_similarity(input, device)

        pred = torch.argmax(logit, dim=0)
        row = 0
        col = 0
        for i in self.multi_weight:
            if i.shape[0] >= pred:
                col = i.shape[0] - pred
                break
            else:
                row += 1
                pred -= i.shape[0]
        if row == target:
            if self.multi_weight[row].shape[0] == col:
                col -= 1
            self.multi_weight[row][col] = self.multi_weight[row][col].to(
                device
            ) + input.to(device)[0].to(device)
        elif len(self.multi_weight[target]) < 10:
            self.multi_weight[target] = torch.cat(
                [self.multi_weight[target].to(device), input.to(device)], dim=0
            ).to(device)
        return torch.tensor([row]).to(device)

    def drop_classes(self, drop, device):
        concatenated = torch.cat([i.to(device) for i in self.multi_weight], dim=0)
        abs_sum = torch.abs(concatenated).sum(dim=1)
        sorted_indices = torch.argsort(abs_sum)[:drop]
        sorted_indices, _ = torch.sort(sorted_indices, descending=False)
        pos = 0
        for r in range(len(self.multi_weight)):
            prev_pos = pos
            pos += self.multi_weight[r].shape[0]

            remove_ind = sorted_indices[sorted_indices < pos]
            remove_ind = remove_ind - prev_pos

            indices_to_keep = [
                i for i in range(self.multi_weight[r].shape[0]) if i not in remove_ind
            ]

            if len(indices_to_keep) > 0:
                tensors_to_keep = torch.index_select(
                    self.multi_weight[r].to(device),
                    dim=0,
                    index=torch.tensor(indices_to_keep).to(device),
                )

                self.multi_weight[r] = tensors_to_keep.to(device)
            sorted_indices = sorted_indices[sorted_indices >= pos]

    def cluster_classes(self, drop, device):
        concatenated = torch.cat([i.to(device) for i in self.multi_weight], dim=0)
        abs_sum = torch.abs(concatenated).sum(dim=1).to(device)
        sorted_indices = torch.argsort(abs_sum)[:drop]
        sorted_indices, _ = torch.sort(sorted_indices, descending=False).to(device)
        pos = 0
        for r in range(len(self.multi_weight)):
            prev_pos = pos
            pos += self.multi_weight[r].shape[0]

            remove_ind = sorted_indices[sorted_indices < pos]
            remove_ind = remove_ind - prev_pos

            to_cluster = torch.index_select(
                self.multi_weight[r], dim=0, index=remove_ind
            ).to(device)
            cluster = torchhd.multiset(to_cluster).to(device)

            indices_to_keep = [
                i for i in range(self.multi_weight[r].shape[0]) if i not in remove_ind
            ]
            tensors_to_keep = torch.index_select(
                self.multi_weight[r].to(device),
                dim=0,
                index=torch.tensor(indices_to_keep).to(device),
            )

            self.multi_weight[r] = tensors_to_keep.to(device)
            most_similar = torch.argmax(
                torchhd.cosine_similarity(cluster, self.multi_weight[r]), dim=0
            )
            self.multi_weight[r] += most_similar.to(device)
            sorted_indices = sorted_indices[sorted_indices >= pos]

    def get_subclasses(self):
        sub_classes = 0
        for i in self.multi_weight:
            sub_classes += i.shape[0]
        return sub_classes

    def reduce_subclasses(
        self,
        train_loader,
        device,
        encode,
        model,
        classes,
        accuracy_full,
        reduce_subclasses="drop",
        threshold=0.03,
    ) -> None:
        for i in range(10):
            accuracy = torchmetrics.Accuracy("multiclass", num_classes=classes).to(
                device
            )

            drop_classes = int(self.get_subclasses() * 0.1)
            if reduce_subclasses == "drop":
                self.drop_classes(drop_classes, device)
            elif reduce_subclasses == "cluster":
                self.cluster_classes(drop_classes)

            with torch.no_grad():
                for samples, labels in tqdm(train_loader, desc="Reduce subclass"):
                    samples = samples.to(device)
                    labels = labels.to(device)
                    samples_hv = encode(samples)
                    outputs = model.multi_similarity(samples_hv, device)

                    pred = torch.argmax(outputs, dim=0)
                    row = 0
                    for i in model.multi_weight:
                        if i.shape[0] >= pred:
                            break
                        else:
                            row += 1
                            pred -= i.shape[0]
                    accuracy.update(torch.tensor([row]).to(device), labels)
            new_acc = accuracy.compute().item()
            if accuracy_full - new_acc > threshold:
                return

    def norm(self, w, eps):
        norms = w.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        w.div_(norms)
        return w

    @torch.no_grad()
    def normalize(self, eps=1e-12) -> None:
        """Transforms all the class prototype vectors into unit vectors.

        After calling this, inferences can be made more efficiently by specifying ``dot=True`` in the forward pass.
        Training further after calling this method is not advised.
        """
        norms = self.weight.norm(dim=1, keepdim=True)
        norms.clamp_(min=eps)
        self.weight.div_(norms)
        if self.method == "multicentroid":
            self.multi_weight = [self.norm(i, eps) for i in self.multi_weight]

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features is not None
        )

    def comp_compress(self, chunks, device):
        comp_weight = torch.empty((self.out_features, int(self.in_features / chunks)))
        self.comp_weight = Parameter(comp_weight)

        w_re = torch.reshape(
            self.weight, (self.out_features, chunks, int(self.in_features / chunks))
        ).to(device)
        self.position_vectors = torchhd.embeddings.Random(
            chunks, int(self.in_features / chunks)
        ).to(device)

        for i in range(self.out_features):
            self.comp_weight.data[i] = torch.sum(
                w_re[i] * self.position_vectors.weight, dim=0
            ).to(device)

    def compress_hv(self, enc, chunks, device):
        return torch.sum(
            torch.reshape(enc, (chunks, int(self.in_features / chunks)))
            * self.position_vectors.weight,
            dim=0,
        ).to(device)

    def forward_comp(self, enc, device):
        return functional.dot_similarity(
            enc.to(device), self.comp_weight.to(device)
        ).to(device)


class IntRVFL(nn.Module):
    r"""Class implementing integer random vector functional link network (intRVFL) model as described in `Density Encoding Enables Resource-Efficient Randomly Connected Neural Networks <https://doi.org/10.1109/TNNLS.2020.3015971>`_.

    Args:
        in_features (int): Size of each input sample.
        dimensions (int): The number of hidden dimensions to use.
        out_features (int): The number of output features, typically the number of classes.
        kappa (int, optional): Parameter of the clipping function limiting the range of values; used as the part of transforming input data.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: ``False``.

    Shape:
        - Input: :math:`(*, d)` where :math:`*` means any number of
          dimensions including none and ``d = in_features``.
        - Output: :math:`(*, n)` where all but the last dimension
          are the same shape as the input and ``n = out_features``.

    Attributes:
        weight: the trainable weights, or class prototypes, of the module of shape
            :math:`(n, d)`. The values are initialized as all zeros.

    """

    __constants__ = ["in_features", "dimensions", "out_features", "kappa"]
    in_features: int
    dimensions: int
    out_features: int
    kappa: Optional[int]
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        dimensions: int,
        out_features: int,
        kappa: Optional[int] = None,
        device=None,
        dtype=None,
        requires_grad=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(IntRVFL, self).__init__()

        self.in_features = in_features
        self.dimensions = dimensions
        self.out_features = out_features
        self.kappa = kappa

        self.encoding = embeddings.Density(
            in_features, self.dimensions, **factory_kwargs
        )

        weight = torch.empty((out_features, dimensions), **factory_kwargs)
        self.weight = Parameter(weight, requires_grad=requires_grad)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def encode(self, x):
        encodings = self.encoding(x)

        if self.kappa is not None:
            encodings = encodings.clipping(self.kappa)

        return encodings

    def forward(self, x, dimensions=10000, f=None, device=None):
        # Make encodings for all data samples in the batch
        encodings = self.encode(x)
        if f != None:
            num_dim = int((f / 100) * dimensions)
            f_mask = torch.randperm(dimensions - 0)[:num_dim]
            encodings[0][f_mask] = encodings[0][f_mask] * -torch.ones(num_dim).to(
                device
            )
        # Get similarity values for each class
        return functional.dot_similarity(encodings, self.weight)

    def dot_similarity(self, x):
        return functional.dot_similarity(x, self.weight)

    # Train the model
    @torch.no_grad()
    def fit_ridge_regression(
        self,
        samples: Tensor,
        labels: Tensor,
        alpha: Optional[float] = 1,
    ) -> None:
        r"""Compute the weights (readout matrix) with :func:`~torchhd.ridge_regression`.

        It is a common way to form classifiers wihtin randomized neural networks see, e.g., `Randomness in Neural Networks: An Overview  <https://doi.org/10.1002/widm.1200>`_.

        Args:
            samples (Tensor): The feature vectors.
            labels (LongTensor): The targets vector, typically the class of each sample.
            alpha (float, optional): Scalar for the variance of the samples. Default is 1.

        Shapes:
           - Samples: :math:`(n, f)`
           - Labels: :math:`(n, c)`

        """
        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}
        n = labels.size(0)

        # Transform to hypervector representations
        encodings = self.encode(samples)

        # Transform classes to one-hot encoding
        one_hot_labels = torch.zeros(n, self.out_features, **factory_kwargs)
        one_hot_labels[torch.arange(n), labels] = 1

        # Compute the readout matrix using the ridge regression
        weights = functional.ridge_regression(encodings, one_hot_labels, alpha=alpha)
        # Assign the obtained classifier to the output
        self.weight.copy_(weights)


class BinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3):
        super(BinaryLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            (torch.rand((out_features, in_features)) * 2 - 1) * 0.001,
            requires_grad=True,
        )

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights))
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # binary_weights_no_grad = torch.sign(real_weights)

        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = (
            binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        )
        y = F.linear(x, binary_weights)
        return y


class BHDC(nn.Module):
    def __init__(self, in_features=10000, out_features=10, dropout_prob=0):
        super(BHDC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob

        self.bn_weight = BinaryLinear(in_features, out_features)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

        weight = torch.empty((out_features, in_features))
        self.weight = Parameter(weight)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def forward(self, x, dot=False):
        x = x.view(-1, self.in_features)
        x = self.dropout(x)
        x = self.bn_weight(x)
        return x

    def binarize(self, x):
        x[x >= 0] = 1
        x[x < 0] = -1
        return x

    def norm_weights(self):
        self.weight = nn.Parameter(torch.sign(self.bn_weight.weight))
