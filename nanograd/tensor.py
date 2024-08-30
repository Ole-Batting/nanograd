import math
from nanograd.var import Var


class Tensor:
    def __init__(self, vars, shape):
        assert math.prod(shape) == len(vars)
        self.vars = vars
        self.shape = tuple(shape)
        self.reverse_shape = reversed(shape)
        self.ndims = len(shape)
        self.index_prod = [math.prod(shape[i:]) for i in range(self.ndims)]

    def __len__(self):
        self.index_prod[0]

    def zeros(shape):
        return Tensor([Var(0.0) for _ in range(math.prod(shape))], shape)

    def _untuplify_index(self, index_tuple):
        assert len(index_tuple) == len(self.shape)
        index = index_tuple[-1]
        for idx, prod in zip(index_tuple[:-1], self.index_prod[1:]):
            index += prod * idx
        return index

    def _tuplify_index(self, index):
        assert index >= 0 and index < len(self)
        index_tuple = []
        for dim in self.reverse_shape:
            index_tuple.append(index % dim)
            index //= dim
        return tuple(reversed(index_tuple))

    def __getitem__(self, index_tuple):
        return self.vars[self._untuplify_index(index_tuple)]

    def __setitem__(self, index_tuple, var):
        self.vars[self._untuplify_index(index_tuple)] = var

    def __repr__(self):
        out = f'Tensor {self.shape}\n{"[" * self.ndims}'
        for index in range(len(self.vars)):
            out += f' {self.vars[index].val:+.2e}'
            for dim, prod in enumerate(self.index_prod):
                if index % prod == (prod - 1):
                    out += f' {"]" * (self.ndims - dim)}\n'
                    out += f'{" " * dim}'
                    out += f'{"[" * (self.ndims - dim)}'
                    break
        return out[:-self.ndims]

    def _tesselate_shapes(self, other):
        assert all([s1 == s2 or 1 in [s1, s2] for s1, s2 in zip(self.shape, other.shape)])
        result_shape = tuple([max(s1, s2) for s1, s2 in zip(self.shape, other.shape)])

    def _elementwise(self, other, op):
        return Tensor([op(a, b) for a, b in zip(self.vars, other.vars)], self.shape)

    def _tesselate_op(self, other, op):
        if self.shape == other.shape:
            return self._elementwise(other, op)

        result_shape = self._tesselate_shapes(other)
        result = Tensor.zeros(result_shape)
        for index in range(len(result)):
            index_tuple = result._tuplify_index(index)
            pass
            



if __name__ == '__main__':
    a = Tensor([Var(i) for i in range(12)], (3, 2, 2))
    print(a)
    print(a[0, 0, 0], a[1, 1, 1])

