import logging
import torch
import numpy as np
from edien.vocab import Vocab


def is_variablisable(variable):
    if hasattr(variable, '__len__'):
        # Below is a hack to determine if the input was variable size
        # In the case of variable size input, the np array contains objects
        # and pytorch can't deal with such arrays, so we don't make them
        # variables
        if variable.dtype == 'O' or len(variable.shape) >= 3:
            return False
    return True


class PaddedVariabliser(object):

    """Implements Sequence padding and pytorch variable encoding"""

    def __init__(self, device='cpu'):
        super(PaddedVariabliser, self).__init__()
        self.max_lengths = None
        self.sequence_lengths = None
        self.device = device

    def _compute_seq_lengths(self, batch):
        """Compute sequence lengths and maximum sequence lengths for
        all input fields."""
        self.sequence_lengths = dict()
        self.max_lengths = dict()
        for field, feature in zip(batch._fields, batch):
            seq_lens = [len(s) if hasattr(s, '__len__') else 1
                        for s in feature]
            self.sequence_lengths[field] = seq_lens
            max_len = max(seq_lens)
            assert max_len != 0
            assert max_len < 250
            self.max_lengths[field] = max_len

    def encode(self, batch):
        # Set sequence lengths..
        # We are assuming that batch contains entries of size
        # bs x ss x ..  more stuff
        # bs x ss is needed though - and ss is variable size
        # We keep sequence lengths for all labels since
        # sometimes the input sequence lengths are different from the
        # output sequence lengths (eg. BERT with subword input)
        self._compute_seq_lengths(batch)
        # Keep current batch information - to use on decoding
        # NOTE: We aren't currently limiting max len
        batch = self._pad(batch)
        batch = self._variablise(batch)
        return batch

    def decode(self, batch):
        assert self.max_lengths is not None
        assert self.sequence_lengths is not None
        batch = self._unvariablise(batch)
        batch = self._unpad(batch)
        self.sequence_lengths = None
        self.max_lengths = None
        return batch

    def _pad(self, X):
        padded = []
        for field, feature in zip(X._fields, X):
            rows = []
            for row in feature:
                # If this is an iterable, attempt padding it
                if hasattr(row, '__len__'):
                    diff = self.max_lengths[field] - len(row)
                    if diff > 0:
                        # Hack for subword model
                        if isinstance(row[0], tuple):
                            row = row + (((Vocab.PAD_IDX,),) * diff)
                        else:
                            row = row + ((Vocab.PAD_IDX,) * diff)
                # elif diff < 0:
                #     print('Warning- truncated input')
                #     row = row[:self.max_lengths[field]]
                rows.append(row)
            padded.append(np.array(rows))
        return X.__class__(*padded)

    def _unpad(self, X):
        unpadded = []
        for field, feature in zip(X._fields, X):
            rows = [row[:sent_len]
                    for row, sent_len
                    in zip(feature, self.sequence_lengths[field])]
            unpadded.append(rows)
        return X.__class__(*unpadded)

    def _variablise(self, X):
        # Move contiguous/constant-size memory arrays directly to gpu
        # If not contiguous, keep on cpu since we are probably going
        # to compute something beforehand.
        X = X.__class__(*(torch.autograd.Variable(torch.tensor(each, device=self.device))
                          if is_variablisable(each)
                          else each
                          for each in X
                          ))
        return X

    def _unvariablise(self, X):
        # Move contiguous/constant-size memory arrays directly to gpu
        # If not contiguous, keep on cpu since we are probably going
        # to compute something beforehand.
        X = X.__class__(*(each
                          if isinstance(each, list)
                          else each.cpu().numpy()
                          for each in X))
        return X


class VariableSequencePacker(object):
    """Class that encapsulates stats and logic needed to transform between
    padded constant sequence length representations and collapsed/packed
    variable sequence length representations."""
    def __init__(self, sequence_lengths):
        super(VariableSequencePacker, self).__init__()
        self.sequence_lengths = sequence_lengths
        self.max_seq_length = max(sequence_lengths)
        self.batch_size = len(sequence_lengths)

    @property
    def mask(self):
        mask = torch.arange(self.max_seq_length)
        mask = mask.view(1, -1).expand(self.batch_size, -1)
        mask = mask < torch.tensor(self.sequence_lengths).view(-1, 1)
        return mask

    def encode(self, X):
        """"""
        batch_size, max_seq_len, *rest_dims = X.shape
        assert max_seq_len >= self.max_seq_length

        if max_seq_len > self.max_seq_length:
            logging.warning('Sequence length dimension is %d, but largest '
                            'sequence length passed in constructor was %d.'
                            ' Truncating to max seq length %d.' %
                            (max_seq_len,
                             self.max_seq_length,
                             self.max_seq_length))
            X = X[:, :self.max_seq_length]
        collapsed = X[self.mask]
        return collapsed

    def decode(self, X, fill_value=0.):
        """"""
        num_values, *rest_dims = X.shape

        sent_lens = torch.tensor(self.sequence_lengths)
        index_1 = torch.repeat_interleave(torch.arange(self.batch_size),
                                          sent_lens)

        offsets = torch.cumsum(sent_lens, dim=0).roll(1)
        offsets[0] = 0
        index_2 = torch.arange(num_values) - torch.repeat_interleave(offsets, sent_lens)

        dims = (self.batch_size, self.max_seq_length, *rest_dims)
        uncollapsed = torch.full(dims, fill_value, device=X.device, dtype=X.dtype)
        uncollapsed[index_1, index_2] = X
        return uncollapsed
