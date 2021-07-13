import numpy as np
import torch
import torch.nn.functional as F
from itertools import chain
from edien.crf import CRF
from edien.preprocess import VariableSequencePacker
import pytorch_transformers as old_transformers
import transformers


class PaddedBatchNorm(torch.nn.Module):
    """ BatchNorm implementation for NLP - where the effective batch size
    often changes a lot due to padding. There are also a few additional
    changes - for instance making sure gamma is positive:
        https://arxiv.org/pdf/1705.07057.pdf
    """

    def __init__(self, shape):
        super(PaddedBatchNorm, self).__init__()
        self.shape = shape

        # Params
        self.log_gamma = torch.nn.Parameter(torch.zeros(self.shape))

        self.beta = torch.nn.Parameter(torch.zeros(self.shape))

        self.eps = 1e-5
        self.decay = 0.1
        self.register_buffer('running_mean', torch.zeros(self.shape))
        self.register_buffer('running_var', torch.ones(self.shape))

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        init.zeros_(self.log_gamma)
        init.zeros_(self.beta)

    def forward(self, h, mask):
        assert len(h.shape) == 4 or len(h.shape) == 2

        mask = mask.to(h.device).float()

        if len(h.shape) == 4:
            sum_dim = (0, 2, 3)
        else:
            mask = mask.view(-1, 1)
            sum_dim = (0,)

        # TODO: Might want to do moving avg
        if self.training:
            # We don't use mean since some inputs are "invalid/padded out"
            sum_act = torch.sum(h * mask, dim=sum_dim, keepdim=True)
            # Count of padded out (these are the same number across idx 1)
            actual_active = torch.sum(mask, dim=sum_dim, keepdim=True)

            # Running average
            running_mean = sum_act / actual_active

            sum_var = torch.sum(((h - running_mean) ** 2) * mask, dim=sum_dim, keepdim=True)
            # Unbiased estimate
            running_var = sum_var / (actual_active - 1)
            running_std = torch.sqrt(running_var + self.eps)

            # Make sure we don't attempt to backprop through time
            # by using detach
            self.running_mean = self.running_mean + (running_mean.detach() - self.running_mean) * self.decay
            self.running_var = self.running_var + (running_var.detach() - self.running_var) * self.decay
        else:
            running_mean = self.running_mean
            running_std = torch.sqrt(self.running_var + self.eps)
        # As done in chainer's original implementation

        # Normalize activation
        h = h - running_mean
        h = h / running_std

        # CHECKS
        # These should give variance ~ 1 and mean ~ 0
        # if len(h.shape) == 4:
        #     m_act = torch.sum(h * mask, dim=sum_dim, keepdim=True).detach()
        #     v_act = torch.sum(((h - running_mean) ** 2) * mask, dim=sum_dim, keepdim=True).detach()
        #     print('\tcnn mean', (m_act/actual_active).view(-1))
        #     print('\tcnn variance', (v_act/actual_active).view(-1))
        # if len(h.shape) == 2:
        #     print('\tmean', (h * mask).sum(dim=0) / actual_active)
        #     print('\tvariance', (((h - running_mean) **2) * mask).sum(dim=0) / actual_active)

        # Make all activations possible again
        h = h * torch.exp(self.log_gamma) + self.beta
        # Mask out invalid output
        h = h * mask

        return h


class PaddedEmbedding(torch.nn.Module):

    PADDING_IDX = -1

    def __init__(self, in_size, embed_dim, dropout=None):
        super(PaddedEmbedding, self).__init__()
        self.in_size = in_size
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.padding_idx = self.in_size
        self.embedding = torch.nn.Embedding(self.in_size + 1,
                                            self.embed_dim)

    def forward(self, X):
        # Avoid modifying X in place
        # Pytorch wants padding idx to be in range + not negative
        mask = (X != self.PADDING_IDX)
        Xcopy = X.clone()
        Xcopy[X == self.PADDING_IDX] = self.padding_idx
        if self.dropout is not None and self.training:
            # We dropout over the index
            dropout_token_mask = torch.rand(self.in_size + 1, device=X.device) < self.dropout
            # Do not dropout standard tokens
            dropout_token_mask[0] = False
            dropout_token_mask[1] = False
            dropout_token_mask[2] = False
            dropout_token_mask[3] = False
            dropout_token_mask[4] = False
            # Set indices to UNK
            Xcopy[dropout_token_mask[Xcopy]] = 0
        embeds = self.embedding(Xcopy)
        embeds = embeds * mask.float().unsqueeze(2)
        return embeds


class CNNWordEncoder(torch.nn.Module):

    MAX_WIDTH = 30
    FILTER_MULTIPLIER = 25
    # FILTER_MULTIPLIER = 1
    PADDING_IDX = -1

    def __init__(self,
                 in_size,
                 embed_dim=15,
                 ngrams=(1, 2, 3, 4, 5, 6),
                 stride=1,
                 num_filters=None,
                 project_dim=128,
                 positional_embed=False,
                 batch_norm=True,
                 char_dropout=None,
                 cnn_dropout=None,
                 project_dropout=None,
                 device='cpu'):

        super(CNNWordEncoder, self).__init__()
        if num_filters is None:
            # http://www.people.fas.harvard.edu/~yoonkim/data/char-nlm.pdf
            # Table 2 small model uses constant size
            num_filters = [n * self.FILTER_MULTIPLIER for n in ngrams]
        assert(len(num_filters) == len(ngrams))

        self.in_size = in_size
        self.embed_dim = embed_dim
        self.ngrams = ngrams
        self.stride = stride
        self.num_filters = num_filters
        self.project_dim = project_dim
        self.positional_embed = positional_embed
        # NOTE: batch_norm is for both conv net and projection layer
        self.batch_norm = batch_norm
        self.char_dropout = char_dropout
        self.cnn_dropout = cnn_dropout
        self.project_dropout = project_dropout
        self.device = device

        self.max_width = self.MAX_WIDTH
        self.total_filters = sum(num_filters)
        self.out_size = self.project_dim

        # Params
        # Pytorch doesn't let PADDING_IDX be arbitrary - eg. a negative number.
        self.embed_layer = PaddedEmbedding(self.in_size, self.embed_dim, dropout=self.char_dropout)
        if self.positional_embed:
            self.positional_embed_layer = PaddedEmbedding(self.max_width,
                                                          self.embed_dim)

        self.cnn_blocks = ['cnn_%d' % n for n in self.ngrams]
        # Since we are padding we can't use default BatchNorm
        self.bn_blocks = ['bn_%d' % n for n in self.ngrams]

        for i, (cnn_name, bn_name) in enumerate(zip(self.cnn_blocks, self.bn_blocks)):
            setattr(self,
                    cnn_name,
                    torch.nn.Conv2d(1,
                                    self.num_filters[i],
                                    (self.ngrams[i], self.embed_dim),
                                    self.stride,
                                    # If we use batch norm
                                    # adding a bias is extraneous
                                    bias=not self.batch_norm))
            if batch_norm:
                setattr(self,
                        bn_name,
                        PaddedBatchNorm(shape=(1, self.num_filters[i], 1, 1)))

        # If we are using batch_norm then we don't need a bias
        self.project = torch.nn.Linear(self.total_filters, self.project_dim, bias=not self.batch_norm)

        if self.batch_norm:
            self.batch_norm_out = torch.nn.BatchNorm1d(self.project_dim)

        if self.cnn_dropout is not None:
            self.cnn_dropout_layer = torch.nn.Dropout(p=self.cnn_dropout)

        if self.project_dropout is not None:
            self.proj_dropout_layer = torch.nn.Dropout(p=self.project_dropout)

        self.cache = dict()

    def forward(self, batch):
        # We need to carry out below on cpu as cupy does not support dtype='O'
        # Below is default case were input is of variable length
        if batch.dtype == 'O':
            all_words = batch.reshape(-1)
        else:
            all_words = batch.squeeze()
            if len(all_words.shape) == 1:
                all_words = [tuple(all_words)]
            else:
                all_words = [tuple(w) for w in all_words]
        word_set = set(w for w in all_words)

        self.encode_words(word_set)
        # At this point we collapse to equal length therefore create array.
        word_ids = torch.tensor([[self.cache[tuple(word)] for word in sent]
                                 for sent in batch],
                                 device=self.device)
        # word_ids = torch.tensor([[self.cache[tuple(word)] for word in sent]
        #                          for sent in batch])
        # Replace padding with zeros please.
        padding_id = self.cache.get((self.PADDING_IDX,), None)
        # Make padding be 0.
        if padding_id is not None:
            self.embedding.data[padding_id].fill_(0.)
        act = F.embedding(word_ids,
                          self.embedding)
        self.clear_cache()

        return act

    def encode_words(self, word_list):

        batch_size = len(word_list)
        width = self.max_width
        word_vars = torch.tensor([[w[i]
                                   if i < len(w)
                                   else self.PADDING_IDX
                                   for i in range(width)]
                                  for w in word_list],
                                 dtype=torch.long,
                                 device=self.device)

        embeddings = self.embed_layer(word_vars)

        if self.positional_embed:
            word_idxs = torch.arange(self.max_width,
                                     device=self.device)
            word_idxs = word_idxs.repeat(batch_size)
            word_idxs[word_vars == self.PADDING_IDX] = self.PADDING_IDX
            pos_embeddings = self.positional_embed_layer(word_idxs)
            # TODO: Need to PAD
            embeddings += pos_embeddings

        # Ignore embedding dim
        stacked = embeddings.view(batch_size, -1, self.embed_dim)

        # Store what part of the input is padding for all words
        mask = (word_vars.detach() != self.PADDING_IDX)
        mask = mask.view(batch_size, -1)

        # Need to expand dims for "in channel"
        mask = mask.unsqueeze(1)

        stacked = stacked.unsqueeze(1)

        # for each in batch_embeddings:
        acts = []
        for cnn_block, bn_block in zip(self.cnn_blocks, self.bn_blocks):
            h = getattr(self, cnn_block)(stacked)

            # Repeat conv mask across filter dimension
            conv_mask = mask.expand(-1, h.shape[1], -1)
            # Make conv_mask same dim as h
            conv_mask = conv_mask.unsqueeze(3)
            # h.shape[2] is MAX_WIDTH - len(ngram) - 1
            # We limit mask according to output size of convnet
            conv_mask = conv_mask[:, :, :h.shape[2], :]

            zeros = torch.zeros_like(h)

            h = torch.where(conv_mask, h, zeros)

            if self.batch_norm:
                h = getattr(self, bn_block)(h, conv_mask)

            h = F.relu(h)

            if self.cnn_dropout:
                h = self.cnn_dropout_layer(h)
            # NOTE: batch_size is num_words in batch
            # h shape : batch_size x num_filters x sent_len x 1
            # =============================================================
            h, _ = torch.max(h, dim=2)
            h = h.squeeze(dim=2)
            # h shape : batch_size x num_filters
            acts.append(h)

        act = torch.cat(acts, dim=1)

        # Project down
        act = self.project(act)

        if self.batch_norm:
            act = self.batch_norm_out(act)

        act = F.relu(act)

        if self.project_dropout:
            act = self.proj_dropout_layer(act)

        self.embedding = act

        for i, word in enumerate(word_list):
            self.cache[tuple(word)] = i

    def word_to_index(self, word):
        return self.cache[tuple(word)]

    def clear_cache(self):
        self.cache = dict()


class Embedder(torch.nn.Module):
    """A general embedder that concatenates the embeddings of an arbitrary
    number of input sequences."""

    def __init__(self, inputs, composition='concat', dropout=None):
        """
        :inputs: dict
        :dropout: float between 0 and 1, how much dropout to apply to the input
        As an example, suppose we want to encode words and part of speech
        tags. If we want: word vocab -> 100 , word emb size -> 100,
        pos vocab -> 10, pos emb size -> 30, we would feed:
        in_sizes = (100, 10) and out_sizes = (100, 30)
        the forward method then assumes that you will feed in the
        sequences in the corresponding order - first word indices and then
        pos tag indices.
        """
        super(Embedder, self).__init__()
        assert composition in ['concat']

        self.inputs = inputs
        self.composition = composition
        self.dropout = dropout

        out_sizes = []
        for embed_name, embedder in sorted(inputs.items(), key=lambda x: x[0]):
            self.set_embed(embed_name, embedder)
            if hasattr(embedder, 'out_size'):
                out_sizes.append(embedder.out_size)
            else:
                out_sizes.append(tuple(embedder.parameters())[0].shape[1])

        if self.composition == 'concat':
            self.out_size = sum(out_sizes)
        else:
            out_size = out_sizes[0]
            assert all(each == out_size for each in out_sizes)
            self.out_size = out_size

    def get_embed(self, index):
        return getattr(self, 'embed_%s' % index)

    def set_embed(self, index, embed):
        setattr(self, 'embed_%s' % index, embed)

    def forward(self, seqs_nt):
        # NOTE: seqs_nt is a named tuple
        if self.composition == 'concat':
            act = torch.cat([self.get_embed(k)(s)
                             for k, s in zip(seqs_nt._fields, seqs_nt)],
                            dim=2)
            if self.dropout is not None and self.training:
                dropout_mask = torch.rand(act.shape[:-1], device=act.device) > self.dropout
                act = act * dropout_mask.unsqueeze(2).float()
        else:
            raise AttributeError('Unknown composition %s' % self.composition)

        return act


class BiLSTMSentenceEncoder(torch.nn.Module):

    def __init__(self, encoder, num_layers, hidden_dim, dropout=None):
        super(BiLSTMSentenceEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout if dropout is not None else 0.

        self.encoder = encoder
        self.lstm = torch.nn.LSTM(self.hidden_dim,
                                  self.hidden_dim,
                                  self.num_layers,
                                  dropout=self.dropout,
                                  batch_first=True,
                                  bidirectional=True)

        self.inputs = tuple(sorted(self.encoder.inputs.keys()))

    def forward(self, X, sent_lens):
        X = self.encoder(X)

        bs, ss, hid = X.shape

        sent_lens = torch.tensor(sent_lens, device=X.device)
        sorted_lens, arg_sort = torch.sort(sent_lens, descending=True)
        X = X[arg_sort]

        acts = torch.nn.utils.rnn.pack_padded_sequence(X, sorted_lens, batch_first=True)

        hs, (_, _) = self.lstm(acts)

        undo_sort = torch.argsort(arg_sort)

        act, sl = torch.nn.utils.rnn.pad_packed_sequence(hs,
                                                         batch_first=True,
                                                         padding_value=0.,
                                                         total_length=ss)

        act = act[undo_sort]

        return act


class BertSentenceEncoder(torch.nn.Module):

    def __init__(self, bert_file, inputs, legacy=True):
        super(BertSentenceEncoder, self).__init__()

        self.bert_file = bert_file
        # Sorry folks, this library changed a lot in a year!
        # The KNN experiments have been updated to a more recent version
        # of transformers but the models were trained using pytorch_transformers
        if legacy:
            self.config = old_transformers.BertConfig()
            self.encoder = old_transformers.BertModel(self.config)
            # NOTE: If you want to train a model starting with a pretrained
            # encoder, uncomment the line below:
            # self.encoder = old_transformers.AutoModel.from_pretrained(self.bert_file)
        else:
            self.encoder = transformers.AutoModel.from_pretrained(self.bert_file)
        self.inputs = inputs

    def forward(self, X, sent_lens):
        # NOTE: sent_lens is output tokens
        word_mask = X.word_mask
        tokens = X.tokens

        max_sent_len = max(sent_lens)

        # Padding in Bert uses the zero index
        attention_mask = tokens != -1
        word_mask[word_mask == -1] = 0
        tokens[tokens == -1] = 0

        # Use sequence of hidden-states at the output of the last layer 
        act = self.encoder(input_ids=tokens, attention_mask=attention_mask)[0]

        # Drop CLS activation
        act = act[:, 1:]

        # batch_size x word_piece_tokens x embed_dim
        bs, ss_plus_sep, emb_dim = act.shape

        # Bert activations also include subtoken activations - which we don't
        # want to use when doing token level prediction
        # we therefore follow the BERT paper and only use the activation from
        # the first subtoken when wordpiece split our token

        # This is a bit complicated.. bear with me
        # In word_mask we have stored a boolean mask with ones in indices
        # where we fed in tokens or the first subtoken of a wordpiece split

        #    word_mask
        # 1 1 0 1 0 0 1 0 0

        #      act           # imagine each number is an embedding (column)
        # 1 2 3 4 5 6 7 8 9

        #   reordered act    # we want to move the subword tokens to the right
        # 1 2 4 7 3 5 6 8 9

        decrease_word_mask = word_mask * torch.arange(word_mask.shape[1], 0, -1, device=word_mask.device).view(1, -1)
        sort_indices = torch.argsort(decrease_word_mask, dim=1, descending=True)
        # sort_indices will place the embeddings in consecutive order at the
        # beginning of the sequence.
        sort_indices = sort_indices.unsqueeze(2).expand(-1, -1, emb_dim)
        act = torch.gather(act, 1, sort_indices)

        # print(act.shape)
        mask_lens = torch.sum(word_mask, 1)
        assert torch.sum(mask_lens == torch.tensor(sent_lens, device=mask_lens.device)) == bs
        # Truncate anything longer than sentence length
        act = act[:, :max_sent_len].contiguous()

        return act


class BottleneckMLP(torch.nn.Module):
    """A single hidden layer with Relu activation"""

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 dropout=None,
                 batch_norm=True):
        super(BottleneckMLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.batch_norm = batch_norm

        if self.hidden_dim:
            self.W_hidden = torch.nn.Linear(self.in_dim,
                                            self.hidden_dim,
                                            bias=not self.batch_norm)
            self.W_hidden.weight.data.normal_(mean=0.0, std=0.02)

            self.W_out = torch.nn.Linear(self.hidden_dim, self.out_dim)
            self.W_out.weight.data.normal_(mean=0.0, std=0.02)
            self.W_out.bias.data.zero_()
            if self.batch_norm:
                self.bn = PaddedBatchNorm((1, hidden_dim))
            if self.dropout:
                self.hid_dropout_layer = torch.nn.Dropout(p=dropout)
        else:
            self.W_out = torch.nn.Linear(self.in_dim, self.out_dim)
            self.W_out.weight.data.normal_(mean=0.0, std=0.02)
            self.W_out.bias.data.zero_()

        if self.dropout:
            self.inp_dropout_layer = torch.nn.Dropout(p=dropout)

    def _compute_activation(self, X, sentence_lengths):

        act = X

        # batch size, sequence size, embed_dim
        bs, ss, emb_dim = act.shape

        if self.dropout:
            act = self.inp_dropout_layer(act)

        # Collapse down to 2D to do follow up mat muls
        act = act.view(-1, emb_dim)

        if self.hidden_dim:
            act = self.W_hidden(act)

            if self.batch_norm:

                packer = VariableSequencePacker(sentence_lengths)
                act = self.bn(act, packer.mask)

            act = F.relu(act)

            if self.dropout:
                act = self.hid_dropout_layer(act)

        # Up to this point padding positions are zero
        # but below a bias is added. We don't care
        # since we will only mask it out for the loss
        # and predictions anyway
        act = self.W_out(act)

        # Put red and green lens glasses back on (Restore to 3D)
        act = act.view(bs, ss, -1)

        return act

    def forward(self, X, sentence_lengths):

        # bs, ss, hid = X.shape
        out = self._compute_activation(X, sentence_lengths)

        return out


class TaskMLP(torch.nn.Module):
    """Single hidden layer MLP"""

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 num_labels,
                 pipe_inputs=None,
                 dropout=None,
                 batch_norm=False,
                 use_crf=False
                 ):
        super(TaskMLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.pipe_inputs = pipe_inputs or []
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_crf = use_crf

        self.mlp = BottleneckMLP(self.in_dim,
                                 self.hidden_dim,
                                 self.num_labels,
                                 dropout=self.dropout,
                                 batch_norm=self.batch_norm)

        if len(self.pipe_inputs) > 0:
            # Create a surrogate embedding with same size as encoder embedding
            self.surrogate_embedding = torch.nn.Parameter(torch.zeros(self.in_dim))
            self.surrogate_embedding.data.normal_(mean=0.0, std=0.02)

        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)

        self.loss = 0.

    def zero_loss(self):
        self.loss = 0.

    def get_piped_preds_name(self, task):
        return '%s_preds' % task

    def forward(self, X, y,
                sentence_lengths,
                batch_data,
                predict_proba=False):

        bs, ss, hid = X.shape

        if self.training:
            for inp_name in self.pipe_inputs:
                surrogate = self.surrogate_embedding.expand_as(X)

                inp = batch_data[inp_name].unsqueeze(-1).expand_as(X)

                X = X + torch.where(inp > 0, surrogate, torch.zeros_like(X))
        else:
            for inp_name in self.pipe_inputs:
                surrogate = self.surrogate_embedding.expand_as(X)

                pred_name = self.get_piped_preds_name(inp_name)
                pred = batch_data[pred_name].unsqueeze(-1).expand_as(X)

                X = X + torch.where(pred > 0, surrogate, torch.zeros_like(X))

        act = self.mlp(X, sentence_lengths)

        # CRF
        if self.use_crf:

            packer = VariableSequencePacker(sentence_lengths)
            mask = packer.mask.to(act.device)
            if y is not None:
                # We normalise the loss by the number of valid labels. Use mask
                # to calculate this since some labels are actually padding.
                # num_predictions = torch.sum(mask)
                loss = - self.crf(act, y, mask=mask, reduction='token_mean')
                self.loss += loss

            if not predict_proba:
                preds = self.crf.decode(act, mask=mask)
                preds = torch.tensor(tuple(chain(*preds)), device=act.device)
                preds = packer.decode(preds, fill_value=-1)
            else:
                raise NotImplemented('Predict proba for CRF not implemented')

        else:
            act = act.view(-1, self.num_labels)

            if y is not None:
                y = y.view(-1)

                # Multiclass
                if self.num_labels > 1:
                    acts = act[y != -1]
                    ys = y[y != -1]
                    loss = F.cross_entropy(acts, ys, reduction='mean')
                else:
                    acts = act[y != -1]
                    ys = y[y != -1].view(-1, 1)
                    loss = F.binary_cross_entropy_with_logits(acts, ys.type_as(acts), reduction='mean')

                self.loss += loss

            act = act.view(bs, ss, self.num_labels).detach()

            if not predict_proba:
                # Multiclass
                if self.num_labels > 1:
                    preds = torch.argmax(act, dim=2)
                else:
                    preds = (act > 0.).squeeze(2).long()
            else:
                raise ValueError('Need to change feeding')
                if self.num_labels > 1:
                    preds = F.softmax(act, dim=2)
                else:
                    preds = torch.sigmoid(act)

        return preds
