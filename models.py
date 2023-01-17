import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from sacremoses import MosesTokenizer
import sentencepiece as spm
import pairing
import paraphrastic_dataset
import utils
from torch.nn.modules.distance import CosineSimilarity
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from evaluate_sts import evaluate_sts
from torch import optim

def load_model(args=None, data=None, gpu=0, model_name=None,
        sp_model=None, megabatch_anneal=None):

    if args:
        gpu = args.gpu
        if 'sp_model' in args:
            sp_model = args.sp_model
        if 'megabatch_anneal' in args:
            megabatch_anneal= args.megabatch_anneal
        model_name = args.load_file
        
    if not gpu:
        model = torch.load(model_name, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_name)

    state_dict = model['state_dict']
    model_args = model['args']
    vocab = model['vocab']
    optimizer = model['optimizer']
    epoch = model['epoch'] + 1

    if sp_model is not None:
        model_args.sp_model = sp_model
    if megabatch_anneal is not None:
        model_args.megabatch_anneal = megabatch_anneal
    model_args.gpu = gpu

    if model_args.model == "avg":
        model = Averaging(data, model_args, vocab, vocab_fr)
    elif args.model == "lstm":
        model = LSTM(data, model_args, vocab, vocab_fr)

    model.load_state_dict(state_dict)
    model.optimizer.load_state_dict(optimizer)

    return model, epoch

class ParaModel(nn.Module):
    def __init__(self, args, vocab):
        super(ParaModel, self).__init__()

        self.args = args
        self.gpu = args.gpu
        self.epochs = args.epochs
        self.debug = args.debug
        self.save_interval = args.save_interval
        self.save_every_epoch = args.save_every_epoch
        self.save_final = args.save_final
        if "report_interval" in args:
            self.report_interval = args.report_interval
        else:
            self.report_interval = args.save_interval

        self.vocab = vocab
        self.rev_vocab = {v:k for k,v in vocab.items()}
        self.lower_case = args.lower_case

        self.delta = args.delta
        self.pool = args.pool
        self.dim = args.dim
        self.lr = args.lr
        self.grad_clip = args.grad_clip

        self.dropout = args.dropout
        self.share_encoder = args.share_encoder
        self.scramble_rate = args.scramble_rate

        self.batchsize = args.batchsize
        self.max_megabatch_size = args.megabatch_size
        self.curr_megabatch_size = 1
        self.megabatch = []
        self.megabatch_anneal = args.megabatch_anneal
        self.increment = False

        self.sim_loss = nn.MarginRankingLoss(margin=self.delta)
        self.cosine = CosineSimilarity()

        self.embedding = nn.Embedding(len(self.vocab), self.dim)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.sp_model)

    def save_params(self, epoch, counter=None):
        if counter is None:
            torch.save({'state_dict': self.state_dict(),
                    'vocab': self.vocab,
                    'args': self.args,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch}, "{0}_{1}.pt".format(self.args.outfile, epoch))
        else:
            torch.save({'state_dict': self.state_dict(),
                    'vocab': self.vocab,
                    'args': self.args,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch, 'counter': counter}, "{0}_{1}_{2}.pt".format(self.args.outfile, epoch, counter))

    def loss_function(self, g1, g2, p1, p2):
        g1g2 = self.cosine(g1, g2)
        g1p1 = self.cosine(g1, p1)
        g2p2 = self.cosine(g2, p2)

        ones = torch.ones(g1g2.size()[0])
        if self.gpu:
            ones = ones.cuda()

        loss = self.sim_loss(g1g2, g1p1, ones) + self.sim_loss(g1g2, g2p2, ones)

        return loss

    def scoring_function(self, g_idxs1, g_lengths1, g_idxs2, g_lengths2, fr0=0, fr1=0):
        g1 = self.encode(g_idxs1, g_lengths1, fr=fr0)
        g2 = self.encode(g_idxs2, g_lengths2, fr=fr1)
        return self.cosine(g1, g2)

    def train_epochs(self, data_file, start_epoch=1):
        start_time = time.time()
        self.megabatch = []
        self.ep_loss = 0
        self.curr_idx = 0

        self.eval()
        evaluate_sts(self)
        self.train()

        dataset = paraphrastic_dataset.ParaphrasticDataset(data_file, "data")

        try:
            for ep in range(start_epoch, self.epochs+1):
                self.curr_idx = 0
                self.ep_loss = 0
                self.megabatch = []
                cost = 0
                counter = 0
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batchsize,
                    shuffle=True,
                    num_workers=4,
                    prefetch_factor=50,
                    collate_fn=paraphrastic_dataset.paraphrastic_collate_fn,
                )

                data_iter = iter(data_loader)

                while(cost is not None):
                    cost = pairing.compute_loss_one_batch(self, data_iter)
                    if cost is None:
                        continue

                    self.ep_loss += cost.item()
                    counter += 1
                    #print("Loss: {0}".format(self.ep_loss))

                   # if counter == 4000:
                   #     import sys
                   #     sys.exit()

                    if counter % self.report_interval == 0:
                        print("Epoch {0}, Counter {1}/{2}".format(ep, counter, len(data_iter)))
                    if self.save_interval > 0 and counter > 0:
                        if counter % self.save_interval == 0:
                            self.eval()
                            evaluate_sts(self)
                            self.train()
                            self.save_params(ep, counter=counter)

                    self.optimizer.zero_grad()
                    cost.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
                    self.optimizer.step()

                self.eval()
                evaluate_sts(self)
                self.train()

                if self.save_every_epoch:
                    self.save_params(ep)

                print('Epoch {0}\tCost: '.format(ep), self.ep_loss / counter)

        except KeyboardInterrupt:
            print("Training Interrupted")

        if self.save_final:
            self.save_params(ep)

        end_time = time.time()
        print("Total Time:", (end_time - start_time))

class Averaging(ParaModel):
    def __init__(self, args, vocab):
        super(Averaging, self).__init__(args, vocab)
        self.parameters = self.parameters()
        self.optimizer = optim.Adam(self.parameters, lr=self.lr)

        if self.gpu:
           self.cuda()

        print(self)
        
    def forward(self, curr_batch):
        g_idxs1 = curr_batch.g1
        g_lengths1 = curr_batch.g1_l

        g_idxs2 = curr_batch.g2
        g_lengths2 = curr_batch.g2_l

        p_idxs1 = curr_batch.p1
        p_lengths1 = curr_batch.p1_l

        p_idxs2 = curr_batch.p2
        p_lengths2 = curr_batch.p2_l

        g1 = self.encode(g_idxs1, g_lengths1)
        g2 = self.encode(g_idxs2, g_lengths2, fr=1)
        p1 = self.encode(p_idxs1, p_lengths1, fr=1)
        p2 = self.encode(p_idxs2, p_lengths2)

        return g1, g2, p1, p2

    def encode(self, idxs, lengths, fr=0):
        word_embs = self.embedding(idxs)

        if self.dropout > 0:
            word_embs = F.dropout(word_embs, p=self.dropout, training=self.training)

        if self.pool == "max":
            word_embs = utils.max_pool(word_embs, lengths, self.gpu)
        elif self.pool == "mean":
            word_embs = utils.mean_pool(word_embs, lengths, self.gpu)

        return word_embs

class LSTM(ParaModel):
    def __init__(self, args, vocab):
        super(LSTM, self).__init__(args, vocab)

        self.hidden_dim = self.args.hidden_dim

        self.e_hidden_init = torch.zeros(2, 1, self.args.hidden_dim)
        self.e_cell_init = torch.zeros(2, 1, self.args.hidden_dim)

        if self.gpu:
            self.e_hidden_init = self.e_hidden_init.cuda()
            self.e_cell_init = self.e_cell_init.cuda()

        self.lstm = nn.LSTM(self.args.dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        if not self.share_encoder:
            self.lstm_fr = nn.LSTM(self.args.dim, self.hidden_dim, num_layers=1,
                                       bidirectional=True, batch_first=True)

        self.parameters = self.parameters()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters), self.args.lr)

        if self.gpu:
           self.cuda()

        print(self)

    def encode(self, inputs, lengths, fr=0):
        bsz, max_len = inputs.size()
        e_hidden_init = self.e_hidden_init.expand(2, bsz, self.hidden_dim).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.hidden_dim).contiguous()
        lens, indices = torch.sort(lengths, 0, True)

        if fr and not self.share_vocab:
            in_embs = self.embedding_fr(inputs)
        else:
            in_embs = self.embedding(inputs)

        if fr and not self.share_encoder:
            if self.dropout > 0:
                in_embs = F.dropout(in_embs, p=self.dropout, training=self.training)
            all_hids, (enc_last_hid, _) = self.lstm_fr(pack(in_embs[indices],
                                                        lens.tolist(), batch_first=True), (e_hidden_init, e_cell_init))
        else:
            if self.dropout > 0:
                in_embs = F.dropout(in_embs, p=self.dropout, training=self.training)
            all_hids, (enc_last_hid, _) = self.lstm(pack(in_embs[indices],
                                                         lens.tolist(), batch_first=True), (e_hidden_init, e_cell_init))

        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]

        if self.pool == "max":
            embs = utils.max_pool(all_hids, lengths, self.gpu)
        elif self.pool == "mean":
            embs = utils.mean_pool(all_hids, lengths, self.gpu)
        return embs

    def forward(self, curr_batch):
        g_idxs1 = curr_batch.g1
        g_lengths1 = curr_batch.g1_l

        g_idxs2 = curr_batch.g2
        g_lengths2 = curr_batch.g2_l

        p_idxs1 = curr_batch.p1
        p_lengths1 = curr_batch.p1_l

        p_idxs2 = curr_batch.p2
        p_lengths2 = curr_batch.p2_l

        g1 = self.encode(g_idxs1, g_lengths1)
        g2 = self.encode(g_idxs2, g_lengths2, fr=1)
        p1 = self.encode(p_idxs1, p_lengths1, fr=1)
        p2 = self.encode(p_idxs2, p_lengths2)

        return g1, g2, p1, p2
