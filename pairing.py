import torch
from torch.nn.functional import pad
from utils import Batch
from utils import BigExample


def get_pairs_batch(model, g1, g1_lengths, g2, g2_lengths):
    with torch.no_grad():

        model.eval()
        all_g1_lengths = torch.cat(g1_lengths)
        all_g2_lengths = torch.cat(g2_lengths)

        g1_max_length = max(i.size(1) for i in g1)
        all_g1 = torch.cat([
            pad(i, (0, g1_max_length - i.size(1)), "constant", 0)
            for i in g1
        ])

        g2_max_length = max(i.size(1) for i in g2)
        all_g2 = torch.cat([
            pad(i, (0, g2_max_length - i.size(1)), "constant", 0)
            for i in g2
        ])

        v_g1 = []
        for i in range(len(g1)):
            v_g1.append(model.encode(g1[i], g1_lengths[i]))

        v_g2 = []
        for i in range(len(g2)):
            v_g2.append(model.encode(g2[i], g2_lengths[i], fr=1))

        v_g1 = torch.cat(v_g1)
        v_g2 = torch.cat(v_g2)

        v_g1_norm = v_g1 / v_g1.norm(dim=1, keepdim=True)
        v_g2_norm = v_g2 / v_g2.norm(dim=1, keepdim=True)

        sims = torch.matmul(v_g2_norm, v_g1_norm.t())
        eye = torch.eye(sims.shape[0]).cuda()
        sims = (1 - eye) * sims - eye
        _, ids = torch.max(sims, 0)
        p1 = all_g2[ids]
        p1_lengths = all_g2_lengths[ids]

        sims = torch.matmul(v_g1_norm, v_g2_norm.t())
        sims = (1 - eye) * sims - eye
        _, ids = torch.max(sims, 0)
        p2 = all_g1[ids]
        p2_lengths = all_g1_lengths[ids]

        def split(arr, lis):
            idx = 0
            output = []
            for i in lis:
                arr2 = arr[idx:idx+i]
                output.append(arr2)
                idx += i
            return output

        p1 = split(p1, [len(i) for i in g2])
        p1_lengths = split(p1_lengths, [len(i) for i in g2])

        p2 = split(p2, [len(i) for i in g1])
        p2_lengths = split(p2_lengths, [len(i) for i in g1])

        _p1 = []
        for i in range(len(p1)):
            _p1.append(p1[i][:,0:max(p1_lengths[i])])
        p1 = _p1

        _p2 = []
        for i in range(len(p2)):
            _p2.append(p2[i][:,0:max(p2_lengths[i])])
        p2 = _p2

        model.train()

        return p1, p1_lengths, p2, p2_lengths

def compute_loss_one_batch(model):
    if len(model.megabatch) == 0:

        if model.megabatch_anneal == 0:
            for i in range(model.max_megabatch_size):
                if model.curr_idx < len(model.mb):
                    model.megabatch.append(model.mb[model.curr_idx][1])
                    model.curr_idx += 1
        else:
            if model.increment and model.curr_megabatch_size < model.max_megabatch_size:
                model.curr_megabatch_size += 1
                model.increment = False
                print("Increasing megabatch size to {0}".format(model.curr_megabatch_size))

            for i in range(model.curr_megabatch_size):
                if model.curr_idx < len(model.mb):
                    model.megabatch.append(model.mb[model.curr_idx][1])
                    model.curr_idx += 1
                    if model.curr_idx % model.megabatch_anneal == 0:
                        model.increment = True

        megabatch = []
        for n, i in enumerate(model.megabatch):
            arr = [model.data[t] for t in i]
            example_arr = []
            for j in arr:
                example = (BigExample(j[0], model.vocab, model.rev_vocab, model.scramble_rate),
                           BigExample(j[1], model.vocab, model.rev_vocab, model.scramble_rate))
                if model.args.debug:
                    print("Logging Pairing: {0} {1}".format(j[0].sentence, j[1].sentence))

                example_arr.append(example)
            megabatch.append(example_arr)

        model.megabatch = megabatch

        if len(model.megabatch) == 0:
            return None

        sents1_list = []
        sents2_list = []

        sents1_lengths_list = []
        sents2_lengths_list = []

        for j in model.megabatch:

            sents1 = [i[0] for i in j]
            sents2 = [i[1] for i in j]

            sents_1_torch, lengths_1_torch = model.torchify_batch(sents1)
            if model.gpu:
                sents_1_torch = sents_1_torch.cuda()
                lengths_1_torch = lengths_1_torch.cuda()

            sents_2_torch, lengths_2_torch = model.torchify_batch(sents2)
            if model.gpu:
                sents_2_torch = sents_2_torch.cuda()
                lengths_2_torch = lengths_2_torch.cuda()

            sents1_list.append(sents_1_torch)
            sents2_list.append(sents_2_torch)

            sents1_lengths_list.append(lengths_1_torch)
            sents2_lengths_list.append(lengths_2_torch)

        p1_sents_list, p1_lengths_list, p2_sents_list, p2_lengths_list, = get_pairs_batch(model, sents1_list,
                                                sents1_lengths_list, sents2_list, sents2_lengths_list)

        model.megabatch = []
        for i in range(len(p1_sents_list)):
            new_batch = Batch()
            new_batch.g1 = sents1_list[i]
            new_batch.g1_l = sents1_lengths_list[i]

            new_batch.g2 = sents2_list[i]
            new_batch.g2_l = sents2_lengths_list[i]

            new_batch.p1 = p1_sents_list[i]
            new_batch.p1_l = p1_lengths_list[i]

            new_batch.p2 = p2_sents_list[i]
            new_batch.p2_l = p2_lengths_list[i]

            model.megabatch.append(new_batch)

    curr_batch = model.megabatch.pop(0)

    g1, g2, p1, p2 = model.forward(curr_batch)

    return model.loss_function(g1, g2, p1, p2)
