from argparse import Namespace
from bart_utils import ShardedBART
import torch
import torch.nn as nn
import traceback


class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        args = Namespace(
            checkpoint=checkpoint
        )
        self.model = ShardedBART(args)

        self.device = device
        self.max_length = max_length

        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

        self.model.eval()
        self.model.to(device)

    def load(self):
        """ Load model from paraphrase finetuning """
        self.model.load_state_dict(torch.load('models/bart_1450.pth', map_location=self.device))

    def score(self, source, target):
        """ Score a single example """
        try:
            with torch.no_grad():
                src_tokens = self.model.tokenizer(
                    [source],
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )['input_ids'].to(self.device)

                tgt_tokens = self.model.tokenizer(
                    [target],
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )['input_ids'].to(self.device)

                output = self.model.model(
                    input_ids=src_tokens,
                    labels=tgt_tokens
                )
                logits = output.logits.view(-1, self.model.config.vocab_size)

                loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                return -loss.mean().item()

        except RuntimeError:
            traceback.print_exc()
            print(f'source: {source}')
            print(f'target: {target}')
            exit(0)

    def score_batch(self, srcs, tgts, batch_size):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.model.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.model.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def test(self):
        """ Test. For debug. """
        src_list = ['I love you so much baby.', 'This is a very good idea. Although simple, but very insightful.']
        tgt_list = ['Do you love me?', "That's stupid."]
        print(f'Batch score')
        print(self.score_batch(src_list, tgt_list, 30))
        print(f'Individual score')
        print(self.score(src_list[0], tgt_list[0]))
        print(self.score(src_list[1], tgt_list[1]))


