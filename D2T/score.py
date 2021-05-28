import argparse
import os
import time
import numpy as np
from utils import *

SRC_HYPO = read_file_to_list('files/src_hypo_prompt.txt')
REF_HYPO = read_file_to_list('files/ref_hypo_prompt.txt')


class Scorer:
    """ Support ROUGE-1,2,L, BERTScore, MoverScore, PRISM, BARTScore """

    def __init__(self, file_path, device='cuda:0'):
        """ file_path: path to the pickle file
            All the data are normal capitalized, and tokenized, including ref_summs, and sys_summ.
        """
        self.device = device
        self.data = read_pickle(file_path)
        print(f'Data loaded from {file_path}.')

    def save_data(self, path):
        save_pickle(self.data, path)

    def score(self, metrics):
        """ metrics: list of metrics """
        for metric_name in metrics:
            if metric_name == 'bert_score':
                from bert_score import BERTScorer

                # Set up BERTScore
                bert_scorer = BERTScorer(
                    lang='en',
                    idf=False,
                    rescale_with_baseline=True,
                    device=self.device
                )
                print(f'BERTScore setup finished. Begin calculating BERTScore.')

                start = time.time()
                for doc_id in self.data:
                    ref_summs = self.data[doc_id]['ref_summs']
                    sys_summ = self.data[doc_id]['sys_summ']
                    P, R, F = bert_scorer.score([sys_summ] * len(ref_summs), ref_summs)
                    P = P.max().item()
                    R = R.max().item()
                    F = F.max().item()
                    self.data[doc_id]['scores']['bert_score_p'] = P
                    self.data[doc_id]['scores']['bert_score_r'] = R
                    self.data[doc_id]['scores']['bert_score_f'] = F
                print(f'Finished calculating BERTScore, time passed {time.time() - start}s.')

            elif metric_name == 'mover_score':
                from moverscore import word_mover_score, get_idf_dict

                # Set up MoverScore
                self.stop_words = []

                ref_lines = []
                for doc_id in self.data:
                    ref_lines.extend(self.data[doc_id]['ref_summs'])
                ref_lines = list(set(ref_lines))
                self.idf_refs = get_idf_dict(ref_lines)

                # IDF for all system hypos, used for MoverScore
                sys_lines = []
                for doc_id in self.data:
                    sys_summ = self.data[doc_id]['sys_summ']
                    sys_lines.append(sys_summ)
                self.idf_hyps = get_idf_dict(sys_lines)
                print(f'MoverScore setup finished. Begin calculating MoverScore.')

                start = time.time()
                for doc_id in self.data:
                    ref_summs = self.data[doc_id]['ref_summs']
                    sys_summ = self.data[doc_id]['sys_summ']
                    scores = word_mover_score(ref_summs, [sys_summ] * len(ref_summs), self.idf_refs, self.idf_hyps,
                                              self.stop_words,
                                              n_gram=1, remove_subwords=True, batch_size=48, device=self.device)
                    score = max(scores)
                    self.data[doc_id]['scores']['mover_score'] = score
                print(f'Finished calculating MoverScore, time passed {time.time() - start}s.')

            elif metric_name == 'rouge':
                import files2rouge

                def rouge(saveto):
                    def get_r_p_f(line):
                        line_ = line.split(" ")
                        r = float(line_[-3][-7:])
                        p = float(line_[-2][-7:])
                        f = float(line_[-1][-7:])
                        return [r, p, f]

                    lines = read_file_to_list(saveto)
                    r1_, r2_, rl_ = None, None, None
                    for line in lines:
                        if line.startswith('1 ROUGE-1 Eval'):
                            r1_ = get_r_p_f(line)
                        if line.startswith('1 ROUGE-2 Eval'):
                            r2_ = get_r_p_f(line)
                        if line.startswith('1 ROUGE-L Eval'):
                            rl_ = get_r_p_f(line)
                    return r1_, r2_, rl_

                print(f'Begin calculating ROUGE.')
                start = time.time()
                blockPrint()
                for doc_id in self.data:
                    ref_summs = self.data[doc_id]['ref_summs']
                    sys_summ = self.data[doc_id]['sys_summ']
                    sys_summ = sys_summ.lower()
                    write_list_to_file([sys_summ], 'hypo.txt')
                    r1, r2, rl = [], [], []
                    for ref_summ in ref_summs:
                        ref_summ = ref_summ.lower()
                        write_list_to_file([ref_summ], 'ref.txt')
                        files2rouge.run('hypo.txt', 'ref.txt', rouge_args="-c 95 -r 1000 -n 2 -a -d",
                                        saveto='saved_out.txt')
                        curr_r1, curr_r2, curr_rl = rouge('saved_out.txt')
                        r1.append(curr_r1)
                        r2.append(curr_r2)
                        rl.append(curr_rl)
                    r1 = np.array(r1)
                    r2 = np.array(r2)
                    rl = np.array(rl)

                    r1 = np.max(r1, axis=0)
                    r2 = np.max(r2, axis=0)
                    rl = np.max(rl, axis=0)
                    self.data[doc_id]['scores']['rouge1_r'] = r1[0]
                    self.data[doc_id]['scores']['rouge1_p'] = r1[1]
                    self.data[doc_id]['scores']['rouge1_f'] = r1[2]
                    self.data[doc_id]['scores']['rouge2_r'] = r2[0]
                    self.data[doc_id]['scores']['rouge2_p'] = r2[1]
                    self.data[doc_id]['scores']['rouge2_f'] = r2[2]
                    self.data[doc_id]['scores']['rougel_r'] = rl[0]
                    self.data[doc_id]['scores']['rougel_p'] = rl[1]
                    self.data[doc_id]['scores']['rougel_f'] = rl[2]
                enablePrint()
                os.system('rm -rf hypo.txt ref.txt saved_out.txt')
                print(f'Finished calculating ROUGE, time passed {time.time() - start}s.')

            elif metric_name == 'prism':
                from prism import Prism
                # Set up Prism
                self.prism = Prism(model_dir='./models/m39v1/', lang='en')
                print(f'PRISM setup finished. Begin calculating PRISM.')

                start = time.time()
                for doc_id in self.data:
                    ref_summs = self.data[doc_id]['ref_summs']
                    ref_summs = [detokenize(line) for line in ref_summs]
                    sys_summ = detokenize(self.data[doc_id]['sys_summ'])
                    ref_hypo_scores, hypo_ref_scores, scores = self.prism.score(cand=[sys_summ] * len(ref_summs),
                                                                                ref=ref_summs,
                                                                                segment_scores=True)
                    ref_hypo, hypo_ref, score = max(ref_hypo_scores), max(hypo_ref_scores), max(scores)
                    self.data[doc_id]['scores']['prism_ref_hypo'] = ref_hypo
                    self.data[doc_id]['scores']['prism_hypo_ref'] = hypo_ref
                    self.data[doc_id]['scores']['prism_avg'] = score
                print(f'Finished calculating PRISM, time passed {time.time() - start}s.')

            elif metric_name == 'bart_score' or metric_name == 'bart_score_cnn' or metric_name == 'bart_score_para':
                """ Vanilla BARTScore, BARTScore-CNN, BARTScore-CNN-Para """
                from bart_score import BARTScorer

                # Set up BARTScore
                if 'cnn' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')
                print(f'BARTScore setup finished. Begin calculating BARTScore.')

                start = time.time()
                for doc_id in self.data:
                    ref_summs = self.data[doc_id]['ref_summs']
                    ref_summs = [detokenize(line) for line in ref_summs]
                    sys_summ = detokenize(self.data[doc_id]['sys_summ'])

                    ref_hypo_scores = np.array(bart_scorer.score_batch(ref_summs, [sys_summ] * len(ref_summs), batch_size=4))
                    hypo_ref_scores = np.array(bart_scorer.score_batch([sys_summ] * len(ref_summs), ref_summs, batch_size=4))
                    ref_hypo = ref_hypo_scores.max()
                    hypo_ref = hypo_ref_scores.max()
                    avg_f = (0.5 * (ref_hypo_scores + hypo_ref_scores)).max()
                    harm_f = (ref_hypo_scores * hypo_ref_scores / (ref_hypo_scores + hypo_ref_scores)).max()

                    self.data[doc_id]['scores'][f'{metric_name}_ref_hypo'] = ref_hypo
                    self.data[doc_id]['scores'][f'{metric_name}_hypo_ref'] = hypo_ref
                    self.data[doc_id]['scores'][f'{metric_name}_avg_f'] = avg_f
                    self.data[doc_id]['scores'][f'{metric_name}_harm_f'] = harm_f
                print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')

            elif metric_name.startswith('prompt'):
                """ BARTScore adding prompts """
                from bart_score import BARTScorer

                def prefix_prompt(l, p):
                    new_l = []
                    for x in l:
                        new_l.append(p + ', ' + x)
                    return new_l

                def suffix_prompt(l, p):
                    new_l = []
                    for x in l:
                        new_l.append(x + ' ' + p + ',')
                    return new_l

                if 'cnn' in metric_name:
                    name = 'bart_score_cnn'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    name = 'bart_score_para'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    name = 'bart_score'
                    bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')

                print(f'BARTScore-P setup finished. Begin calculating BARTScore-P.')
                start = time.time()
                for doc_id in self.data:
                    ref_summs = self.data[doc_id]['ref_summs']
                    ref_summs = [detokenize(line) for line in ref_summs]
                    sys_summ = detokenize(self.data[doc_id]['sys_summ'])
                    sys_summs = [sys_summ] * len(ref_summs)
                    for prompt in REF_HYPO:
                        ref_hypo_scores_en = np.array(
                            bart_scorer.score_batch(suffix_prompt(ref_summs, prompt), sys_summs, batch_size=4))
                        hypo_ref_scores_en = np.array(
                            bart_scorer.score_batch(suffix_prompt(sys_summs, prompt), ref_summs, batch_size=4))
                        ref_hypo_scores_de = np.array(
                            bart_scorer.score_batch(ref_summs, prefix_prompt(sys_summs, prompt), batch_size=4))
                        hypo_ref_scores_de = np.array(
                            bart_scorer.score_batch(sys_summs, prefix_prompt(ref_summs, prompt), batch_size=4))
                        ref_hypo_en = ref_hypo_scores_en.max()
                        hypo_ref_en = hypo_ref_scores_en.max()
                        avg_f_en = (0.5 * (ref_hypo_scores_en + hypo_ref_scores_en)).max()
                        harm_f_en = (ref_hypo_scores_en * hypo_ref_scores_en / (ref_hypo_scores_en + hypo_ref_scores_en)).max()
                        ref_hypo_de = ref_hypo_scores_de.max()
                        hypo_ref_de = hypo_ref_scores_de.max()
                        avg_f_de = (0.5 * (ref_hypo_scores_de + hypo_ref_scores_de)).max()
                        harm_f_de = (ref_hypo_scores_de * hypo_ref_scores_de / (ref_hypo_scores_de + hypo_ref_scores_de)).max()

                        self.data[doc_id]['scores'][f'{name}_ref_hypo_en_{prompt}'] = ref_hypo_en
                        self.data[doc_id]['scores'][f'{name}_hypo_ref_en_{prompt}'] = hypo_ref_en
                        self.data[doc_id]['scores'][f'{name}_avg_f_en_{prompt}'] = avg_f_en
                        self.data[doc_id]['scores'][f'{name}_harm_f_en_{prompt}'] = harm_f_en
                        self.data[doc_id]['scores'][f'{name}_ref_hypo_de_{prompt}'] = ref_hypo_de
                        self.data[doc_id]['scores'][f'{name}_hypo_ref_de_{prompt}'] = hypo_ref_de
                        self.data[doc_id]['scores'][f'{name}_avg_f_de_{prompt}'] = avg_f_de
                        self.data[doc_id]['scores'][f'{name}_harm_f_de_{prompt}'] = harm_f_de
                print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')

            else:
                raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--output', type=str, required=True,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--mover_score', action='store_true', default=False,
                        help='Whether to calculate MoverScore')
    parser.add_argument('--rouge', action='store_true', default=False,
                        help='Whether to calculate ROUGE')
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    parser.add_argument('--bart_score_cnn', action='store_true', default=False,
                        help='Whether to calculate BARTScore-CNN')
    parser.add_argument('--bart_score_para', action='store_true', default=False,
                        help='Whether to calculate BARTScore-Para')
    parser.add_argument('--prism', action='store_true', default=False,
                        help='Whether to calculate PRISM')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Whether to calculate BARTScore-P. Can be bart_ref, '
                             'bart_cnn_ref, bart_para_ref')
    args = parser.parse_args()

    scorer = Scorer(args.file, args.device)

    METRICS = []
    if args.bert_score:
        METRICS.append('bert_score')
    if args.mover_score:
        METRICS.append('mover_score')
    if args.rouge:
        METRICS.append('rouge')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.bart_score_para:
        METRICS.append('bart_score_para')
    if args.prism:
        METRICS.append('prism')
    if args.prompt is not None:
        prompt = args.prompt
        assert prompt in ['bart_ref', 'bart_cnn_ref', 'bart_para_ref']
        METRICS.append(f'prompt_{prompt}')

    scorer.score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()

"""
python score.py --file BAGEL/data.pkl --device cuda:1 --output BAGEL/scores.pkl --bert_score --mover_score --bart_score --bart_score_cnn --bart_score_para --prism

"""