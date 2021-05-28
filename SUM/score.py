import argparse
import os
import time
import numpy as np
from utils import *

SRC_HYPO = read_file_to_list('files/src_hypo_prompt.txt')
REF_HYPO = read_file_to_list('files/ref_hypo_prompt.txt')


class Scorer:
    """ Support ROUGE-1,2,L, BERTScore, MoverScore, PRISM, BARTScore """

    def __init__(self, file_path, device='cuda:0', multi_ref=False):
        """ file_path: path to the pickle file
            All the data are normal capitalized, and tokenized, including src, ref_summ, ref_summs, and sys_summ.
        """
        self.multi_ref = multi_ref
        self.device = device
        self.data = read_pickle(file_path)
        print(f'Data loaded from {file_path}.')

        self.sys_names = self.get_sys_names()

        if not multi_ref:
            self.single_ref_lines = self.get_single_ref_lines()
            print(f'In a single-reference setting.')
        else:
            self.multi_ref_lines = self.get_multi_ref_lines()
            self.ref_num = len(self.multi_ref_lines[0])
            print(f'In a multi-reference setting.')

    def get_sys_names(self):
        first_id = list(self.data.keys())[0]
        return list(self.data[first_id]['sys_summs'].keys())

    def get_single_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summ'])
        return ref_lines

    def get_multi_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summs'])
        return ref_lines

    def get_sys_lines(self, sys_name):
        sys_lines = []
        for doc_id in self.data:
            sys_lines.append(self.data[doc_id]['sys_summs'][sys_name]['sys_summ'])
        return sys_lines

    def get_src_lines(self):
        src_lines = []
        for doc_id in self.data:
            src_lines.append(self.data[doc_id]['src'])
        return src_lines

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
                ref_lines = self.single_ref_lines if not self.multi_ref else self.multi_ref_lines
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    if not self.multi_ref:
                        P, R, F = bert_scorer.score(sys_lines, ref_lines)
                    else:
                        total_num = len(sys_lines)
                        P, R, F = np.zeros(total_num), np.zeros(total_num), np.zeros(total_num)
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_P, curr_R, curr_F = bert_scorer.score(sys_lines, ref_list)
                            P += curr_P.numpy()
                            R += curr_R.numpy()
                            F += curr_F.numpy()
                        P, R, F = P / self.ref_num, R / self.ref_num, F / self.ref_num
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                            'bert_score_p': P[counter],
                            'bert_score_r': R[counter],
                            'bert_score_f': F[counter]
                        })
                        counter += 1
                print(f'Finished calculating BERTScore, time passed {time.time() - start}s.')

            elif metric_name == 'mover_score':
                from moverscore import word_mover_score, get_idf_dict

                # Set up MoverScore
                with open('files/stopwords.txt', 'r', encoding='utf-8') as f:
                    self.stop_words = set(f.read().strip().split(' '))

                # IDF for all system hypos, used for MoverScore
                self.sys_lines = []
                for name in self.sys_names:
                    sys_lines = self.get_sys_lines(name)
                    self.sys_lines.extend(sys_lines)
                self.idf_hyps = get_idf_dict(self.sys_lines)
                print(f'MoverScore setup finished. Begin calculating MoverScore.')

                start = time.time()
                if not self.multi_ref:
                    ref_lines = self.single_ref_lines
                    idf_refs = get_idf_dict(ref_lines)
                else:
                    ref_lines = self.multi_ref_lines
                    idf_refs = get_idf_dict(sum(ref_lines, []))
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    if not self.multi_ref:
                        scores = word_mover_score(ref_lines, sys_lines, idf_refs, self.idf_hyps, self.stop_words,
                                                  n_gram=1, remove_subwords=True, batch_size=48, device=self.device)
                    else:
                        scores = np.zeros(len(sys_lines))
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_scores = word_mover_score(ref_list, sys_lines, idf_refs, self.idf_hyps,
                                                           self.stop_words, n_gram=1, remove_subwords=True,
                                                           batch_size=48, device=self.device)
                            scores += np.array(curr_scores)
                        scores = scores / self.ref_num
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['mover_score'] = scores[counter]
                        counter += 1
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
                if not self.multi_ref:
                    ref_lines = [line.lower() for line in self.single_ref_lines]
                else:
                    ref_lines = [[text.lower() for text in line] for line in self.multi_ref_lines]

                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [line.lower() for line in sys_lines]
                    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
                    for sys_line, ref_line in zip(sys_lines, ref_lines):
                        write_list_to_file([sys_line], 'hypo.txt')
                        if not self.multi_ref:
                            write_list_to_file([ref_line], 'ref.txt')
                            files2rouge.run('hypo.txt', 'ref.txt', rouge_args="-c 95 -r 1000 -n 2 -a -d",
                                            saveto='saved_out.txt')
                            r1, r2, rl = rouge('saved_out.txt')
                            rouge1_scores.append(r1)
                            rouge2_scores.append(r2)
                            rougel_scores.append(rl)
                        else:
                            r1, r2, rl = [], [], []
                            for i in range(self.ref_num):
                                write_list_to_file([ref_line[i]], 'ref.txt')
                                files2rouge.run('hypo.txt', 'ref.txt', rouge_args="-c 95 -r 1000 -n 2 -a -d",
                                                saveto='saved_out.txt')
                                curr_r1, curr_r2, curr_rl = rouge('saved_out.txt')
                                r1.append(curr_r1)
                                r2.append(curr_r2)
                                rl.append(curr_rl)
                            r1, r2, rl = np.array(r1), np.array(r2), np.array(rl)
                            r1, r2, rl = np.mean(r1, axis=0), np.mean(r2, axis=0), np.mean(rl, axis=0)
                            rouge1_scores.append(r1)
                            rouge2_scores.append(r2)
                            rougel_scores.append(rl)
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                            'rouge1_r': rouge1_scores[counter][0],
                            'rouge1_p': rouge1_scores[counter][1],
                            'rouge1_f': rouge1_scores[counter][2],
                            'rouge2_r': rouge2_scores[counter][0],
                            'rouge2_p': rouge2_scores[counter][1],
                            'rouge2_f': rouge2_scores[counter][2],
                            'rougel_r': rougel_scores[counter][0],
                            'rougel_p': rougel_scores[counter][1],
                            'rougel_f': rougel_scores[counter][2]
                        })
                        counter += 1
                enablePrint()
                os.system('rm -rf hypo.txt ref.txt saved_out.txt')
                print(f'Finished calculating ROUGE, time passed {time.time() - start}s.')

            elif metric_name == 'prism':
                from prism import Prism
                # Set up Prism
                self.prism = Prism(model_dir='./models/m39v1/', lang='en')
                print(f'PRISM setup finished. Begin calculating PRISM.')

                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                src_lines = [detokenize(line) for line in src_lines]
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(line) for line in sys_lines]
                    # Calculate Both src-based and ref-based
                    src_hypo_scores = self.prism.score(cand=sys_lines, src=src_lines, segment_scores=True)
                    if not self.multi_ref:
                        ref_hypo_scores, hypo_ref_scores, scores = self.prism.score(cand=sys_lines, ref=ref_lines,
                                                                                    segment_scores=True)
                    else:
                        total_num = len(sys_lines)
                        ref_hypo_scores, hypo_ref_scores, scores = np.zeros(total_num), np.zeros(total_num), np.zeros(
                            total_num)
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_ref_hypo_scores, curr_hypo_ref_scores, curr_scores = self.prism.score(cand=sys_lines,
                                                                                                       ref=ref_list,
                                                                                                       segment_scores=True)
                            ref_hypo_scores += curr_ref_hypo_scores
                            hypo_ref_scores += curr_hypo_ref_scores
                            scores += curr_scores

                        ref_hypo_scores = ref_hypo_scores / self.ref_num
                        hypo_ref_scores = hypo_ref_scores / self.ref_num
                        scores = scores / self.ref_num

                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_ref_hypo'] = ref_hypo_scores[counter]
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_hypo_ref'] = hypo_ref_scores[counter]
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_avg'] = scores[counter]
                        self.data[doc_id]['sys_summs'][sys_name]['scores']['prism_src_hypo'] = src_hypo_scores[counter]
                        counter += 1
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
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                src_lines = [detokenize(line) for line in src_lines]
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]
                for sys_name in self.sys_names:
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(line) for line in sys_lines]
                    src_hypo = bart_scorer.score_batch(src_lines, sys_lines, batch_size=4)
                    if not self.multi_ref:
                        ref_hypo = np.array(bart_scorer.score_batch(ref_lines, sys_lines, batch_size=4))
                        hypo_ref = np.array(bart_scorer.score_batch(sys_lines, ref_lines, batch_size=4))
                    else:
                        ref_hypo, hypo_ref = np.zeros(len(sys_lines)), np.zeros(len(sys_lines))
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_ref_hypo = np.array(bart_scorer.score_batch(ref_list, sys_lines, batch_size=4))
                            curr_hypo_ref = np.array(bart_scorer.score_batch(sys_lines, ref_list, batch_size=4))
                            ref_hypo += curr_ref_hypo
                            hypo_ref += curr_hypo_ref
                        ref_hypo = ref_hypo / self.ref_num
                        hypo_ref = hypo_ref / self.ref_num
                    avg_f = (ref_hypo + hypo_ref) / 2
                    harm_f = (ref_hypo * hypo_ref) / (ref_hypo + hypo_ref)
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                            f'{metric_name}_src_hypo': src_hypo[counter],
                            f'{metric_name}_hypo_ref': hypo_ref[counter],
                            f'{metric_name}_ref_hypo': ref_hypo[counter],
                            f'{metric_name}_avg_f': avg_f[counter],
                            f'{metric_name}_harm_f': harm_f[counter]
                        })
                        counter += 1
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
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                src_lines = [detokenize(line) for line in src_lines]
                if not self.multi_ref:
                    ref_lines = [detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]

                # SRC -> HYPO prompt
                if 'src' in metric_name:
                    for prompt in SRC_HYPO:
                        for sys_name in self.sys_names:
                            sys_lines = self.get_sys_lines(sys_name)
                            sys_lines = [detokenize(line) for line in sys_lines]
                            src_hypo_en = bart_scorer.score_batch(suffix_prompt(src_lines, prompt), sys_lines, batch_size=4)
                            src_hypo_de = bart_scorer.score_batch(src_lines, prefix_prompt(sys_lines, prompt), batch_size=4)
                            counter = 0
                            for doc_id in self.data:
                                self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                                    f'{name}_src_hypo_en_{prompt}': src_hypo_en[counter],
                                    f'{name}_src_hypo_de_{prompt}': src_hypo_de[counter]
                                })
                                counter += 1

                # REF <-> HYPO prompt
                if 'ref' in metric_name:
                    for prompt in REF_HYPO:
                        for sys_name in self.sys_names:
                            sys_lines = self.get_sys_lines(sys_name)
                            sys_lines = [detokenize(line) for line in sys_lines]
                            if not self.multi_ref:
                                ref_hypo_en = np.array(bart_scorer.score_batch(suffix_prompt(ref_lines, prompt), sys_lines, batch_size=4))
                                hypo_ref_en = np.array(bart_scorer.score_batch(suffix_prompt(sys_lines, prompt), ref_lines, batch_size=4))
                                ref_hypo_de = np.array(bart_scorer.score_batch(ref_lines, prefix_prompt(sys_lines, prompt), batch_size=4))
                                hypo_ref_de = np.array(bart_scorer.score_batch(sys_lines, prefix_prompt(ref_lines, prompt), batch_size=4))
                            else:
                                ref_hypo_en, hypo_ref_en, ref_hypo_de, hypo_ref_de = np.zeros(len(sys_lines)), np.zeros(len(sys_lines)), \
                                                                                     np.zeros(len(sys_lines)), np.zeros(len(sys_lines))
                                for i in range(self.ref_num):
                                    ref_list = [x[i] for x in ref_lines]
                                    curr_ref_hypo_en = np.array(bart_scorer.score_batch(suffix_prompt(ref_list, prompt), sys_lines, batch_size=4))
                                    curr_hypo_ref_en = np.array(bart_scorer.score_batch(suffix_prompt(sys_lines, prompt), ref_list, batch_size=4))
                                    curr_ref_hypo_de = np.array(bart_scorer.score_batch(ref_list, prefix_prompt(sys_lines, prompt), batch_size=4))
                                    curr_hypo_ref_de = np.array(bart_scorer.score_batch(sys_lines, prefix_prompt(ref_list, prompt), batch_size=4))
                                    ref_hypo_en += curr_ref_hypo_en
                                    hypo_ref_en += curr_hypo_ref_en
                                    ref_hypo_de += curr_ref_hypo_de
                                    hypo_ref_de += curr_hypo_ref_de
                                ref_hypo_en = ref_hypo_en / self.ref_num
                                hypo_ref_en = hypo_ref_en / self.ref_num
                                ref_hypo_de = ref_hypo_de / self.ref_num
                                hypo_ref_de = hypo_ref_de / self.ref_num
                            avg_f_en = (ref_hypo_en + hypo_ref_en) / 2
                            avg_f_de = (ref_hypo_de + hypo_ref_de) / 2
                            harm_f_en = (ref_hypo_en * hypo_ref_en) / (ref_hypo_en + hypo_ref_en)
                            harm_f_de = (ref_hypo_de * hypo_ref_de) / (ref_hypo_de + hypo_ref_de)
                            counter = 0
                            for doc_id in self.data:
                                self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                                    f'{name}_hypo_ref_en_{prompt}': hypo_ref_en[counter],
                                    f'{name}_ref_hypo_en_{prompt}': ref_hypo_en[counter],
                                    f'{name}_avg_f_en_{prompt}': avg_f_en[counter],
                                    f'{name}_harm_f_en_{prompt}': harm_f_en[counter],
                                    f'{name}_hypo_ref_de_{prompt}': hypo_ref_de[counter],
                                    f'{name}_ref_hypo_de_{prompt}': ref_hypo_de[counter],
                                    f'{name}_avg_f_de_{prompt}': avg_f_de[counter],
                                    f'{name}_harm_f_de_{prompt}': harm_f_de[counter]
                                })
                                counter += 1
                print(f'Finished calculating BARTScore-P, time passed {time.time() - start}s.')


            else:
                raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--multi_ref', action='store_true', default=False,
                        help='Whether we are using multiple references to calculate scores.')
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
                        help='Whether to calculate BARTScore-P. Can be bart_src, bart_ref, bart_cnn_src, '
                             'bart_cnn_ref, bart_para_src, bart_para_ref')
    args = parser.parse_args()

    scorer = Scorer(args.file, args.device, args.multi_ref)

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
        assert prompt in ['bart_src', 'bart_ref', 'bart_cnn_src',
                          'bart_cnn_ref', 'bart_para_src', 'bart_para_ref']
        METRICS.append(f'prompt_{prompt}')

    scorer.score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()

"""
python score.py --file Newsroom/data.pkl --device cuda:0 --output Newsroom/scores.pkl --bert_score --mover_score --rouge --bart_score --bart_score_cnn --bart_score_para --prism

"""
