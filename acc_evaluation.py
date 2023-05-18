import json
import argparse
import warnings
import pandas as pd


warnings.filterwarnings('ignore')


def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def faithful_inference(results, bleu1s, bleu4s, rouges, similarities):
    faithfulness = {}
    for i in range(len(results)):
        key = results[i]
        if key not in faithfulness:
            faithfulness[key] = 0
        
        faithfulness[key] += bleu1s[i] + bleu4s[i] + rouges[i] + similarities[i] # TODO: weights
    
    return int(max(faithfulness, key=faithfulness.get))


def get_scores(result_file, data_file, n_paths, faithful=False):
    # read result file
    results = json.load(open(result_file))["results"]
    num = len(results)
    assert num == 4241

    if faithful:
        bleu1s = json.load(open(result_file))["bleu1s"]
        bleu4s = json.load(open(result_file))["bleu4s"]
        rouges = json.load(open(result_file))["rouges"]
        similarities = json.load(open(result_file))["similarities"]

    # read data file
    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

    # update data
    for index, row in res_pd.iterrows():

        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']

        if isinstance(results[index], list):
            n_results = results[index][:n_paths]

            if faithful:
                n_bleu1s = bleu1s[index][:n_paths]
                n_bleu4s = bleu4s[index][:n_paths]
                n_rouges = rouges[index][:n_paths]
                n_similarities = similarities[index][:n_paths]
                pred = faithful_inference(n_results, n_bleu1s, n_bleu4s, n_rouges, n_similarities) 
            else:
                pred = max(n_results, key=n_results.count) # self-consistency (majority vote)
        else:
            pred = int(results[index])

        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100
    #assert result_file.split('_')[-1] == "{:.3f}.json".format(acc_average)

    scores = {
        'acc_natural':
        get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social':
        get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language':
        get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_has_text':
        get_acc_with_contion(res_pd, 'has_text', True),
        'acc_has_image':
        get_acc_with_contion(res_pd, 'has_image', True),
        'acc_no_context':
        get_acc_with_contion(res_pd, 'no_context', True),
        'acc_grade_1_6':
        get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
        get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
        "{:.2f}".format(acc_average),
    }

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default="data/scienceqa/problems.json")
    parser.add_argument('--result_file', type=str, default="results/chatgpt/test/QCM-ALE/visual_clues_ocrs.json/paths_10/2shots_seed_3.json")
    parser.add_argument('--n_paths', type=int, default=10)
    parser.add_argument('--faithful_inference', action='store_true', default=False)
    args = parser.parse_args()

    print("Data file: ", args.data_file)
    print("Result file: ", args.result_file)

    scores = get_scores(args.result_file, args.data_file, args.n_paths, args.faithful_inference)
    print_scores(scores)