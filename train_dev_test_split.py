# Script to generate train-dev-test split (80-10-10 split on 400K words)
# adapted from https://github.com/sleepinyourhat/quora-duplicate-questions-util/blob/master/quora_convert.py

import unicodecsv
import nltk.tokenize

LABELS = ['neutral', 'entailment']

dev_set = []
test_set = []
training_set = []

tt = nltk.tokenize.treebank.TreebankWordTokenizer()
ext = ".tsv"

with open('word2vec/data/quora_duplicate_questions' + ext, 'rbU') as csvfile:
    reader = unicodecsv.reader(csvfile, delimiter="\t")
    for i, row in enumerate(reader):
        if i < 1:
            continue
        if len(row) == 0:
            continue

        example = {}
        example['pairID'] = row[0]
        if len(row[3]) < 1 or len(row[4]) < 1:
            continue
        example['sentence1'] = row[3]
        example['sentence2'] = row[4]
        example['gold_label'] = LABELS[int(row[5])]
        example['pairID'] = row[0]

        example['sentence1_parse'] = example['sentence1_binary_parse'] = ' '.join(tt.tokenize(example['sentence1']))
        example['sentence2_parse'] = example['sentence2_binary_parse'] = ' '.join(tt.tokenize(example['sentence2']))

        if i <= 40000:
            dev_set.append(example)
        elif i <= 80000:
            test_set.append(example)
        else:
            training_set.append(example)


def write_txt(data, filename):
    with open(filename, 'w') as outfile:
        outfile.write(
            'gold_label\tsentence1_binary_parse\tsentence2_binary_parse\tsentence1_parse\tsentence2_parse\tsentence1\tsentence2\tcaptionID\tpairID\tlabel1\tlabel2\tlabel3\tlabel4\tlabel5\n')
        for item in data:
            tab_sep_string = item['gold_label'] + '\t' + \
                item['sentence1_binary_parse'] + '\t' + item['sentence2_binary_parse'] + '\t' + \
                item['sentence1_parse'] + '\t' + item['sentence2_parse'] + '\t' + \
                item['sentence1'] + '\t' + item['sentence2'] + \
                '\t\t' + item['pairID']
            for i in range(5):
                tab_sep_string += '\t'
            outfile.write(tab_sep_string + "\n")
        outfile.close()

print(len(training_set), len(dev_set), len(test_set))
write_txt(training_set, 'data_split/quora_duplicate_questions_train.txt')
write_txt(dev_set, 'data_split/quora_duplicate_questions_dev.txt')
write_txt(test_set, 'data_split/quora_duplicate_questions_test.txt')
