# Copied from https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr/metrics_structured_rep.py
# python metrics_structured_rep.py CVS_PREDICTIONS_FILE TEST_JSON_FILE
import json
import sys

# Load the predictions file. Assume it is a CSV.
predictions = {}
for line in open(sys.argv[1]).readlines():
    if line:
        splits = line.strip().split(",")
        # We assume identifiers are in the format "split-####-#".
        identifier = splits[0]
        prediction = splits[1]
        predictions[identifier] = prediction

# Load the labeled examples.
labeled_examples = [json.loads(line) for line in open(sys.argv[2]).readlines() if line]

# There should be len(labeled_examples) predictions. If not, identify the ones
# that are missing, and exit.
total_num = len(labeled_examples)
if len(predictions) != total_num:
    print("Some predictions are missing!")

    for example in labeled_examples:
        lookup = example["identifier"]
        if not lookup in predictions:
            print("Missing prediction for item " + str(lookup))
    exit()

# Get the precision by iterating through the examples and checking the value
# that was predicted.
# Also update the "consistency" dictionary that keeps track of whether all
# predictions for a given sentence were correct.
num_correct = 0.
consistency_dict = {}

for example in labeled_examples:
    if not example["identifier"].split("-")[0] in consistency_dict:
        consistency_dict[example["identifier"].split("-")[0]] = True
    lookup = example["identifier"]
    prediction = predictions[lookup]
    if prediction.lower() == example["label"].lower():
        num_correct += 1.
    else:
        consistency_dict[example["identifier"].split("-")[0]] = False

# Calculate consistency.
num_consistent = 0.
unique_sentence = len(consistency_dict)
for identifier, consistent in consistency_dict.items():
    if consistent:
        num_consistent += 1

# Report values.
print("precision=" + str(num_correct / total_num))
print("consistency=" + str(num_consistent / unique_sentence))