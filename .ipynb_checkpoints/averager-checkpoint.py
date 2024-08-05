import os
import re,statistics
import json

"""
This file is used to calculate the overall accuracies of the model
"""

def write_dict_to_txt(dictionary, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(dictionary, file, indent=4)
        print(f"Dictionary successfully written to '{file_path}'.")
    except Exception as e:
        print(f"Error writing dictionary to '{file_path}': {e}")

def get_accuracies():
    path = "./results_dominance/DEAP/"

    models = ["facebio/"]

    for model in models:
        subjects_accuracies = {}
        for i in os.listdir(path + model):
            sub_paths = path + model + i
            k_fold_accuracies = []
            for j in os.listdir(sub_paths):
                phrase = "best accuracy is "
                with open(sub_paths + "/" + j + "/" + j + '.txt') as file:
                    for line in file:
                        # Regular expression pattern
                        pattern = r"best accuracy is (\d+(\.\d+)?) in epoch (\d+)"

                        match = re.search(pattern, line)

                        if match:
                            value = float(match.group(1))
                            k_fold_accuracies.append(value)
            subject = i.replace("s", "")
            subjects_accuracies[int(subject)] = statistics.mean(k_fold_accuracies)

        subjects_accuracies = dict(sorted(subjects_accuracies.items()))

        subjects_accuracies["accuracy by subjects average"] = statistics.mean(subjects_accuracies.values())

        accuracy_file = f"accuracies_{model.replace('/','')}.txt"

        write_dict_to_txt(subjects_accuracies, path + model + accuracy_file)

get_accuracies()


