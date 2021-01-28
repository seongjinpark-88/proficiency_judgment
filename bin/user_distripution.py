import os
from collections import defaultdict
import pandas as pd

print(os.getcwd())
data_location = "data/perception_results"

count_participant = defaultdict(int)
acc_counter = defaultdict(dict)
flu_counter = defaultdict(dict)
comp_counter = defaultdict(dict)

for file in os.listdir(data_location):
    filepath = os.path.join(data_location, file)

    data = pd.read_csv(filepath, sep=",")
    # print(data["participant_id"].unique())
    participant_number = len(data["participant_id"].unique())
    if "acc" in file:
        count_participant["acc"] += participant_number
    elif "flu" in file:
        count_participant["flu"] += participant_number
    elif "comp" in file:
        count_participant["comp"] += participant_number
print(count_participant)


#     for i in range(1, len(data)):
#         line = data[i].rstrip().split(",")
#         print(line)
#         participant_id = line[3]
#         response_type = line[7]
#         if response_type == "Scale_response":
#             if "acc" in file:
#                 try:
#                     acc_counter[participant_id] += 1
#                 except:
#                     acc_counter[participant_id] = 1
#             elif "flu" in file:
#                 try:
#                     flu_counter[participant_id] += 1
#                 except:
#                     flu_counter[participant_id] = 1
#             elif "comp" in file:
#                 try:
#                     comp_counter[participant_id] += 1
#                 except:
#                     comp_counter[participant_id] = 1
#
# print(acc_counter)