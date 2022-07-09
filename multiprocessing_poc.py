from func.ultis import load_obj

Ovules_data_dict = load_obj("dataset_info/Ovules_dataset_info")
print(len(Ovules_data_dict["train"].keys()))
count = 0
for o in Ovules_data_dict["train"].keys():
    if (count % 2) == 0:
        print("------")
        print(count/2 + 1)
    print("\"" + o + "\",")

    count += 1