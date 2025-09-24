
case_num = range(161, 201)

case_list = [f"Case{i}.nii.gz" for i in case_num]

with open("test_list.txt", "w") as f:
    for case in case_list:
        f.write(case + "\n")