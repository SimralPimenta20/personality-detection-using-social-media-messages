import pandas as pd

factors = ["agr", "neutral", "open", "ext", "cn"]

for factor_name in factors:
    file_name = factor_name + ".csv"
    data = pd.read_csv(file_name, sep = ",")
    data.columns = ["label","body_text"]

    yes = 0
    no = 0

    for x in list(data["label"]):
        if x == "y":
            yes = yes + 1
        else:
            no = no + 1

    print(factor_name, "Yes:",yes,"No:",no)

    #now we will create new data frames
    #Shuffle the Dataset.
    shuffled_df = data.sample(frac=1,random_state=4)

    #Find the majority label and minority label
    if yes>no:
        majority_label = "y"
        majority = yes
        minority_label = "n"
        minority = no
    else:
        majority_label = "n"
        majority = no
        minority_label = "y"
        minority = yes
        
    #Put all the minority class in a separate dataset.
    minority_df = shuffled_df.loc[shuffled_df['label'] == minority_label]

    #Randomly select minority number observations from the majority class
    majority_df = shuffled_df.loc[shuffled_df['label'] == majority_label].sample(n=minority,random_state=42)

    #Concatenate both dataframes again
    normalized_df = pd.concat([minority_df,majority_df])

    #save them in csv file
    new_file_name = factor_name + "_balanced.csv"
    normalized_df.to_csv(new_file_name, index = False, header = True)

    #Check whether they are balanced
    data = pd.read_csv(new_file_name, sep = ",")
    
    yes = 0
    no = 0

    for x in list(data["label"]):
        if x == "y":
            yes = yes + 1
        else:
            no = no + 1

    print(factor_name, "Yes:",yes,"No:",no)
