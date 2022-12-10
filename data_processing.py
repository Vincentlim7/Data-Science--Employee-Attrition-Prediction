import pandas as pd
from sklearn.model_selection import train_test_split


def oldOneHotEncoding(df):
    """First version of our One Hot Encoding code (before we when found pandas.get_dummies())

    Args:
        df (pandas dataframe): dataframe to one hot encode on

    Returns:
        pandas dataframe: the final dataframe
    """
    size = len(df)

    travel_frequently = []
    travel_rarely = []
    non_travel = []

    research = []
    sales = []
    human = []

    life_sciences = []
    medical = []
    marketing = []
    technical = []
    other = []

    sales_executive = []
    research_scientist = []
    laboratory_technician = []
    manufacturing_director = []
    healthcare_representative = []

    married = []
    single = []
    divorced = []

    for index in range(size):
        travel_status = df["BusinessTravel"][index]
        department_status = df["Department"][index]
        education_status = df["EducationField"][index]
        jobRole_status = df["JobRole"][index]
        marital_status = df["MaritalStatus"][index]
        
        if travel_status == "Travel_Frequently":
            travel_frequently.append(1)
            travel_rarely.append(0)
            non_travel.append(0)
        elif travel_status == "Travel_Rarely":
            travel_frequently.append(0)
            travel_rarely.append(1)
            non_travel.append(0)
        else:
            travel_frequently.append(0)
            travel_rarely.append(0)
            non_travel.append(1)
            
        if department_status == "Research & Development":
            research.append(1)
            sales.append(0)
            human.append(0)
        elif department_status == "Sales":
            research.append(0)
            sales.append(1)
            human.append(0)
        else:
            research.append(0)
            sales.append(0)
            human.append(1)
            
        if education_status == "Life Sciences":
            life_sciences.append(1)
            medical.append(0)
            marketing.append(0)
            technical.append(0)
            other.append(0)
        elif education_status == "Medical":
            life_sciences.append(0)
            medical.append(1)
            marketing.append(0)
            technical.append(0)
            other.append(0)
        elif education_status == "Marketing":
            life_sciences.append(0)
            medical.append(0)
            marketing.append(1)
            technical.append(0)
            other.append(0)
        elif education_status == "Technical Degree":
            life_sciences.append(0)
            medical.append(0)
            marketing.append(0)
            technical.append(1)
            other.append(0)
        else:
            life_sciences.append(0)
            medical.append(0)
            marketing.append(0)
            technical.append(0)
            other.append(1)
        
        if jobRole_status == "Sales Executive":
            sales_executive.append(1)
            research_scientist.append(0)
            laboratory_technician.append(0)
            manufacturing_director.append(0)
            healthcare_representative.append(0)
        elif jobRole_status == "Research Scientist":
            sales_executive.append(0)
            research_scientist.append(1)
            laboratory_technician.append(0)
            manufacturing_director.append(0)
            healthcare_representative.append(0)
        elif jobRole_status == "Laboratory Technician":
            sales_executive.append(0)
            research_scientist.append(0)
            laboratory_technician.append(1)
            manufacturing_director.append(0)
            healthcare_representative.append(0)
        elif jobRole_status == "Manufacturing Director":
            sales_executive.append(0)
            research_scientist.append(0)
            laboratory_technician.append(0)
            manufacturing_director.append(1)
            healthcare_representative.append(0)
        else:
            sales_executive.append(0)
            research_scientist.append(0)
            laboratory_technician.append(0)
            manufacturing_director.append(0)
            healthcare_representative.append(1)
            
        if marital_status == "Married":
            married.append(1)
            single.append(0)
            divorced.append(0)
        elif marital_status == "Single":
            married.append(0)
            single.append(1)
            divorced.append(0)
        else:
            married.append(0)
            single.append(0)
            divorced.append(1)
    
    return df
            
            

    business_index = df.columns.get_loc("BusinessTravel")
    df = df.drop("BusinessTravel", axis = 1)
    df.insert(business_index, "Non_Travel", non_travel)
    df.insert(business_index, "Travel_Rarely", travel_rarely)
    df.insert(business_index, "Travel_Frequently", travel_frequently)

    department_index = df.columns.get_loc("Department")
    df = df.drop("Department", axis = 1)
    df.insert(department_index, "Human Resources", human)
    df.insert(department_index, "Sales", sales)
    df.insert(department_index, "Research & Development", research)

    education_index = df.columns.get_loc("EducationField")
    df = df.drop("EducationField", axis = 1)
    df.insert(education_index, "EducationOther", other)
    df.insert(education_index, "Technical Degree", technical)
    df.insert(education_index, "Marketing", marketing)
    df.insert(education_index, "Medical", medical)
    df.insert(education_index, "Life Sciences", life_sciences)

    jobRole_index = df.columns.get_loc("JobRole")
    df = df.drop("JobRole", axis = 1)
    df.insert(jobRole_index, "Healthcare Representative", healthcare_representative)
    df.insert(jobRole_index, "Manufacturing Director", manufacturing_director)
    df.insert(jobRole_index, "Laboratory Technician", laboratory_technician)
    df.insert(jobRole_index, "Research Scientist", research_scientist)
    df.insert(jobRole_index, "Sales Executive", sales_executive)

    marital_index = df.columns.get_loc("MaritalStatus")
    df = df.drop("MaritalStatus", axis = 1)
    df.insert(marital_index, "Divorced", divorced)
    df.insert(marital_index, "Single", single)
    df.insert(marital_index, "Married", married)



if __name__ == "__main__":
    df = pd.read_csv('data/raw_data.csv')

    # Dropping columns
    df = df.drop(["EmployeeCount", "EmployeeNumber", "Over18", "RelationshipSatisfaction", "StandardHours"], axis = 1)

    # Formatting columns
    df['Attrition'] = df['Attrition'].replace(["Yes", "No"], [1, 0])
    df['Gender'] = df['Gender'].replace(["Male", "Female"], [0, 1])
    df['OverTime'] = df['OverTime'].replace(["Yes", "No"], [1, 0])

    # One-Hot Encoding
    cat_features = []
    for col, value in df.iteritems():
        if value.dtype == 'object':
            cat_features.append(col)
        
    num_features = df.columns.difference(cat_features)

    # Concat the two dataframes together columnwise
    df = pd.concat([df[num_features], pd.get_dummies(df[cat_features])], axis=1)

    df.to_csv("data/data.csv", index=False)

    # Generating train/test files
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)