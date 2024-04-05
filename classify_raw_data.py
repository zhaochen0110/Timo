import json
import pandas as pd
import os
from tqdm import tqdm
import re

dic = {
    'ambiguity_resolution/Interpretation.csv': 'ambiguity_resolution_interpretation',
    'ambiguity_resolution/Shift - Calendar.csv': 'ambiguity_resolution_shift_calendar',
    'ambiguity_resolution/Shift - LT.csv': 'ambiguity_resolution_shift_lt',
    'ambiguity_resolution/Shift - MT.csv': 'ambiguity_resolution_shift_mt',
    'ambiguity_resolution/Shift - ST.csv': 'ambiguity_resolution_shift_st',
    'arithmetic/Application.csv': 'arithmetic_application',
    'arithmetic/Date Computation.csv': 'arithmetic_date_computation',
    'arithmetic/Hour Adjustment (12h).csv': 'arithmetic_hour_adjustment(12h)',
    'arithmetic/Hour Adjustment (24h).csv': 'arithmetic_hour_adjustment (24h)',
    'arithmetic/Month Shift.csv': 'arithmetic_month_shift',
    'arithmetic/Time Computation.csv': 'arithmetic_time_computation',
    'arithmetic/Time Zone Conversion.csv': 'arithmetic_time_zone_conversion',
    'arithmetic/Week Identification.csv': 'arithmetic_week_identification',
    'arithmetic/Year Shift.csv': 'arithmetic_year_shift',
    'causality/Cause.csv': 'causality_cause',
    'causality/Effect.csv': 'causality_effect',
    'duration/Analogy Inference.csv': 'duration_analogy_inference',
    'duration/Commonsense.csv': 'duration_commonsense',
    'duration/Computation.csv': 'duration_computation',
    'duration/Direct Comparison.csv': 'duration_direct_comparison',
    'duration/Facts.csv': 'duration_facts',
    'duration/Multi-Step Comparison.csv': 'duration_multi-step_comparison',
    'duration/Reading Comprehension.csv': 'duration_reading_comprehension',
    'frequency/Application.csv': 'frequency_application',
    'frequency/Commonsense.csv': 'frequency_commonsense',
    'frequency/Comparison.csv': 'frequency_comparison',
    'frequency/Computation.csv': 'frequency_computation',
    'frequency/Facts.csv': 'frequency_facts',
    'frequency/Reading Comprehension.csv': 'frequency_reading_comprehension',
    'nli/nli.csv':'nli',
    'ordering/Commonsense.csv': 'ordering_commonsense',
    'ordering/Facts.csv': 'ordering_facts',
    'relation/relation.csv': 'relation',
    'storytelling/storytelling.csv': 'storytelling',
    'typical_time/Commonsense.csv': 'typical_time_commonsense',
    'typical_time/Comparison.csv':'typical_time_comparsion',
    'typical_time/Facts.csv': 'typical_time_facts',
    'typical_time/Reading Comprehension.csv': 'typical_time_reading_comprehension'
}
raw_data = []
train_data_folder = 'train_data'
for root in ['ambiguity_resolution', 'arithmetic', 'causality', 'duration', 'frequency', 
             'nli', 'ordering', 'relation', 'storytelling', 'typical_time']:
    if root == 'ambiguity_resolution':
        file = ['Interpretation.csv', 'Shift - Calendar.csv', 'Shift - LT.csv', 'Shift - MT.csv', 'Shift - ST.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Answer'],task))

    elif root == 'arithmetic':
        file = ['Application.csv', 'Date Computation.csv', 'Hour Adjustment (12h).csv', 'Hour Adjustment (24h).csv', 'Month Shift.csv',
                'Time Computation.csv', 'Time Zone Conversion.csv', 'Week Identification.csv', 'Year Shift.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Option D'],row['Answer'],task))

    elif root == 'causality':
        file = ['Cause.csv', 'Effect.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Premise'],row['Question'],row['Option A'],row['Option B'],row['Answer'],task))
        
    elif root == 'duration':
        file = ['Analogy Inference.csv', 'Commonsense.csv', 'Computation.csv', 'Direct Comparison.csv', 'Facts.csv', 'Multi-Step Comparison.csv', 'Reading Comprehension.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Answer'],task))

    elif root == 'frequency':
        file = ['Application.csv', 'Commonsense.csv', 'Comparison.csv', 'Computation.csv', 'Facts.csv', 'Reading Comprehension.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Answer'],task))

    elif root == 'nli':
        file = ['nli.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Premise'],row['Hypothesis'],row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Answer'],task))

    elif root == 'ordering':
        file = ['Commonsense.csv', 'Facts.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Answer'],task))

    elif root == 'relation':
        file = ['relation.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Answer'],task))

    elif root == 'storytelling':
        file = ['storytelling.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Story'],row['Question'],row['Option A'],row['Option B'],row['Answer'],task))

    elif root == 'typical_time':
        file = ['Commonsense.csv', 'Comparison.csv', 'Facts.csv', 'Reading Comprehension.csv']
        for file_path in file:
            task = dic[root+'/'+file_path]
            all_data = pd.read_csv(train_data_folder+'/'+root+'/'+file_path)
            for index, row in all_data.iterrows():
                raw_data.append((row['Question'],row['Option A'],row['Option B'],row['Option C'],row['Answer'],task))

def classified_data(datas):
    output = []
    for data in tqdm(datas):
        flag = True
        for origin_data in raw_data:
            # print(origin_data)
            flag1 = True
            for i in range(len(origin_data)-2):
                if str(origin_data[i]) not in data['prompt']:
                    flag1 = False
                    break
            if flag1:
                flag = False
                output.append(
                    {
                    'prompt':data['prompt'],
                    'label':origin_data[-2],
                    'generated_lst':data['generated_lst'],
                    'task':origin_data[-1]   
                    }           
                )
                break

    classified_data = []
    for data in output:
        positive = []
        negative = []
        label = data['label']
        generated_lst = data['generated_lst']
        for generated_data in generated_lst:
            is_match = False
            if 'the answer is' in generated_data.lower():
                match = re.search(r'the answer is \((.*?)\)', generated_data.lower())
                if match:
                    is_match = True
            elif 'the correct answer is' in generated_data.lower():
                match = re.search(r'the correct answer is \((.*?)\)', generated_data.lower())
                if match:
                    is_match = True 
            if is_match:
                if match.group(1) == label.lower():
                    positive.append(generated_data)
                elif match.group(1) in ['a','b','c','d']:
                    negative.append(generated_data)
        classified_data.append({
            'source':data['task'],
            'prompt':data['prompt'],
            'positive':positive,
            'negative':negative,
            'label':data['label']
        })

    return classified_data
