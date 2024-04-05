import pandas as pd
from datasets import Dataset
import pdb


def load_datasets(path: str):
    dataset = []
    part = path.split('/')[-2]
    if part in ['ambiguity_resolution', 'duration', 'frequency', 'nli', 'ordering', 'relation', 'typical_time']:
        mcq = 3
    if part in ['arithmetic']:
        mcq = 4
    if part in ['causality', 'storytelling']:
        mcq = 2
    
    # pdb.set_trace()
    questions = pd.read_csv(path)
    
    
    for index,row in questions.iterrows():
        question = row['Question']
        gound_truth = row['Answer']
        op_A = row['Option A']
        op_B = row['Option B']
        if mcq == 3:
            op_C = row['Option C']
            if 'nli' in path:
                Hypothesis = row['Hypothesis']
                Premise = row['Premise']
                dataset.append({'question': question, 'OptionA': op_A, 'OptionB': op_B, 'OptionC': op_C, 'answer': gound_truth, 'Premise': Premise, 'Hypothesis': Hypothesis})
            else:
                dataset.append({'question': question, 'OptionA': op_A, 'OptionB': op_B, 'OptionC': op_C, 'answer': gound_truth})
        elif mcq == 4:
            op_C = row['Option C']
            op_D = row['Option D']
            dataset.append({'question': question, 'OptionA': op_A, 'OptionB': op_B, 'OptionC': op_C, 'OptionD': op_D, 'answer': gound_truth})
        else:
            if 'causality' in path:
                Premise = row['Premise']
                dataset.append({'question': question, 'OptionA': op_A, 'OptionB': op_B, 'answer': gound_truth, 'Premise': Premise})
            elif 'storytelling' in path:
                Story = row['Story']
                dataset.append({'question': question, 'OptionA': op_A, 'OptionB': op_B, 'answer': gound_truth, 'Story': Story})
            else:
                dataset.append({'question': question, 'OptionA': op_A, 'OptionB': op_B, 'answer': gound_truth})


    dataset = Dataset.from_list(dataset)

    return dataset


def score(predictions, references):
    if len(predictions) != len(references):
        return {
            'error': 'predictions and references have different '
            'length'
        }
    
    acc_total = 0
    count = 0      



    for num in range(len(predictions)):
        count += 1
        ground_truth = references[num]
        answer = predictions[num]
        predict = []
        if 'answer is' in answer:
            ans = answer.split('answer is')[1]
            if '[' in ans and ']' in ans:
                first_open_bracket = ans.find('[')
                first_close_bracket = ans.find(']')
                pred = ans[first_open_bracket:first_close_bracket+1]
            elif '(' in ans and ')' in ans:
                first_open_parenthesis = ans.find('(')
                first_close_parenthesis = ans.find(')')
                pred = ans[first_open_parenthesis:first_close_parenthesis+1]
            else:
                # In case no brackets or parentheses are found
                pred = ans.strip()
        else:
            pred = ''

        for i in pred:
            if i in ['A','B','C','D']:
                predict.append(i)   
                    
        if len(predict)==1 and ground_truth in predict:
            acc_total += 1
    

    return {'acc': acc_total/count}



def extract_result(answer, ground_truth):
    predict = []
    if 'answer is' in answer:
        ans = answer.split('answer is')[1]
        if '[' in ans and ']' in ans:
            first_open_parenthesis = ans.find('[')
            first_close_parenthesis = ans.find(']')
        else:
            first_open_parenthesis = ans.find('(')
            first_close_parenthesis = ans.find(')')              
        pred = ans[first_open_parenthesis:first_close_parenthesis+1]
    else:
        pred = ''

    for i in pred:
        if i in ['A','B','C','D']:
            predict.append(i) 
                    
    if len(predict)==1 and ground_truth in predict:
        return True
    else:
        return False
