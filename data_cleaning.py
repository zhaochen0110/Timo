import json
import re

def data_clean(raw_data):
    clean_data =[]
    for data in raw_data:
        new_generate_lst = []
        generated_lst = data['generated_lst']
        prompt = data['prompt']
        prompt = prompt.replace('### Instruction: ','').replace('\n ### Response:','')
        for i in range(len(generated_lst)):
            generated_data = generated_lst[i]
            if '\n\n' in generated_data:
                generated_data = generated_data.split('\n\n')[0]
            generated_data_lower = generated_data.lower()
            if generated_data_lower.count('the answer is')>1 or generated_data_lower.count('the correct answer is')>1:
                continue
            generated_data_lst = generated_data.split('\n')
            last_sentence = generated_data_lst[-1].lower()
            is_match = False
            if 'the answer is' not in last_sentence and 'the correct answer is' not in last_sentence:
                flag = False
                is_overgeneration = False
                for sentence in generated_data_lst:
                    sentence = sentence.lower()
                    if 'the answer is' in sentence or 'the correct answer is' in sentence:
                        flag = True
                    if '### instruction' in sentence and flag:
                        is_overgeneration = True
                if is_overgeneration:
                    # print(1)
                    sentences = []
                    for sentence in generated_data_lst:
                        if '### instruction' in sentence.lower():
                            break
                        else:
                            sentences.append(sentence)
                    generated_data='\n'.join(sentences)
                    if 'the answer is' in generated_data.lower():
                        match = re.search(r'the answer is \((.*?)\)', generated_data.lower())
                        if match:
                            is_match = True
                    elif 'the correct answer is' in generated_data.lower():
                        match = re.search(r'the correct answer is \((.*?)\)', generated_data.lower())
                        if match:
                            is_match = True 
            else:
                if 'the answer is' in generated_data.lower():
                    match = re.search(r'the answer is \((.*?)\)', generated_data.lower())
                    if match:
                        is_match = True
                elif 'the correct answer is' in generated_data.lower():
                    match = re.search(r'the correct answer is \((.*?)\)', generated_data.lower())
                    if match:
                        is_match = True 
            if is_match:
                new_generate_lst.append(generated_data)
        if new_generate_lst!=[]:
            clean_data.append({'prompt':prompt,'generated_lst':new_generate_lst})
    return clean_data
