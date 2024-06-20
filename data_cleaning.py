import re


def clean_generated_data(generated_data):
    """Clean the raw generated responses by processing each entry to remove invalid content and extract valid answers"""
    if '\n\n' in generated_data:
        generated_data = generated_data.split('\n\n')[0]

    generated_data_lower = generated_data.lower()
    
    # Skip if the phrase "the answer is" or "the correct answer is" appears more than once
    if generated_data_lower.count('the answer is') > 1 or generated_data_lower.count('the correct answer is') > 1:
        return None

    generated_data_lst = generated_data.split('\n')
    last_sentence = generated_data_lst[-1].lower()

    # Check for the presence of the answer phrases in the last sentence
    if 'the answer is' not in last_sentence and 'the correct answer is' not in last_sentence:
        flag = False
        is_overgeneration = False
        for sentence in generated_data_lst:
            sentence_lower = sentence.lower()
            if 'the answer is' in sentence_lower or 'the correct answer is' in sentence_lower:
                flag = True
            if '### instruction' in sentence_lower and flag:
                is_overgeneration = True

        # If overgeneration is detected, truncate the sentences before the new instruction
        if is_overgeneration:
            sentences = []
            for sentence in generated_data_lst:
                if '### instruction' in sentence.lower():
                    break
                sentences.append(sentence)
            generated_data = '\n'.join(sentences)

    match = re.search(r'the answer is \((.*?)\)', generated_data.lower()) or \
            re.search(r'the correct answer is \((.*?)\)', generated_data.lower())

    return generated_data if match else None


def data_clean(raw_data):
    """
    iterates through a list of prompts and their generated responses, 
    cleaning and retaining only valid entries, and then returns the cleaned data
    """
    clean_data = []
    for data in raw_data:
        prompt = data['prompt'].replace('### Instruction: ', '').replace('\n ### Response:', '')
        
        # Clean and filter the generated list using the clean_generated_data function
        new_generate_lst = [
            clean_generated_data(generated_data)
            for generated_data in data['generated_lst']
            if clean_generated_data(generated_data)
        ]

        if new_generate_lst:
            clean_data.append({'prompt': prompt, 'generated_lst': new_generate_lst, 'label': data['label'], 'task': data['task']})
    
    return clean_data