import re

def classified_data(datas):
    """Classify the generated response as either positive or negative."""
    classified_data = []
    
    for data in datas:
        positive = []
        negative = []
        label = data['label'].lower()
        generated_lst = data['generated_lst']
        
        for generated_data in generated_lst:
            is_match = False
            generated_data_lower = generated_data.lower()
            
            if 'the answer is' in generated_data_lower:
                match = re.search(r'the answer is \((.*?)\)', generated_data_lower)
                is_match = bool(match)
            elif 'the correct answer is' in generated_data_lower:
                match = re.search(r'the correct answer is \((.*?)\)', generated_data_lower)
                is_match = bool(match)
                
            if is_match:
                matched_answer = match.group(1)
                if matched_answer == label:
                    positive.append(generated_data)
                elif matched_answer in ['a', 'b', 'c', 'd']:
                    negative.append(generated_data)
        
        classified_data.append({
            'source': data['task'],
            'prompt': data['prompt'],
            'positive': positive,
            'negative': negative,
            'label': data['label']
        })
    
    return classified_data