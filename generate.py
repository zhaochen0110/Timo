from vllm import LLM, SamplingParams
import json
import argparse
import pdb
import csv
from data_cleaning import data_clean
from classify_raw_data import classified_data
import re
from config import *
import ray
ray.init()

def generate_answers(prompts, llm, sampling_params, num_answers=5):
    """Generate responses using mathematical models (MathLLM)"""
    all_prompts = [prompt for prompt in prompts for _ in range(num_answers)]

    all_outputs = llm.generate(all_prompts, sampling_params)
    return all_outputs


def all_answer(answers, prompts, all_outputs, tasks, num_answers=5):
    """
    Retrieve and organize generated responses for each question.
    processes the provided data to group generated responses for each question based on the given prompts and corresponding answers. 
    """
    results = []
    for i, entry in enumerate(answers):
        generated_lst = []

        for j in range(num_answers):
            generated_answer = all_outputs[i * num_answers + j].outputs[0].text
            generated_lst.append(generated_answer)
        
        results.append((prompts[i], generated_lst, entry, tasks[i]))

    return results

def find_last_number(s):
    if 'Score:' in s:
        s = s.split('Score:')[1]
    if 'points' in s:
        s = s.split('points')[0]
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else None


def main(model_path, generate, train_data_path, is_score, chunk_size, save_path):
    """
    uses a language model to generate, clean, score, and select text data. 
    It reads input data, generates responses in chunks, and cleans and classifies the generated data if the `generate` flag is true. 
    If the `is_score` flag is true, it scores positive and negative prompts, 
    selects the best responses based on these scores, and saves the results. 
    The final output includes prompts with their best positive and negative responses, saved to a file for DPO training in next step.
    """
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=8)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300)
    if generate:
        print('begin to generate!')
        file = open(train_data_path, 'r', encoding='utf-8')
        prompts = []
        answers = []
        tasks = []
        for line in file.readlines():
            sentence = json.loads(line)
            prompts.append(sentence['input'])
            answers.append(sentence['answer'])
            tasks.append(sentence['task'])

        prompts = prompts
        answers = answers

        all_outputs = []

        for i in range(0, len(prompts), chunk_size):
            part_prompts = prompts[i:i+chunk_size]
            part_outputs = generate_answers(part_prompts, llm, sampling_params)
            
            all_outputs.extend(part_outputs)

        all_results = all_answer(answers, prompts, all_outputs, tasks)

        output_data = []

        for prompt, generated_lst, label, task in all_results:
            prompt = "### Instruction:" + prompt.split("### Instruction:")[-1]
            output_data.append({
                'prompt': prompt,
                'generated_lst': generated_lst,
                'label':label,
                'task':task
            })

        cleaned_data = data_clean(output_data)
        all_data = classified_data(cleaned_data)
    
        with open(save_path+'/generated_data.json', 'w', encoding='utf-8') as f:
            for data in all_data:
                json_data = json.dumps(data)
                f.write(json_data+'\n')
    else:
        if not generate:
            all_data = []
            print('read the generate file')
            with open(save_path+'/generated_data.json', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    all_data.append(data)
    
    if is_score: 
        positive_prompts = []
        negative_prompts = []
        data_info = []
        for data in all_data:
            prompt = data['prompt']
            task = data['source']
            positive = data['positive']
            negative = data['negative']
            label = data['label']
            positive_ind = len(positive_prompts)
            negative_ind = len(negative_prompts)
            if len(positive) == 0 or len(negative) == 0:
                continue

            for response in positive:  # self critic
                model_input = DEFAULT_LLM_AS_CHOSEN_PROMPT.replace("{{ prompt }}", prompt).replace("{{ response }}", response)
                positive_prompts.extend([model_input]*3)

            for response in negative:
                model_input = DEFAULT_LLM_AS_REJECT_PROMPT.replace("{{ prompt }}", prompt).replace("{{ response }}", response)
                negative_prompts.extend([model_input]*3)  
            
            data = {
                'prompt':prompt,
                'positive_index':positive_ind,
                'negative_index':negative_ind,
                'positive':positive,
                'negative':negative,
                'source':task,
                'label':label
            }
            data_info.append(data)

        positive_outputs = llm.generate(positive_prompts, sampling_params)
        positive_scores = []
        with open(save_path+'/positive_score.json', 'w', encoding='utf-8') as f:
            for output in positive_outputs:
                score = output.outputs[0].text
                data = {'score':score}
                positive_scores.append(data)
                json_data = json.dumps(data)
                f.write(json_data+'\n')


        
        with open(save_path+'/negative_score.json', 'w', encoding='utf-8') as f:
            print("begin to generate negative_score")
            negative_scores = []

            for i in range(0, len(negative_prompts), chunk_size):
                negative_prompts_chunk = negative_prompts[i:i+chunk_size]
                negative_outputs = llm.generate(negative_prompts_chunk, sampling_params)

                for output in negative_outputs:
                    score = output.outputs[0].text
                    data = {'score':score}
                    negative_scores.append(data)   
                    json_data = json.dumps(data)
                    f.write(json_data+'\n')

        with open(save_path+'/data_info.json', 'w', encoding='utf-8') as f:
            for data in data_info:
                json_data = json.dumps(data)
                f.write(json_data+'\n')


        final_data = []

        for data in data_info:     # choose the best positive as chosen and best negative as rejected
            positive = data['positive']
            positive_ind = data['positive_index']
            if len(positive) == 1:
                best_positive = positive[0]
            else:
                best_index = 0
                best_score = -1
                for i in range(0, len(positive)*3, 3):
                    score1 = positive_scores[positive_ind+i]['score']
                    score2 = positive_scores[positive_ind+i+1]['score']
                    score3 = positive_scores[positive_ind+i+2]['score']
                    all_scores = []
                    for j in [score1, score2, score3]:
                        if j == None:
                            continue
                        else:
                            if find_last_number(j):
                                response_score = int(find_last_number(j))
                                if 0 <= response_score <= 5:
                                    all_scores.append(response_score)
                    if len(all_scores) > 0:
                        mean_score = sum(all_scores)/len(all_scores)
                    else:
                        mean_score = -1
                    if mean_score>best_score:
                        best_index = i//3
                        best_score = mean_score

                best_positive = positive[best_index]

            negative = data['negative']
            negative_ind = data['negative_index']
            if len(negative) == 1:
                best_negative = negative[0]
            else:
                best_index = 0
                best_score = -1
                for i in range(0, len(negative)*3, 3):
                    score1 = negative_scores[negative_ind+i]['score']
                    score2 = negative_scores[negative_ind+i+1]['score']
                    score3 = negative_scores[negative_ind+i+2]['score']
                    all_scores = []
                    for j in [score1, score2, score3]:
                        if j == None:
                            continue
                        else:
                            if find_last_number(j):
                                response_score = int(find_last_number(j))
                                if 0 <= response_score <= 5:
                                    all_scores.append(response_score)
                    if len(all_scores) > 0:
                        mean_score = sum(all_scores)/len(all_scores)
                    else:
                        mean_score = -1
                    if mean_score>best_score:
                        best_index = i//3
                        best_score = mean_score
                
                best_negative = negative[best_index]
            data = {
                'source':data['source'],
                'prompt':data['prompt'],
                'chosen':best_positive,
                'rejected':best_negative
            }
            final_data.append(data)

        with open(save_path+'/selected_data.json', 'w', encoding='utf-8') as f:
            for data in final_data:
                json_data = json.dumps(data)
                f.write(json_data+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model_path', type=str, metavar='N',
                        default='', help='use which model as generate model and reward model')
    parser.add_argument('--generate', type=bool, metavar='N',
                        default=False, help='whether need to generate data')
    parser.add_argument('--train_data_path', type=str, metavar='N',
                        default='', help='the data to generate response')
    parser.add_argument('--score', type=bool, metavar='N',
                        default=True, help='whether need to use reward to score the data')
    parser.add_argument('--save_path', type=str, metavar='N',
                        default='', help='whether need to use reward to score the data')
    args = parser.parse_args()
    print(args)
    model_path = args.model_path
    generate = args.generate
    train_data_path = args.train_data_path
    is_score = args.score
    save_path = args.save_path
    chunk_size = 50000
    main(model_path, generate, train_data_path, is_score, chunk_size, save_path)
