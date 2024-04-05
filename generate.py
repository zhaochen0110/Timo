from vllm import LLM, SamplingParams
from config import load_datasets, score, extract_result
import json
import argparse
import pdb
import csv
from data_cleaning import data_clean
from classify_raw_data import classified_data
import re
import ray
ray.init()

def generate_answers(prompts, llm, sampling_params, num_answers=5):
    all_prompts = [prompt for prompt in prompts for _ in range(num_answers)]

    all_outputs = llm.generate(all_prompts, sampling_params)
    return all_outputs

def label_answers(answers, prompts, all_outputs, num_answers=5):
    results = []
    for i, entry in enumerate(answers):
        correct_answer = entry
        correct_response = None
        incorrect_response = None

        for j in range(num_answers):
            generated_answer = all_outputs[i * num_answers + j].outputs[0].text
            is_correct = extract_result(generated_answer, correct_answer)

            if is_correct and correct_response is None:
                correct_response = generated_answer
            elif not is_correct and incorrect_response is None:
                incorrect_response = generated_answer

            if correct_response is not None and incorrect_response is not None:
                break

        if correct_response is not None and incorrect_response is not None:
            results.append((prompts[i], correct_response, incorrect_response))

    return results

def all_answer(answers, prompts, all_outputs, num_answers=5):
    results = []
    for i, entry in enumerate(answers):
        generated_lst = []

        for j in range(num_answers):
            generated_answer = all_outputs[i * num_answers + j].outputs[0].text
            generated_lst.append(generated_answer)
        
        results.append((prompts[i], generated_lst))

    return results

def find_last_number(s):
    if 'Score:' in s:
        s = s.split('Score:')[1]
    if 'points' in s:
        s = s.split('points')[0]
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else None

DEFAULT_LLM_AS_CHOSEN_PROMPT = '''Use the following 5-point scoring system to review the user's question and the corresponding correct answer:
- Add 1 point if the response is relevant to the user's inquiry and shows some temporal reasoning ability, even if it is incomplete or contains irrelevant content.
- Add another point if the response correctly understands the temporal aspects of the user's question but doesn't fully resolve the query or provide a direct answer.
- Award a third point if the response accurately uses internal temporal knowledge to effectively answer the basic elements of the user's question.
- Grant a fourth point if the response not only effectively uses temporal reasoning but is also clearly written from an AI Assistant’s perspective, addressing the user's question directly and comprehensively, and is well-organized.
- Bestow a fifth point for a response that excellently applies temporal reasoning, reflecting expert knowledge in time-based queries, and demonstrates a high-quality, engaging, and insightful answer.
User: The letterhead on correspondence still bears the Bexar County Legal Aid name, even though the organization is no longer. When was the letterhead printed? Is "last second" possible?
 (A) TRUE
 (B) Undetermined
 (C) FALSE
<response> The statement "the organization is no longer" indicates that Bexar County Legal Aid is extinct. Therefore, the letterhead with the name "Bexar County Legal Aid" cannot be printed once the organization is no longer in existence. The answer is (C)</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.
The response is relevant to the user's inquiry as it addresses the question about the letterhead of an extinct organization. However, it does not show temporal reasoning ability regarding when the letterhead was printed: +1
The response does not correctly understand the temporal aspects of the user's question. It does not address the timing of the letterhead's printing or the possibility of "last second" printing: +0
The response does not use internal temporal knowledge to answer the basic elements of the user's question regarding the timing of the printing: +0
The response is from an AI Assistant’s perspective but is not comprehensive or well-organized in addressing the user's question about the timing of the letterhead's printing: +0
The response does not apply temporal reasoning, nor does it reflect expert knowledge in time-based queries. It does not provide a high-quality, engaging, or insightful answer regarding the timing of the printing: +0
Score: 1+0+0+0+0 = 1 points
User: Which event lasted the longest: Rescue of the Chilean miners, Korean War, or Gulf War?
 (A) Rescue of the Chilean miners
 (B) Korean War
 (C) Gulf War
<response> The duration of the Korean War (1950-1953) was more than three times longer than the Gulf War (1990-1991) and the Rescue of the Chilean miners (2010), which lasted 10 and 6 days, respectively. Therefore, the answer is (B)</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.
The response is relevant to the user's inquiry and shows temporal reasoning by comparing the durations of the events: +1
It correctly understands the temporal aspects of the user's question by identifying the specific time frames of each event: +1
The response uses internal temporal knowledge effectively to answer the basic elements of the user's question, identifying which event lasted the longest: +1
The response is clearly written from an AI Assistant’s perspective, directly addressing the user's question comprehensively and is well-organized: +1
The response applies temporal reasoning well, reflecting knowledge in time-based queries, but it may not reach the level of high-quality, engaging, and insightful answer that would merit the fifth point: +0
Score: 1+1+1+1+0 = 4 points
User: According to the Premise: Two women standing outdoors on a winter day in the countryside. and the Hypothesis: The women are all getting off the plane, having just arrived for their vacation in Hawaii. answer the question: Do the hypothesis and premise logically entail each other, contradict each other, or remain undetermined?
 (A) entailment
 (B) neutral
 (C) contradiction
<response> To assess the logical relationship between the premise and hypothesis, we examine whether the hypothesis is a logical consequence of the given premise. In this case, the premise describes two women standing outside on a winter day, while the hypothesis suggests that the women have just arrived for their vacation in Hawaii. This is a significant departure from the information provided in the premise, so the hypothesis does not logically follow from the premise. Therefore, the hypothesis and premise do not logically entail each other, but rather contradict each other, leading to a contradiction relationship. The answer is (C)</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.
Relevance and Temporal Reasoning: The response is relevant to the user's inquiry, addressing the logical relationship between the premise and hypothesis. It shows temporal reasoning by understanding that the premise and hypothesis describe two different scenarios in time: +1
Understanding Temporal Aspects: The response correctly understands that the premise and hypothesis describe different temporal contexts (winter day vs. arriving in Hawaii). It doesn't resolve the query in terms of temporal logic but does address the contradiction: +1
Use of Internal Temporal Knowledge: The response uses the information given in the premise and hypothesis to determine that they are contradictory. It doesn't require internal temporal knowledge beyond what is presented: +1
Effectiveness and Organization: The response is clear, direct, and well-organized. It effectively uses reasoning to explain why the premise and hypothesis contradict each other: +1
Expert Temporal Reasoning: The response demonstrates a good understanding of the logical relationship between the premise and hypothesis. It might not reflect "expert" knowledge since the scenario is straightforward, but it is high-quality and insightful within the context: +1
Score: 1+1+1+1+1 = 5 points
User: {{ prompt }}
<response>{{ response }}</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.'''


DEFAULT_LLM_AS_REJECT_PROMPT = '''Use the following 5-point scoring system to review the user's question and the corresponding incorrect answer that attempts to use temporal knowledge but fails to correctly solve the problem:
- Add 1 point if the response attempts to be relevant to the user’s inquiry and shows an attempt at temporal reasoning, even if the information is incorrect.
- Add another point if the response partially correctly attempts to handle the temporal aspects of the user's question but includes errors or misconceptions.
- Award a third point if the response clearly tries to use temporal knowledge but fails to accurately address the basic elements of the user’s question.
- Grant a fourth point if the response, while attempting to reason temporally from an AI Assistant’s perspective, shows some logic but contains errors or misses key information.
- Bestow a fifth point for those answers that make an effort in temporal reasoning but are incorrect, potentially misleading the user but not completely deviating from the topic of the question.
User: According to the Premise: Two women standing outdoors on a winter day in the countryside. and the Hypothesis: The women are all getting off the plane, having just arrived for their vacation in Hawaii. answer the question: Do the hypothesis and premise logically entail each other, contradict each other, or remain undetermined?
 (A) entailment
 (B) neutral
 (C) contradiction
<response> To analyze the relationship between the premise and hypothesis, we observe that the premise describes two women standing outside on a winter day. The hypothesis suggests that the women are all getting off a plane, having just arrived for their vacation in Hawaii. While this could be the case, the premise doesn't provide enough information to confirm the hypothesis. Therefore, the relationship between the premise and hypothesis remains undetermined. The answer is (B)</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.
The response attempts to be relevant to the user’s inquiry and shows an attempt at temporal reasoning, even if the information is incorrect: +1
The response does not partially correctly attempt to handle the temporal aspects of the user's question; it incorrectly concludes that the relationship is undetermined rather than contradictory: +0
The response tries to use temporal knowledge by discussing the premise and hypothesis but fails to accurately address the basic elements of the user’s question, which are the seasonal and location discrepancies: +1
The response, while attempting to reason temporally, shows some logic in trying to connect the premise and hypothesis but contains a critical error by not recognizing the clear contradiction: +1
The response makes an effort in temporal reasoning but is incorrect and potentially misleading by suggesting a neutral relationship instead of the correct contradictory one: +1
Score: 1+0+1+1+1 = 4 points
User: South Korean teenager Park Tae-Hwan took aim at a second Asian Games gold medal here Tuesday in the men's 400m freestyle, setting up a final clash with his biggest rival Zhang Lin of China. DOHA, Dec 5 , 2006, 2006. What is the relationship between the event 'took' and the time 'Dec 5 , 2006'?
 (A) BEGINS
 (B) SIMULTANEOUS
 (C) BEFORE
<response> Analyzing the events, we find that the action of taking aim at a gold medal is simultaneous with the time 'Dec 5 , 2006' in the statement. The relationship is SIMULTANEOUS. The answer is (B)</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.
The response attempts to be relevant to the user’s inquiry and shows an attempt at temporal reasoning, even if the information is incorrect: +1
The response partially correctly attempts to handle the temporal aspects of the user's question but includes errors or misconceptions: +1
The response clearly tries to use temporal knowledge but fails to accurately address the basic elements of the user’s question: +1
The response, while attempting to reason temporally from an AI Assistant’s perspective, shows some logic but contains errors or misses key information: +1
The response makes an effort in temporal reasoning but is incorrect, potentially misleading the user but not completely deviating from the topic of the question: +1
Score: 1+1+1+1+1 = 5 points
User: A festival is being organized 'a year prior'. When would that be?
 (A) the night before
 (B) dusk
 (C) the last year 
<response> The phrase 'a year prior' indicates that the festival is being planned or organized at least a year before the current date. This implies that the planned event will take place in the subsequent year. The answer is (A).</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.
The response attempts to be relevant to the user's inquiry and shows an attempt at temporal reasoning, even though it incorrectly identifies "a year prior" as "the night before.": +1
The response does not partially correctly handle the temporal aspects of the user's question. It incorrectly equates "a year prior" with "the night before," which is a significant error: +0
The response tries to use temporal knowledge by discussing the planning time in relation to the current date, but it fails to accurately address the basic element of the user's question, which is identifying when "a year prior" would be: +1
While the response attempts to reason temporally from an AI Assistant’s perspective, it does not show logical reasoning in this context because "the night before" is not logically connected to "a year prior.": +0
The answer makes an effort in temporal reasoning by trying to connect the phrase "a year prior" to a timeframe, but it is incorrect and potentially misleading. However, it does not completely deviate from the topic of the question: +1
Score: 1+0+1+0+1 = 3 points
User: {{ prompt }}
<response>{{ response }}</response>
After examining the user’s instruction and the response:
- Conclude with the score using the format: “Score: <total points>”.'''


def main(model_path, generate, train_data_path, is_score, generate_data_path, chunk_size, save_path):
    llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=8)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300)
    if generate:
        print('begin to generate!')
        file = open(train_data_path, 'r', encoding='utf-8')
        prompts = []
        answers = []
        for line in file.readlines():
            sentence = json.loads(line)
            prompts.append(sentence['input'])
            answers.append(sentence['answer'])

        prompts = prompts
        answers = answers

        all_outputs = []

        for i in range(0, len(prompts), chunk_size):
            part_prompts = prompts[i:i+chunk_size]
            part_outputs = generate_answers(part_prompts, llm, sampling_params)
            
            all_outputs.extend(part_outputs)

        all_results = all_answer(answers, prompts, all_outputs)

        output_data = []

        for prompt, generated_lst in all_results:
            prompt = "### Instruction:" + prompt.split("### Instruction:")[-1]
            output_data.append({
                'prompt': prompt,
                'generated_lst': generated_lst
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

            for response in positive:
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

        for data in data_info:
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
    parser.add_argument('--generate_data_path', type=str, metavar='N',
                        default='', help='you can also use the data already generated to score')
    parser.add_argument('--save_path', type=str, metavar='N',
                        default='', help='whether need to use reward to score the data')
    args = parser.parse_args()
    print(args)
    model_path = args.model_path
    generate = args.generate
    train_data_path = args.train_data_path
    is_score = args.score
    generate_data_path = args.generate_data_path
    save_path = args.save_path
    chunk_size = 50000
    main(model_path, generate, train_data_path, is_score, generate_data_path, chunk_size, save_path)
