import json
import pandas as pd
import argparse
from config import score
from vllm import LLM, SamplingParams

def evaluate_model(model_name, data_path, excel_folder, output_path):
    """
    Evaluate the model's comprehensive temporal reasoning ability using our test dataset
    
    Parameters:
	model_name (str): The path to the language model to be evaluated. 
	data_path (str): The path to the test dataset file. 
	excel_folder (str): The folder where the Excel file containing the evaluation results will be saved.
	output_path (str): The path where the generated outputs will be saved in JSON format.
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as file:
        prompts, answers, tasks = [], [], []
        for line in file.readlines():
            sentence = json.loads(line)
            prompts.append(sentence['input'])
            answers.append(sentence['answer'])
            tasks.append(sentence['task'])

    # Define the inference model
    # , tensor_parallel_size=8
    llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature=0, max_tokens=300)

    # Generate answers
    outputs = llm.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    # Save the generated outputs in their original format
    output_data = []
    for prompt, output, answer, task in zip(prompts, outputs, answers, tasks):
        prompt = "### Instruction:" + prompt.split("### Instruction:")[-1]
        output_data.append({'input': prompt, 'output': output, 'answer': answer, 'task': task})
    
    file_name = model_name.split('/')[-1]
    output_path = output_path + '/' + file_name + '.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    # Calculate overall accuracy
    overall_performance = score(outputs, answers)
    overall_accuracy = overall_performance['acc']

    # Calculate accuracy for each task
    task_accuracies = {}
    for task in set(tasks):
        task_indices = [i for i, x in enumerate(tasks) if x == task]
        task_outputs = [outputs[i] for i in task_indices]
        task_answers = [answers[i] for i in task_indices]
        task_performance = score(task_outputs, task_answers)
        task_accuracies[task] = task_performance['acc']

    # Sort the tasks alphabetically
    sorted_tasks = sorted(task_accuracies.keys())

    # Prepare data for Excel
    data_for_excel = {
        "Task": ["Overall"] + sorted_tasks,
        "Accuracy": [overall_accuracy] + [task_accuracies[task] for task in sorted_tasks]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data_for_excel)

    # Save to Excel file
    excel_path = f"{excel_folder}/{model_name.split('/')[-1]}.xlsx"
    df.to_excel(excel_path, index=False)



    print(f"Accuracies for {model_name.split('/')[-1]} saved to '{excel_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM model")
    parser.add_argument("model_name", type=str, help="Path to the model")
    parser.add_argument("data_path", type=str, help="Path to the dataset file")
    parser.add_argument("excel_folder", type=str, help="Folder to save Excel results")
    parser.add_argument("output_path", type=str, help="Path to save the generated outputs")
    
    args = parser.parse_args()

    evaluate_model(args.model_name, args.data_path, args.excel_folder, args.output_path)