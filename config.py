import pandas as pd
import pdb

def score(predictions, references):
    """Accuracy calculation"""
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
