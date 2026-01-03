import re, sys
import nltk

import opencc
converter = opencc.OpenCC('s2tw.json')

'''
Multi-Choices - mc
'''
# Get the selected choice from the generated response
def extract_choice(response):
    patterns = [
        r"\b([A-Ea-e1-5])\b",
        r"\(\s*([A-Ea-e1-5])\s*\)\s*",
        r"correct answer is\s*[(|【|（|［|\[|<|＜]([A-Ea-e1-5])[)|】|）|］|\]|>|＞)]\s*",
        r"(?<=答案)\s*.+\s*\(([A-Ea-e1-5])\)",
        r"答案是:?\s*\(?([A-Ea-e1-5])\)?\s*",
        r"\(([A-Ea-e1-5])\).+正確",
        r"\[\/?INST\]\s*\(([A-Ea-e1-5])\)",
        r"ASSISTAs?NT:\s*\(([A-Ea-e1-5])\)",
        r"\s?[【|（|［|\[|<|＜]([A-Ea-e1-5])[】|）|］|\]|>|＞)]\s?",
        r"(?<=\w|\s)\(([A-Ea-e1-5])\)(?=\w|\s)",
        r"\(([A-Ea-e1-5])\)\s*[\u4E00-\u9FFF]",
        r"[\u4E00-\u9FFF]\s*\(([A-Ea-e1-5])\)",
        r"(?<=選項)\s*\(([A-Ea-e1-5])\)",
        r"([A-Ea-e1-5])\.\s*[\u4E00-\u9FFF]",
        r"([A-Ea-e1-5])\.\s*\w",
        r"([A-Ea-e1-5])\.?\s*[「|【|『|《|«]",
        r"\(\s*([A-Ea-e1-5])\s*\)",
        r"\(?([A-Ea-e1-5])\)\s*",
    ]

    # Remove unnecessary generated content (e.g., newline characters)
    # response = re.sub(r"\n", "", response)

    # Find generated results that match the pattern (the entire sentence can only contain one ABCD letter, more than one is not counted)
    for regex in patterns:
        list_ = re.findall(regex, response)
        if len(list_) > 0:
            return list_[0].upper()
    
    return ''

# Evaluation
def evaluate_mc(ground_truth_file, list_predicted_results, task_name):
    list_choices = "ABCDE"
    correct = 0
    wrong = 0
    total_count = 0
    accuracy = 0
    list_wrong_data = []
    for index, instance in enumerate(ground_truth_file.items()):
        # Accumulate the number of questions
        total_count += 1

        # Get the question ID and reference data
        id = instance[0]
        d = instance[1]

        # Get reference data
        query_id = id.strip()
        answers 	= d['choices'][d['answer']]
        answer_index = d['answer']
       
        # If there are multiple correct answers, compare one by one
        if type(answers) != list:
              answers = [answers]
       
        # Get the generated result
        prediction = list_predicted_results[index]['generated_text']

        # If the generated result matches the ABCD format, compare the answers
        choice = extract_choice(prediction)
        if  list_choices[answer_index].upper() == choice.upper():
            correct += 1
        else:
            list_wrong_data.append({
                "task_name": task_name,
                "id": query_id,
                "source": d,
                "inferred_results": list_predicted_results[index],
                "generated_text": prediction,
            })
            wrong += 1

    # Calculate accuracy
    accuracy = round((correct / total_count) * 100.0, 2)

    return correct, wrong, total_count, accuracy, list_wrong_data


'''
Exact Matching
'''
# If the generated answer is not in the form of options ABCD, but rather the text of the options, use prefix matching
# Source: https://github.com/mtkresearch/MR-Models/blob/main/TC-Eval/evaluate.py#L16C1-L20C61
def prefix_exact_match(answers, prediction):
    if not prediction: 
        return 0
    
    # Convert text to Taiwanese characters/language
    prediction = converter.convert(prediction)

    # Get each answer and compare
    pem = 0
    for ans in answers:
        ans = converter.convert(ans)
        if prediction.strip().startswith(ans.strip()):
            pem = 1
            break
    return pem


'''
Longest Common (Sub) String - lcs
'''
nltk.download('punkt')

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
                '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
                '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

# Remove punctuation
def remove_punctuation(in_str):
    in_str = in_str.lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
                '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
                '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

# Find the longest common substring
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j]+1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p-mmax:p], mmax

# Calculate F1 score (for multiple answers, return the highest matching score)
def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision 	= 1.0 * lcs_len/len(prediction_segs)
        recall 		= 1.0 * lcs_len/len(ans_segs)
        f1 			= (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)

# Calculate exact match score, set em to 1 if ans and prediction are exactly the same
def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

# Evaluate LCS
def evaluate_lcs(ground_truth_file, list_predicted_results, task_name):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    threshold = 0.4
    list_wrong_data = []
    for index, instance in enumerate(ground_truth_file.items()):
        id = instance[0]
        d = instance[1]

        total_count += 1
        query_id    = id.strip()
        query_text  = d['question'].strip()
        answers 	= d['choices'][d['answer']]

        if type(answers) != list:
            answers = [answers]

        flag_id_in = False
        for _ in list_predicted_results:
            if query_id == _['id']:
                flag_id_in = True
                break

        if flag_id_in == False:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            skip_count += 1
            continue

        prediction 	= list_predicted_results[index]['generated_text']
        
        # Calculate f1-score and exact matching score
        f1_score = calc_f1_score(answers, prediction)
        em_score = calc_em_score(answers, prediction)
        f1 += f1_score
        em += em_score

        # If the current average (f1-score + em_score) is not greater than the threshold, record the wrong data
        if (f1_score + em_score) * 0.5 < threshold:
            list_wrong_data.append({
                "task_name": task_name,
                "id": query_id,
                "source": d,
                "generated_text": prediction,
            })


    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count, list_wrong_data

