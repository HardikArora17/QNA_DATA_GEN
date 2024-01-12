from tqdm.autonotebook import tqdm
import pandas as pd


def create_data(file_name):
    qna_data = pd.read_csv(file_name)
    abstracts = qna_data['abstract']
    questions = qna_data['question']
    answers = qna_data['answer']

    instruct_data = []

    for abstract, ques, answer in tqdm(zip(abstracts, questions, answers), desc="READING DATA..."):
        try:
            instruction = f"Study the whole scientific abstract given below and try to answer the given question based on abstract.\n[ABSTRACT]: {abstract}\n [QUESTION]: {ques}"
            format_llm = f"<s> [INST] {instruction} [/INST] {answer} </s>"
            instruct_data.append(format_llm)
        except:
            continue

    print("TOTAL TRAINING SAMPLES GATHERED: ", len(instruct_data))

    return instruct_data
