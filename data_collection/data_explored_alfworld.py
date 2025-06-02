# Collecting explored successful trajectory dataset
# Organize all data (calibration data, explored successful trajectory dataset, expert sub-trajectory dataset) and add reward

import json
from glob import glob
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import math

# 检查CUDA是否可用
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device)

def get_embedding(text):
    # 编码文本
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    # 获取模型输出
    with torch.no_grad():
        output = model(**encoded_input)
    # 返回 [CLS] token 的嵌入向量
    return output.last_hidden_state[:, 0, :].cpu().numpy()

def cal_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


# print(cal_similarity("Go to bedroom", "Go to bedroom"))


def parse_action(llm_output: str) -> str:
    llm_output = llm_output.strip()
    pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
    action = re.findall(pattern, llm_output)
    if len(action) == 0:
        return ""
    else:
        return action[0]


def plan_ndtw(plan1, plan2):
    n, m = len(plan1), len(plan2)
    D = [[0 for j in range(m)] for i in range(n)]
    C = [[0 for j in range(m)] for i in range(n)]
    
    # 填充距离矩阵 D
    for i in range(n):
        for j in range(m):
            D[i][j] = 1 - cal_similarity(parse_action(plan1[i]), parse_action(plan2[j]))
    
    # 填充累积距离矩阵 C
    C[0][0] = D[0][0]
    for i in range(1, n):
        C[i][0] = D[i][0] + C[i-1][0]
    for j in range(1, m):
        C[0][j] = D[0][j] + C[0][j-1]
    
    for i in range(1, n):
        for j in range(1, m):
            C[i][j] = D[i][j] + min(C[i-1][j], C[i][j-1], C[i-1][j-1])
    
    # 计算DTW距离
    dtw_distance = C[n-1][m-1]
    
    # 归一化 (使用对角线长度归一化)
    ndtw = dtw_distance / math.sqrt(n**2 + m**2)
    
    return ndtw


all_data = []
for file in glob("outputs/alfworld/explore/explore_step_explore/*.json"):
    data = json.load(open(file))
    for item in data:
        all_data.append(item)

alfworld_sft = json.load(open("data/alfworld_sft.json"))

RFT_alfworld_data = []
for item in all_data:
    if item['iteration'] != 0:
        continue
    if item['agent_final_reward'] != True:
        continue
    
    new_item = {'id': item['id'], 'game_file': item['game_file']}
    assert item['agent_conversations'][-1]['role'] == 'user'
    agent_conversations = item['agent_conversations'][:2] + item['agent_conversations'][12:-1]
    # print('agent_conversations:', agent_conversations)
    
    new_conversations = []
    for each in agent_conversations:
        if each['role'] == 'user':
            new_conversations.append({
                'from': 'human',
                'value': each['content']
            })
        else:
            new_conversations.append({
                'from': 'gpt',
                'value': each['content']
            })
            
    new_item['conversations'] = new_conversations
    RFT_alfworld_data.append(new_item)

alfworld_sft_dict = {}
for each in alfworld_sft:
    alfworld_sft_dict[each['game_file']] = each['conversations']
alfworld_sft_dict

RFT_alfworld_data_pair = []
for i in range(len(RFT_alfworld_data)):
    gt_conversations = alfworld_sft_dict[RFT_alfworld_data[i]['game_file']]
    new_dict = {}
    new_dict['id'] = RFT_alfworld_data[i]['game_file']
    new_dict['prompt'] = gt_conversations[:3]
    new_dict['rejected'] = gt_conversations[3:]
    new_dict['chosen'] = RFT_alfworld_data[i]['conversations'][3:]
    RFT_alfworld_data_pair.append(new_dict)
    
    
from tqdm import tqdm
for i in tqdm(range(len(RFT_alfworld_data_pair))):
    chosen_list = []
    for item in RFT_alfworld_data_pair[i]['chosen']:
        if item['from'] == 'gpt':
            chosen_list.append(item['value'])
    reject_list = []
    for item in RFT_alfworld_data_pair[i]['rejected']:
        if item['from'] == 'gpt':
            reject_list.append(item['value'])
    reward = plan_ndtw(chosen_list, reject_list)
    print(reward)

    RFT_alfworld_data_pair[i]['deviation'] = reward

json.dump(RFT_alfworld_data_pair, open('outputs/alfworld/constructed_data/alfworld_rft_rewards.json', "w"), indent=4)



# 原始的pairs数据
import json

alfworld_data_path = 'outputs/alfworld/constructed_data/total.json'

alfworld_data = json.load(open(alfworld_data_path))

from tqdm import tqdm
for i in tqdm(range(len(alfworld_data))):
    chosen_list = []
    for item in alfworld_data[i]['chosen']:
        if item['from'] == 'gpt':
            chosen_list.append(item['value'])
    reject_list = []
    for item in alfworld_data[i]['rejected']:
        if item['from'] == 'gpt':
            reject_list.append(item['value'])
    reward = plan_ndtw(chosen_list, reject_list)
    print(reward)

    alfworld_data[i]['deviation'] = reward

json.dump(alfworld_data, open('outputs/alfworld/constructed_data/alfworld_steca_rewards.json', "w"), indent=4)




# Organize all data (calibration data, explored successful trajectory dataset, expert sub-trajectory dataset) and add reward
import json

data1 = json.load(open("outputs/alfworld/constructed_data/alfworld_steca_rewards.json"))
data2 = json.load(open("outputs/alfworld/constructed_data/alfworld_rft_rewards.json"))

for i in range(len(data1)):
    data1[i]['reward'] = 1 + 0.01*data1[i]['deviation']
    for j in range(len(data1[i]['prompt'])):
        data1[i]['prompt'][j]['value'] = data1[i]['prompt'][j]['value'].strip()
    for j in range(len(data1[i]['chosen'])):
        data1[i]['chosen'][j]['value'] = data1[i]['chosen'][j]['value'].strip()
    for j in range(len(data1[i]['rejected'])):
        data1[i]['rejected'][j]['value'] = data1[i]['rejected'][j]['value'].strip()

new_data2 = []
for i in range(len(data2)):
    data2[i]['reward'] = 1 - 0.01*data2[i]['deviation']
    for j in range(len(data2[i]['prompt'])):
        data2[i]['prompt'][j]['value'] = data2[i]['prompt'][j]['value'].strip()
    for j in range(len(data2[i]['chosen'])):
        data2[i]['chosen'][j]['value'] = data2[i]['chosen'][j]['value'].strip()
    for j in range(len(data2[i]['rejected'])):
        data2[i]['rejected'][j]['value'] = data2[i]['rejected'][j]['value'].strip()
    
    if data2[i]['deviation'] < 0.013:
        new_data2.append(data2[i])


final_data = []
final_data.extend(data1)
final_data.extend(new_data2)

for i in range(len(final_data)):
    final_data[i]['id'] = str(i)

json.dump(final_data, open('outputs/alfworld/constructed_data/final_data_rft_train.json', "w"), indent=4)