# expert step rewards

import glob
import numpy as np
import json

traj_path = 'outputs/virtualhome/expert/monte_carlo_sample_iteration'
results = {}
results_original_reward = {}
sample_num = len(glob.glob(traj_path + "/*"))
for i in range(sample_num):
    paths = glob.glob(f"{traj_path}/sampled_traj_{i}/*.json")
    cur_results = []
    for path in paths:
        cur_results.extend(json.load(open(path)))
    for item in cur_results:
        id = item['path']
        iteration = item['iteration']
        id_iteration = f"{id}__{iteration}"
        if id_iteration not in results:
            results[id_iteration] = [item['agent_final_reward']]
        else:
            results[id_iteration].append(item['agent_final_reward'])
        if id_iteration not in results_original_reward:
            results_original_reward[id_iteration] = item['agent_step_reward']
        else:
            assert results_original_reward[id_iteration] == item['agent_step_reward']

            
final_results = {}
for key, value in results.items():
    final_results[key] = {
        "monte_carlo_reward": np.mean(value),
        "env_reward": results_original_reward[key]
    }
    
gt_data = {}
for each in final_results.keys():
    id_iteration = each.split("__")
    id = id_iteration[0]
    iteration = id_iteration[1]
    reward = final_results[each]['monte_carlo_reward']
    gt_data[each] = {"id": id, "iteration": int(iteration), "monte_carlo_step_reward": reward}



with open('data/new_train_sft.json', 'r') as f:
    new_train_sft = json.load(f)

new_train_sft_dict = {}
for each in new_train_sft:
    new_train_sft_dict[each['path']] = each['conversations']
    
def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item['role'] == "assistant":
            message['from'] = "gpt"
            message['value'] = item['content'].strip()
        else:
            message['from'] = "human"
            message['value'] = item['content'].strip()
        messages.append(message)
    return messages

import glob

to_be_sample_data = []
to_be_sample_data_dict = {}
data_path = 'outputs/virtualhome/expert/expert_step_explore'
number = 0

# 使用 glob.glob 来匹配文件路径
for file in glob.glob(f"{data_path}/*.json"):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            agent_step_output = item['agent_step_output']
            new_item = {}
            new_item['id'] = item['id']
            new_item['iteration'] = item['iteration']
            new_item['agent_step_output'] = agent_step_output
            new_item['gpt_step_reward'] = item['agent_step_reward']
            new_item['gpt_step_conversations'] = template_change(item['agent_step_conversations'][0:2] + item['agent_step_conversations'][12:-1])
            new_item['path'] = item['path']
            new_item['gpt_conversations'] = new_train_sft_dict[item['path']]
            new_item['gpt_reward'] = 1
            text_index = item['path'] + '__' + str(item['iteration'])
            if text_index not in final_results:
                number = number + 1
                continue
            new_item['monte_carlo_step_reward'] = final_results[item['path'] + '__' + str(item['iteration'])]['monte_carlo_reward']
            to_be_sample_data.append(new_item)
            to_be_sample_data_dict[item['path']+'__'+str(item['iteration'])] = new_item
            


# explore step rewards

import glob
import numpy as np
import json

traj_path = 'outputs/virtualhome/explore/monte_carlo_sample_iteration'
results = {}
results_original_reward = {}
sample_num = len(glob.glob(traj_path + "/*"))
for i in range(sample_num):
    paths = glob.glob(f"{traj_path}/sampled_traj_{i}/*.json")
    cur_results = []
    for path in paths:
        cur_results.extend(json.load(open(path)))
    for item in cur_results:
        id = item['path']
        iteration = item['iteration']
        id_iteration = f"{id}__{iteration}"
        if id_iteration not in results:
            results[id_iteration] = [item['agent_final_reward']]
        else:
            results[id_iteration].append(item['agent_final_reward'])
        if id_iteration not in results_original_reward:
            results_original_reward[id_iteration] = item['agent_step_reward']
        else:
            assert results_original_reward[id_iteration] == item['agent_step_reward']
final_results = {}
for key, value in results.items():
    final_results[key] = {
        "monte_carlo_reward": np.mean(value),
        "env_reward": results_original_reward[key]
    }
    
output_data = []
for file in glob.glob(f"outputs/virtualhome/explore/explore_step_explore/*.json"):
    data = json.load(open(file))
    for item in data:
        id = item['path']
        iteration = item['iteration']
        id_iteration = f"{id}__{iteration}"
        
        if id_iteration in final_results:
            item['monte_carlo_step_reward'] = final_results[id_iteration]['monte_carlo_reward']
        else:
            item['monte_carlo_step_reward'] = item['agent_step_reward']
            print('+1')
        output_data.append(item)
        

instruction_part = {
    "webshop": [
        {
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
    "sciworld": [
        {
            "from": "human",
            "value": "You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.\n\nThe available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
    "alfworld": [
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
    "virtualhome":[
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\nAction: your next action\".\n\nThe available actions are:\nwalk to {obj}\nrun to {obj}\ngrab {obj}\nopen {obj}\nclose {obj}\nput {obj} on {recep}\nput {obj} in {recep}\nswitch on {obj}\nswitch off {obj}\ndrink {obj}\nlook at {obj}\nsit on {obj}\nstand up\nwatch {obj}\nwipe {obj}\ntype on {obj}\nwash {obj}\ncut {obj}\neat {obj}\nsleep\nwake up\nplug in {obj}\nplug out {obj}\npour {obj} into {recep}\nmove {obj}\nrelease\nturn to {obj}\n\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        }]
}

icl_prompt_part = {
    "webshop":[
        {
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should use the search bar to look for the product I need.\nAction: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L'eau D'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L'eau D'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L'Eau d'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L'Eau D'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L'eau D'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L'eau d'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L'eau d'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the product I need, which is B000VOHH8I.\nAction: click[B000VOHH8I]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.\nAction: click[6.76 fl oz (pack of 1)]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the 'Buy Now' button to purchase the product.\nAction: click[Buy Now]"
        }
    ],
    "alfworld": [{
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger."
        },
        {
            "from": "gpt",
            "value": "Thought: I need to place a toilet paper roll on the toilet paper hanger. First, I should check if there's already a roll on the hanger or if it's empty.\nAction: go to toiletpaperhanger 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toiletpaperhanger 1, you see nothing."
        },
        {
            "from": "gpt",
            "value": "Thought: The toilet paper hanger is empty, so I need to find a new roll of toilet paper to put on it.\nAction: go to toilet 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1."
        },
        {
            "from": "gpt",
            "value": "Thought: I've found a toilet paper roll on the toilet. I should pick it up so I can place it on the hanger.\nAction: take toiletpaper 1 from toilet 1"
        },
        {
            "from": "human",
            "value": "Observation: You pick up the toiletpaper 1 from the toilet 1."
        },
        {
            "from": "gpt",
            "value": "Thought: Now that I have the toilet paper roll, I need to return to the toilet paper hanger to install it.\nAction: go to toiletpaperhanger 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toiletpaperhanger 1, you see nothing."
        },
        {
            "from": "gpt",
            "value": "Thought: I'm back at the toilet paper hanger with the roll in hand. It's time to put the toilet paper on the hanger.\nAction: put toiletpaper 1 in/on toiletpaperhanger 1"
        }],
    "virtualhome":[
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\nAction: your next action\".\n\nThe available actions are:\nwalk to {obj}\nrun to {obj}\ngrab {obj}\nopen {obj}\nclose {obj}\nput {obj} on {recep}\nput {obj} in {recep}\nswitch on {obj}\nswitch off {obj}\ndrink {obj}\nlook at {obj}\nsit on {obj}\nstand up\nwatch {obj}\nwipe {obj}\ntype on {obj}\nwash {obj}\ncut {obj}\neat {obj}\nsleep\nwake up\nplug in {obj}\nplug out {obj}\npour {obj} into {recep}\nmove {obj}\nrelease\nturn to {obj}\n\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "The task is Drink (Drink water)."
        },
        {
            "from": "gpt",
            "value": "Thought: To drink water, I need to first locate a water source in the household, so moving towards the dining room might bring me closer to finding water or a suitable container for drinking.\nAction: Walk to dining room"
        },
        {
            "from": "human",
            "value": "Observation:"
        },
        {
            "from": "gpt",
            "value": "Thought: I chose to walk to the water glass as it is likely located in the dining room where I can access water easily, aligning with the goal of drinking water.\nAction: Walk to water glass"
        },
        {
            "from": "human",
            "value": "Observation: water_glass is CLEAN."
        },
        {
            "from": "gpt",
            "value": "Thought: The current observation indicates that I've located a clean water glass, which is essential to proceed with the task of drinking water. Now, I need to fill it with water to accomplish the task goal.\nAction: Find water glass"
        },
        {
            "from": "human",
            "value": "Observation: water_glass is CLEAN."
        },
        {
            "from": "gpt",
            "value": "Thought: The observation indicates that the water glass is clean, and since the task is to drink water, holding the glass will allow further steps to achieve this goal.\nAction: Grab water glass"
        },
        {
            "from": "human",
            "value": "Observation: character, water_glass is CLEAN. You have water_glass."
        },
        {
            "from": "gpt",
            "value": "Thought: Since I have already obtained the water glass, the next logical step is to drink from it to complete the task of drinking water.\nAction: Drink water glass"
        }]
}

def is_empty_conversations(conversation):
    for item in conversation:
        if item['value'].strip() == "":
            return True
    return False


# construct expert sub-trajectory dataset

from tqdm import tqdm

win = 0
lose = 0
tie = 0
error = 0
step_monte_carlo_threshold = 0.01
traj_threshold = 0.01
number = 0
pm_data = []
for item in tqdm(output_data):
        
    iteration = item['iteration']
    path = item['path']

    if (iteration == 0) or (iteration == 1):
        continue

    agent_conversations = template_change(item['agent_conversations'])
    agent_step_reward = item['monte_carlo_step_reward']
    agent_final_reward = item['agent_final_reward']
    agent_step_conversations = item['agent_step_conversations']
    
    if f"{path}__{iteration}" not in to_be_sample_data_dict:
        number = number + 1
        continue

    gpt_conversations = to_be_sample_data_dict[f"{path}__{iteration}"]['gpt_conversations']
    gpt_step_reward = to_be_sample_data_dict[f"{path}__{iteration}"]['monte_carlo_step_reward']
    gpt_final_reward = to_be_sample_data_dict[f"{path}__{iteration}"]['gpt_reward']
    gpt_step_conversations = to_be_sample_data_dict[f"{path}__{iteration}"]['gpt_step_conversations']

    # new_reward = gt_data[f"{id}_{iteration-1}"]['monte_carlo_step_reward']

    gpt_last_step_reward = to_be_sample_data_dict[f"{path}__{iteration-1}"]['monte_carlo_step_reward']
    
    gpt_length = len(gpt_step_conversations)
    agent_length = len(gpt_step_conversations) - len(instruction_part['virtualhome'])+len(icl_prompt_part['virtualhome'])
    

    if is_empty_conversations(agent_conversations) or is_empty_conversations(gpt_conversations):
        continue
    
    if agent_final_reward == True:
        agent_final_reward = 1
    else:
        agent_final_reward = 0

    if agent_step_reward == None:
        continue

    if agent_final_reward > gpt_final_reward + 0.01 and agent_step_reward > gpt_step_reward + step_monte_carlo_threshold:
        win += 1
        pm_data.append({
            "id": f"{path}__{iteration}",
            "prompt": gpt_conversations[:gpt_length-1],
            "chosen": agent_conversations[agent_length-1: -1],
            "rejected": gpt_conversations[gpt_length-1:],
        })
    elif gpt_final_reward > agent_final_reward and gpt_step_reward > agent_step_reward + step_monte_carlo_threshold:
        lose += 1
        pm_data.append({
            "id": f"{path}__{iteration}",
            "prompt": gpt_conversations[:gpt_length-1],
            "chosen": gpt_conversations[gpt_length-1:],
            "rejected": agent_conversations[agent_length-1: -1],
        })
    else:
        tie += 1


json.dump(pm_data, open('outputs/virtualhome/constructed_data/step_pm.json', "w"), indent=4) 


# construct explored successful trajectory dataset

pm_data = []
for item in output_data:

    # game_file = item['game_file']
    # id = game_file_to_id[game_file]

    path = item['path']
    iteration = item['iteration']

    if iteration != 0:
        continue

    agent_final_reward = item['agent_final_reward']
    gpt_reward = to_be_sample_data_dict[f"{path}__{0}"]['gpt_reward']
    gpt_conversations = to_be_sample_data_dict[f"{path}__{0}"]['gpt_conversations']
    
    agent_conversations = template_change(item['agent_conversations'])
    if is_empty_conversations(agent_conversations) or is_empty_conversations(gpt_conversations):
        continue
    if agent_final_reward > gpt_reward + traj_threshold:
        win += 1
        pm_data.append({
            "id": f"{path}__{iteration}",
            "prompt": gpt_conversations[:len(instruction_part['virtualhome'])+1],
            "chosen": agent_conversations[len(icl_prompt_part['virtualhome'])+1: -1],
            "rejected": gpt_conversations[len(instruction_part['virtualhome'])+1:],
        })
    elif gpt_reward>agent_final_reward + traj_threshold:
        lose += 1
        pm_data.append({
            "id": f"{path}__{iteration}",
            "prompt": gpt_conversations[:len(instruction_part['virtualhome'])+1],
            "chosen": gpt_conversations[len(instruction_part['virtualhome'])+1:],
            "rejected": agent_conversations[len(icl_prompt_part['virtualhome'])+1: -1],
        })
    else:
        tie += 1
        
json.dump(pm_data, open('outputs/virtualhome/constructed_data/traj_pm.json', "w"), indent=4) 