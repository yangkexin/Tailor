import pickle as pkl
import torch
import torch.nn as nn
sentiment_paths = ["yelp_sentiment0_gpt2_prefix128.pkl","yelp_random_sentiment1_gpt2_prefix128.pkl"]
food_paths = ["yelp_food0_gpt2_prefix128.pkl","yelp_food1_gpt2_prefix128.pkl","yelp_food2_gpt2_prefix128.pkl"]
s_prompt = []
f_prompt = []
for path in sentiment_random_paths:
    temp_prompt = pkl.load(open(path,"rb"))
      s_prompt.append(temp_prompt)
    for path in food_random_paths:
      temp_prompt = pkl.load(open(path,"rb"))
      f_prompt.append(temp_prompt)
task_dict = dict()
task_dict[0]=s_prompt[0]
task_dict[1]=s_prompt[1]
task_dict[2]=f_prompt[0]
task_dict[3]=f_prompt[1]
task_dict[4]=f_prompt[2]
pkl.dump(task_dict,open("single_task_dict.pkl","wb"))

prompt_dict = dict()
prompt_dict["0_0"]=torch.cat((s_prompt[0],f_prompt[0]),dim=0)
prompt_dict["0_1"]=torch.cat((s_prompt[0],f_prompt[1]),dim=0)
prompt_dict["0_2"]=torch.cat((s_prompt[0],f_prompt[2]),dim=0)
prompt_dict["1_0"]=torch.cat((s_prompt[0],f_prompt[0]),dim=0)
prompt_dict["1_1"]=torch.cat((s_prompt[0],f_prompt[1]),dim=0)
prompt_dict["1_2"]=torch.cat((s_prompt[0],f_prompt[2]),dim=0)

pkl.dump(prompt_dict,open("prompt_task_dict.pkl","wb"))
      
