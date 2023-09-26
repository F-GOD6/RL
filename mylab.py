from tqdm import tqdm
import numpy as np
def train_on_policy_agent(env, agent, num_episodes):
    return_list=[]
    for i in range(10):
        with tqdm(total=int(num_episodes/10),desc="第%d轮"%i) as pbar:
            for i_epoisode in range(int(num_episodes/10)):
                state=env.reset()[0]
                done=False
                transition_dict={
                    "states":[],
                    "rewards":[],
                    "actions":[],
                    "dones":[],
                    "next_states":[]
                }
                episode_return = 0
                while not done:
                    action=agent.take_action(state)
                    next_state,reward,terminated,truncated,_=env.step(action)

                    transition_dict["states"].append(state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    state=next_state
                    episode_return+=reward

                    done=terminated or truncated
                    transition_dict["dones"].append(done)
                return_list.append(episode_return)
                agent.update(transition_dict)
                if((i_epoisode+1)%10==0):
                    pbar.set_postfix({
                        'episode':'%d'%(num_episodes/10*i+i_epoisode+1),
                        'return':'%.3f'%np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  # 计算累积和，插入0以处理边界情况
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size  # 计算中间部分的移动平均值
    r = np.arange(1, window_size-1, 2)  # 构建递增的序列以计算开始和结束部分的移动平均值
    begin = np.cumsum(a[:window_size-1])[::2] / r  # 计算开始部分的移动平均值
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]  # 计算结束部分的移动平均值，并反转顺序
    return np.concatenate((begin, middle, end))  # 合并开始、中间和结束部分的移动平均值，并返回结果