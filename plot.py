import pickle
import matplotlib.pyplot as plt
import os

#foldertaglist = ['Minigrid5x5', 'Minigrid6x6', 'Minigrid8x8_avs_3', 'Minigrid8x8_avs_7',
#                 'Minigrid8x8_alter2', 'Minigrid8x8_alter3' ]
#foldertaglist = os.listdir('results/new_results')
foldertaglist = ['MiniGrid8x8_alter1']
for foldertag in foldertaglist:
    #rewards_path = os.path.join('results/new_results', foldertag, 'rewards.pickle')
    rewards_path = os.path.join('results', foldertag, 'rewards_1234567.pickle')
    params_path = os.path.join('results/new_results', foldertag, 'params.pickle')

    with open(rewards_path, 'rb') as f:
        rewards = pickle.load(f)

    try:
        with open(params_path, 'rb') as f:
            params = pickle.load(f)

    except FileNotFoundError:
        with open('params.pickle', 'rb') as f:
            params = pickle.load(f)

    smoothed_rewards = [rewards[0]]
    alpha = 0.99
    for i in range(1,len(rewards)):
        smoothed_rewards.append(alpha*smoothed_rewards[i-1] + (1-alpha)*rewards[i])

    plt.figure(1)
    plt.xlabel(['epsiode'])
    plt.ylabel('reward/episode')
    plt.title(f'n_hidden: {params["n_hidden"]}, learning_rate: {params["learning_rate"]}')
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.legend(['rewards', f'smoothed_rewards with alpha: {alpha}'])
    #plt.show()
    #fig_name = os.path.join('results/new_results', foldertag, f'{foldertag}rewards')
    fig_name = os.path.join('results', foldertag, f'{foldertag}rewards_1234567')
    plt.savefig(fig_name)
    plt.close()