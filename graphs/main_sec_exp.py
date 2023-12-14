import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import pingouin as pg
from pingouin import mwu
import scipy.stats as st
from scipy.stats import mannwhitneyu

def convert_types(df):
    df['g1_CLRL_RL_vs_HS']= df['g1_CLRL_RL_vs_HS'].fillna(0).astype(int)
    df['g1_CLRL_RL_vs_CL'] = df['g1_CLRL_RL_vs_CL'].fillna(0).astype(int)
    df['g1_CLRL_HS_vs_CL'] = df['g1_CLRL_HS_vs_CL'].fillna(0).astype(int)
    df['g1_A1_CLRL_confident'] = df['g1_A1_CLRL_confident'].fillna(0).astype(int)
    df['g2_RLCL_RL_vs_HS'] = df['g2_RLCL_RL_vs_HS'].fillna(0).astype(int)
    df['g2_RLCL_RL_vs_CL'] = df['g2_RLCL_RL_vs_CL'].fillna(0).astype(int)
    df['g2_RLCL_HS_vs_CL'] = df['g2_RLCL_HS_vs_CL'].fillna(0).astype(int)
    df['g2_A1_RLCL_confident'] = df['g2_A1_RLCL_confident'].fillna(0).astype(int)
    df['g3_HSCL_RL_vs_HS'] = df['g3_HSCL_RL_vs_HS'].fillna(0).astype(int)
    df['g3_HSCL_RL_vs_CL'] = df['g3_HSCL_RL_vs_CL'].fillna(0).astype(int)
    df['g3_HSCL_HS_vs_CL'] = df['g3_HSCL_HS_vs_CL'].fillna(0).astype(int)
    df['g3_A1_HSCL_confident'] = df['g3_A1_HSCL_confident'].fillna(0).astype(int)
    df['sat HL understood better'] = df['sat HL understood better'].fillna(0).astype(int)
    df['sat HL sufficient'] = df['sat HL sufficient'].fillna(0).astype(int)
    df['sat HL irrelevant'] = df['sat HL irrelevant'].fillna(0).astype(int)
    df['sat HL check'] = df['sat HL check'].fillna(0).astype(int)
    df['sat HL useful'] = df['sat HL useful'].fillna(0).astype(int)
    df['sat HL useful scenarios'] = df['sat HL useful scenarios'].fillna(0).astype(int)
    #df['Explanation_video_1'] = df['Explanation_video_1'].fillna(0).astype(int)
    #df['Explanation_video_2'] = df['Explanation_video_2'].fillna(0).astype(int)
    df['g1_HSCL_RL_vs_HS'] = df['g1_HSCL_RL_vs_HS'].fillna(0).astype(int)
    df['g1_HSCL_RL_vs_CL'] = df['g1_HSCL_RL_vs_CL'].fillna(0).astype(int)
    df['g1_HSCL_HS_vs_CL'] = df['g1_HSCL_HS_vs_CL'].fillna(0).astype(int)
    df['g1_A2_HSCL_confident'] = df['g1_A2_HSCL_confident'].fillna(0).astype(int)
    df['g2_CLRL_RL_vs_HS'] = df['g2_CLRL_RL_vs_HS'].fillna(0).astype(int)
    df['g2_CLRL_RL_vs_CL'] = df['g2_CLRL_RL_vs_CL'].fillna(0).astype(int)
    df['g2_CLRL_HS_vs_CL'] = df['g2_CLRL_HS_vs_CL'].fillna(0).astype(int)
    df['g2_A2_CLRL_confident'] = df['g2_A2_CLRL_confident'].fillna(0).astype(int)
    df['g3_RLCL_RL_vs_HS'] = df['g3_RLCL_RL_vs_HS'].fillna(0).astype(int)
    df['g3_RLCL_RL_vs_CL'] = df['g3_RLCL_RL_vs_CL'].fillna(0).astype(int)
    df['g2_CLRL_HS_vs_CL'] = df['g2_CLRL_HS_vs_CL'].fillna(0).astype(int)
    df['g3_A2_RLCL_confident'] = df['g3_A2_RLCL_confident'].fillna(0).astype(int)
    df['sat HL understood better RD'] = df['sat HL understood better RD'].fillna(0).astype(int)
    df['sat HL sufficient RD'] = df['sat HL sufficient RD'].fillna(0).astype(int)
    df['sat HL irrelevant RD'] = df['sat HL irrelevant RD'].fillna(0).astype(int)
    df['sat HL check RD'] = df['sat HL check RD'].fillna(0).astype(int)
    df['sat HL useful RD'] = df['sat HL useful RD'].fillna(0).astype(int)
    df['sat HL useful scenarios RD'] = df['sat HL useful scenarios RD'].fillna(0).astype(int)
    df['g1_RLCL_RL_vs_HS'] = df['g1_RLCL_RL_vs_HS'].fillna(0).astype(int)
    df['g1_RLCL_RL_vs_CL'] = df['g1_RLCL_RL_vs_CL'].fillna(0).astype(int)
    df['g1_RLCL_HS_vs_CL'] = df['g1_RLCL_HS_vs_CL'].fillna(0).astype(int)
    df['g1_A3_RLCL_confident'] = df['g1_A3_RLCL_confident'].fillna(0).astype(int)
    df['g2_HSCL_RL_vs_HS'] = df['g2_HSCL_RL_vs_HS'].fillna(0).astype(int)
    df['g2_HSCL_RL_vs_CL'] = df['g2_HSCL_RL_vs_CL'].fillna(0).astype(int)
    df['g2_HSCL_HS_vs_CL'] = df['g2_HSCL_HS_vs_CL'].fillna(0).astype(int)
    df['g2_A3_HSCL_confident'] = df['g2_A3_HSCL_confident'].fillna(0).astype(int)
    df['g3_CLRL_RL_vs_HS'] = df['g3_CLRL_RL_vs_HS'].fillna(0).astype(int)
    df['g3_CLRL_RL_vs_CL'] = df['g3_CLRL_RL_vs_CL'].fillna(0).astype(int)
    df['g3_CLRL_HS_vs_CL'] = df['g3_CLRL_HS_vs_CL'].fillna(0).astype(int)
    df['g3_A3_CLRL_confident'] = df['g3_A3_CLRL_confident'].fillna(0).astype(int)
    df['sat HL understood better RD.1'] = df['sat HL understood better RD.1'].fillna(0).astype(int)
    df['sat HL sufficient RD.1'] = df['sat HL sufficient RD.1'].fillna(0).astype(int)
    df['sat HL irrelevant RD.1'] = df['sat HL irrelevant RD.1'].fillna(0).astype(int)
    df['sat HL check RD.1'] = df['sat HL check RD.1'].fillna(0).astype(int)
    df['sat HL useful RD.1'] = df['sat HL useful RD.1'].fillna(0).astype(int)
    df['sat HL useful scenarios RD.1'] = df['sat HL useful scenarios RD.1'].fillna(0).astype(int)
    df['Age_1'] = df['Age_1'].fillna(0).astype(int)
    df['Gender'] = df['Gender'].fillna(0).astype(int)
    df['prior_knoweldge'] = df['prior_knoweldge'].fillna(0).astype(int)
    df['Q_TotalDuration'] = df['Q_TotalDuration'].fillna(0).astype(int)
    return df



def remove_attention_check (df):
    df['sat HL check'] = df['sat HL check'].fillna(0).astype(int)
    df['sat HL check RD'] = df['sat HL check RD'].fillna(0).astype(int)
    df['sat HL check RD.1'] = df['sat HL check RD.1'].fillna(0).astype(int)

    index_check = df[(df['sat HL check'] + df['sat HL check RD']+df['sat HL check RD.1'] != 2)].index
    df.drop(index_check, inplace=True)
    return df

def number_of_participants_per_condition(df):
    num_CH = len(df[df.condition =='CH'])
    num_CRD = len(df[df.condition == 'CRD'])
    num_RD = len(df[df.condition == 'RD'])
    print('number of participants in condition "CH ',num_CH)
    print('number of participants in condition "CRD" ', num_CRD)
    print('number of participants in condition "RD" ', num_RD)
    return (num_CH,num_CRD,num_RD)


def duration_per_condition(df):
    df_CH = df[df.condition =='CH']
    df_CRD = df[df.condition == 'CRD']
    df_RD = df[df.condition == 'RD']
    mean_duration_CH = df_CH['Q_TotalDuration'].mean()
    mean_duration_CRD = df_CRD['Q_TotalDuration'].mean()
    mean_duration_RD = df_RD['Q_TotalDuration'].mean()
    print("asd")
    #return (num_CH,num_CRD,num_RD)

def sex_overall(df):
    count_Female = 0
    count_Male = 0
    for index, row in df.iterrows():
        if row['Gender']==1:
            count_Male +=1
        elif row['Gender']==2:
            count_Female+=1
    print("number of female overall:", count_Female)
    print("number of male overall:", count_Male)
    y = np.array([count_Female,count_Male])
    mylabels = ["Female", "Male"]

    plt.pie(y, labels=mylabels)
    plt.show()

def sex_per_condition(df):
    participants_per_cond = number_of_participants_per_condition(df)
    sum_by_condition = df.groupby('condition', as_index=False).sum()
    num_of_Female_per_condition = sum_by_condition['Gender']-participants_per_cond
    num_of_Male_per_condition = participants_per_cond -num_of_Female_per_condition
    labels = sum_by_condition['condition']
    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, num_of_Male_per_condition, width, label='Men')
    ax.bar(labels, num_of_Female_per_condition, width, bottom=num_of_Male_per_condition,
           label='Women')

    ax.set_ylabel('Number of participants')
    ax.legend()
    plt.show()

def demography(df):
    mean_age = df['Age_1'].mean()
    print("the mean age is:", mean_age)
    sex_overall(df)
    sex_per_condition(df)

def create_df_for_agent_graphs(df):
    mini_df = df[
        ['condition',
         'g1_CLRL_RL_vs_HS', 'g1_CLRL_RL_vs_CL', 'g1_CLRL_HS_vs_CL',
         'g2_RLCL_RL_vs_HS', 'g2_RLCL_RL_vs_CL', 'g2_RLCL_HS_vs_CL',
         'g3_HSCL_RL_vs_HS', 'g3_HSCL_RL_vs_CL', 'g3_HSCL_HS_vs_CL',
         'g1_HSCL_RL_vs_HS', 'g1_HSCL_RL_vs_CL', 'g1_HSCL_HS_vs_CL',
         'g2_CLRL_RL_vs_HS', 'g2_CLRL_RL_vs_CL', 'g2_CLRL_HS_vs_CL',
         'g3_RLCL_RL_vs_HS', 'g3_RLCL_RL_vs_CL', 'g3_RLCL_HS_vs_CL',
         'g1_RLCL_RL_vs_HS', 'g1_RLCL_RL_vs_CL', 'g1_RLCL_HS_vs_CL',
         'g2_HSCL_RL_vs_HS', 'g2_HSCL_RL_vs_CL', 'g2_HSCL_HS_vs_CL',
         'g3_CLRL_RL_vs_HS', 'g3_CLRL_RL_vs_CL', 'g3_CLRL_HS_vs_CL',
         'Gender','Age_1'
         ]]
    return mini_df

def create_bar_ALL_Agent_by_condition(df_agents):
    #df_agents['All Agents'] = df_agents[['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']].mean(axis=1) * 2
    df_agents['All Agents'] = df_agents[['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']].mean(axis=1)
    sns.set(font_scale=1.5)
    graph = sns.barplot(x="condition", y='All Agents', data=df_agents, estimator=np.mean, ci=95, capsize=.2, color='blue')
    graph.set_xlabel("Condition", fontsize=20)
    graph.set_ylabel("Correctness Rate", fontsize=20)
    graph.axhline(0.3)
    graph.set_title('Highway Environment ')
    #graph.bar_label(graph.containers[0])
    plt.show()


def create_bar_Agent_by_condition(q_df):

    q_df['Agent 1']= q_df[['g1_CLRL_RL_vs_HS', 'g1_CLRL_RL_vs_CL', 'g1_CLRL_HS_vs_CL','g2_CLRL_RL_vs_HS',
                           'g2_CLRL_RL_vs_CL', 'g2_CLRL_HS_vs_CL','g3_CLRL_RL_vs_HS', 'g3_CLRL_RL_vs_CL',
                           'g3_CLRL_HS_vs_CL']].sum(axis = 1)/3
    q_df['Agent 2'] = q_df[[ 'g1_RLCL_RL_vs_HS', 'g1_RLCL_RL_vs_CL', 'g1_RLCL_HS_vs_CL',
                             'g2_RLCL_RL_vs_HS', 'g2_RLCL_RL_vs_CL', 'g2_RLCL_HS_vs_CL',
                             'g3_RLCL_RL_vs_HS', 'g3_RLCL_RL_vs_CL', 'g3_RLCL_HS_vs_CL',
                             ]].sum(axis=1)/3
    q_df['Agent 3'] = q_df[[ 'g1_HSCL_RL_vs_HS', 'g1_HSCL_RL_vs_CL', 'g1_HSCL_HS_vs_CL',
                             'g2_HSCL_RL_vs_HS', 'g2_HSCL_RL_vs_CL', 'g2_HSCL_HS_vs_CL',
                             'g3_HSCL_RL_vs_HS', 'g3_HSCL_RL_vs_CL', 'g3_HSCL_HS_vs_CL',
                             ]].sum(axis=1)/3

    agent_columns = ('Agent 1','Agent 2','Agent 3')
    agent_title = ('Change Lane Right Lane', 'Right Lane Change Lane', 'High Speed Change Lane')
    agent_title = ('First study - Agent 1', 'First study  - Agent 2', 'First study - Agent 3')
    for i in range(len(agent_columns)):
        y_axis = agent_columns[i]
        title_name = agent_title[i]
        sns.set(font_scale=1.5)
        graph = sns.barplot(x="condition", y=y_axis, data=q_df, estimator=np.mean, ci=95, capsize=.2, color='blue')
        graph.set_xlabel("Explanation Type", fontsize=15)
        graph.set_ylabel('Correctness Rate', fontsize=15)
        graph.set_title(title_name)
        graph.axhline(0.3)
        graph.bar_label(graph.containers[0])
        plt.show()

    con_CH = q_df[q_df.condition == 'CH']['Agent 1'].to_frame()
    con_RD = q_df[q_df.condition == 'RD']['Agent 1'].to_frame()
    con_CRD = q_df[q_df.condition == 'CRD']['Agent 1'].to_frame()
    print(np.mean(con_CH))
    U_H_RD_FS_RD, p_H_RD_FS_RD = mannwhitneyu(con_RD, con_CRD)
    print("the U1 is:", U_H_RD_FS_RD, "the p- value is:", p_H_RD_FS_RD)

    U, P = mannwhitneyu(con_CH, con_CRD)
    print("the U1 is:", U, "the p- value is:", P)

    return q_df

def create_bar_ALL_Agent_by_condition(df_agents):
    df_agents['All Agents'] = df_agents[['Agent 1', 'Agent 2', 'Agent 3']].mean(axis=1)
    sns.set(font_scale=1.5)
    graph = sns.barplot(x="condition", y='All Agents', data=df_agents, estimator=np.mean, ci=95, capsize=.2, color='blue')
    graph.set_xlabel("Explanation Type", fontsize=30)
    graph.set_ylabel("Correctness Rate", fontsize=30)
    graph.axhline(0.3)
    graph.set_title('Study 1 (between-subject)', fontsize=30)
    #graph.bar_label(graph.containers[0])
    plt.show()

def create_bar_ALL_Agent_by_sex(df_agents):
    df_agents['All Agents'] = df_agents[['Agent 1', 'Agent 2', 'Agent 3']].mean(axis=1)
    sns.set(font_scale=1.5)
    graph = sns.barplot(x="Gender", y='All Agents', data=df_agents, estimator=np.mean, ci=95, capsize=.2, color='blue')
    graph.set_xlabel("Gender", fontsize=30)
    x_labels = ['Male','Female']
    graph.xaxis.set_ticklabels(x_labels)
    graph.set_ylabel("Correctness Rate", fontsize=30)
    graph.axhline(0.3)
    graph.set_title('Study 1 Correctness by Gender', fontsize=20)
    #graph.bar_label(graph.containers[0])
    plt.show()


def convert_df_agents_to_bin_by_age(df_agents):
    #temp = df_agents[df_agents.Age_1 > 17 and df_agents.Age_1 <30]
    df_agents['age_group'] = pd.cut(df_agents['Age_1'],bins=[18,30,40,55],
                                    labels=['18-29','30-39','40-55'],include_lowest=True)
    print("sf")
    return df_agents



def create_bar_ALL_Agent_by_Age(df_agents):

    df_agents['All Agents'] = df_agents[['Agent 1', 'Agent 2', 'Agent 3']].mean(axis=1)
    sns.set(font_scale=1.5)
    graph = sns.barplot(x="age_group", y='All Agents', data=df_agents, estimator=np.mean, ci=95, capsize=.2, color='blue')
    graph.set_xlabel("Age", fontsize=30)
    graph.set_ylabel("Correctness Rate", fontsize=30)
    graph.axhline(0.3)
    graph.set_title('Study 1 Correctness by Age', fontsize=20)
    #graph.bar_label(graph.containers[0])
    plt.show()


def temp_1(df_agents):
    df_agent_1 = df_agents[["condition", "Agent 1"]]
    df_agent_1 = df_agent_1.rename(columns={"Agent 1": "score"})
    df_agent_1['Agent'] = 'Agent 1'

    df_agent_2 = df_agents[["condition", "Agent 2"]]
    df_agent_2 = df_agent_2.rename(columns={"Agent 2": "score"})
    df_agent_2['Agent'] = 'Agent 2'

    df_agent_3 = df_agents[["condition", "Agent 3"]]
    df_agent_3 = df_agent_3.rename(columns={"Agent 3": "score"})
    df_agent_3['Agent'] = 'Agent 3'

    agents = pd.concat([df_agent_1, df_agent_2, df_agent_3], axis=0)
    agents = agents.replace("CRD","CORD")
    agents = agents.replace("CH", "COViz")
    order_list = ["COViz","RD","CORD"]

    conditions = ['COViz', 'RD', 'CORD',"SUM"]
    palette_colors = sns.color_palette('tab10')
    palette_dict = {condition: color for condition, color in zip(conditions, palette_colors)}

    graph = sns.barplot(x="Agent", y='score', hue='condition',
                        #palette=['tab:blue', 'tab:green', 'tab:orange'],
                        palette=palette_dict,
                        data=agents,
                        estimator=np.mean, ci=95, capsize=.2, color='blue'
                        )#order = order_list)
    graph.set_xlabel("", fontsize=30)
    graph.set_ylabel("Correctness Rate", fontsize=30)
    graph.legend(loc=2, bbox_to_anchor=(1,1))
    graph.set_title('Study 2', fontsize=30)
    # graph.bar_label(graph.containers[0])
    plt.show()



def statistical_confidance(df_agents):

    con_CH = df_agents[df_agents.condition =='CH']['All Agents'].to_frame()
    con_RD = df_agents[df_agents.condition == 'RD']['All Agents'].to_frame()
    con_CRD = df_agents[df_agents.condition == 'CRD']['All Agents'].to_frame()
    print(np.mean(con_CH))
    U_H_RD_FS_RD, p_H_RD_FS_RD = mannwhitneyu(con_RD, con_CRD)
    print("the U1 is:", U_H_RD_FS_RD, "the p- value is:", p_H_RD_FS_RD)
   #U_H_H_RD, p_H_H_RD = mannwhitneyu(con_H_RD, con_H_RD)
    #print("the U1 is:", U_H_H_RD, "the p- value is:", p_H_H_RD)

    print("Real!!!")
    FS_RD_VS_H_LESS = mwu(con_CRD, con_CH, alternative='less')
    FS_RD_VS_H_GRATER = mwu(con_CRD, con_CH, alternative='greater')
    print(FS_RD_VS_H_LESS)
    print(FS_RD_VS_H_GRATER)

    H_RD_VS_H_GRATER = mwu(con_RD, con_CH, alternative='greater')
    print(H_RD_VS_H_GRATER)

    U_FS_RD_H, p_FS_RD_H = mannwhitneyu(con_CRD, con_CH)
    print("For FS+RD and H,","the U1 is:", U_FS_RD_H, "the p- value is:", p_FS_RD_H)
    print("sdfsdfsdfsdsdfsdf")
    U_H_RD_H, p_H_RD_H = mannwhitneyu(con_RD, con_CH)
    print("For H+RD and H,","the U1 is:", U_H_RD_H, "the p- value is:", p_H_RD_H)
    print("done")



def sum_all_sat(df_union):
    df_union['sum all sat CH']= (df_union['sat HL understood better RD']+df_union['sat HL sufficient RD']+df_union['sat HL irrelevant RD']+df_union['sat HL useful RD']+df_union['sat HL useful scenarios RD'])/5
    df_union['sum all sat CRD']= (df_union['sat HL understood better RD.1']+df_union['sat HL sufficient RD.1']+df_union['sat HL irrelevant RD.1']+df_union['sat HL useful RD.1']+df_union['sat HL useful scenarios RD.1'])/5
    df_union['sum all sat RD']=(df_union['sat HL understood better']+df_union['sat HL sufficient']+df_union['sat HL irrelevant']+df_union['sat HL useful']+df_union['sat HL useful scenarios'])/5
    df_union['satisfaction']= df_union['sum all sat RD']+ df_union['sum all sat CRD']+ df_union['sum all sat CH']
    return df_union

#def sum_all_confidant(df_union):
#    df_union['sum all confidant CH']= (df_union['sat HL understood better RD']+df_union['sat HL sufficient RD']+df_union['sat HL irrelevant RD']+df_union['sat HL useful RD']+df_union['sat HL useful scenarios RD'])/5
#    df_union['sum all sat CRD']= (df_union['sat HL understood better RD.1']+df_union['sat HL sufficient RD.1']+df_union['sat HL irrelevant RD.1']+df_union['sat HL useful RD.1']+df_union['sat HL useful scenarios RD.1'])/5
#    df_union['sum all sat RD']=(df_union['sat HL understood better']+df_union['sat HL sufficient']+df_union['sat HL irrelevant']+df_union['sat HL useful']+df_union['sat HL useful scenarios'])/5
#    df_union['satisfaction']= df_union['sum all sat RD']+ df_union['sum all sat CRD']+ df_union['sum all sat CH']
#    return df_union




def graph_for_sat(df_union):
    df_union=sum_all_sat(df_union)

    #graph = sns.boxplot(data=df_union, x="condition", y='satisfaction', color='blue')
    graph = sns.barplot(x="condition", y='satisfaction', data=df_union, estimator=np.mean, ci=95, capsize=.2, color='blue')
    #graph = sns.barplot(x="IS_RD", y='satisfaction', data=df_union, estimator=np.mean, ci=95, capsize=.2, color='blue')
    graph.set_xlabel("Explanation Type", fontsize=30)
    graph.set_ylabel("Satisfaction Rate", fontsize=30)
    graph.axhline(3)
    graph.bar_label(graph.containers[0])
    graph.set_title('Study 1 - satisfaction', fontsize=30)
    plt.show()



def plot_confident_rate(df_CH,df_RD,df_CRD):
    df_CH,df_RD,df_CRD = overall_confident(df_CH, df_RD, df_CRD)
    df_CH_temp = df_CH[['confident', 'condition']]
    df_RD_temp = df_RD[['confident', 'condition']]
    df_CRD_temp = df_CRD[['confident', 'condition']]
    frames = [df_CH_temp, df_RD_temp, df_CRD_temp]
    df_plot = pd.concat(frames)
    graph = sns.barplot(x="condition", y='confident', data=df_plot,
                    estimator=np.mean, ci=95, capsize=.2, color='blue')
    graph.set_xlabel("Explanation Type", fontsize=20)
    graph.set_ylabel("Confident Rate", fontsize=20)
    graph.axhline(3)
    graph.bar_label(graph.containers[0])
    plt.show()
if __name__ == '__main__':
    # load data
    mturk_df = pd.read_csv('Batch_4903529_batch_results.csv')
    qualtrics_df = pd.read_csv('CRD+-+Number2_October+18,+2022_23.05.csv')
    mturk_df_5_6 = pd.read_csv('Batch_4904951_batch_results.csv')
    qualtrics_df_5_6 = pd.read_csv('CRD+-+Number2_October+19,+2022_22.39.csv')
    # adjust data
    qualtrics_df.drop(axis=0, labels=[0, 1], inplace=True)
    qualtrics_df = qualtrics_df.rename(columns={'Random ID': 'Answer.surveycode'})
    mturk_df['Answer.surveycode'] = mturk_df['Answer.surveycode'].astype(str)
    qualtrics_df_5_6.drop(axis=0, labels=[0, 1], inplace=True)
    qualtrics_df_5_6 = qualtrics_df_5_6.rename(columns={'Random ID': 'Answer.surveycode'})
    mturk_df_5_6['Answer.surveycode'] = mturk_df_5_6['Answer.surveycode'].astype(str)
    df = qualtrics_df.merge(mturk_df, on='Answer.surveycode', how='inner')
    df_1 = qualtrics_df_5_6.merge(mturk_df_5_6, on='Answer.surveycode', how='inner')

    df_union = pd.concat([df_1, df], axis=0, ignore_index=True)
    print('number of rows in the beginning:', len(df_union.index))
    df_union = remove_attention_check(df_union)
    print('number of rows to analyze:', len(df_union.index))
    df_union = convert_types(df_union)
    duration_per_condition(df_union)
    participants_per_cond = number_of_participants_per_condition(df_union)
    demography(df_union)
    q_df = create_df_for_agent_graphs(df_union)
    df_agents = create_bar_Agent_by_condition(q_df)
    create_bar_ALL_Agent_by_condition(df_agents)

    create_bar_ALL_Agent_by_sex(df_agents)
    df_agents = convert_df_agents_to_bin_by_age(df_agents)
    create_bar_ALL_Agent_by_Age(df_agents)

    statistical_confidance(df_agents)
    graph_for_sat(df_union)
    #TEMPPP(df_agents)
    temp_1(df_agents)