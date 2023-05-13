import pandas as pd

df = pd.read_csv("kaggle/df_wc_ranked.csv")
#计算分数，胜3分 平1分 负0分
def result_finder(home, away):
    if home > away:
        return pd.Series([0, 3, 0])
    if home < away:
        return pd.Series([1, 0, 3])
    else:
        return pd.Series([2, 1, 1])

results = df.apply(lambda x: result_finder(x["home_score"], x["away_score"]), axis=1)

df[["result", "home_team_points", "away_team_points"]] = results
print(results)

import seaborn as sns
import matplotlib.pyplot as plt
#画热力图
#plt.figure(figsize=(15, 10))
#sns.heatmap(df[["total_points_home", "rank_home", "total_points_away", "rank_away"]].corr())
#plt.show()
#现在，我们创建的列将有助于创建特性:排名差异，在比赛中赢得的分数与球队面对的排名，以及在比赛中的进球差异。应该为两支球队（客场和主场）创建所有不存在差异的特征。
df["rank_dif"] = df["rank_home"] - df["rank_away"]
df["sg"] = df["home_score"] - df["away_score"]
df["points_home_by_rank"] = df["home_team_points"]/df["rank_away"]
df["points_away_by_rank"] = df["away_team_points"]/df["rank_home"]

#为了创建的功能，我会在主队和客场队的数据集分离的数据集，统一他们和计算过去的游戏值。之后，我将再次分离并合并它们，检索原始数据集。此过程优化了特征的创建。
home_team = df[["date", "home_team", "home_score", "away_score", "rank_home", "rank_away","rank_change_home", "total_points_home", "result", "rank_dif", "points_home_by_rank", "home_team_points"]]
away_team = df[["date", "away_team", "away_score", "home_score", "rank_away", "rank_home","rank_change_away", "total_points_away", "result", "rank_dif", "points_away_by_rank", "away_team_points"]]

home_team.columns = [h.replace("home_", "").replace("_home", "").replace("away_", "suf_").replace("_away", "_suf") for h in home_team.columns]
away_team.columns = [a.replace("away_", "").replace("_away", "").replace("home_", "suf_").replace("_home", "_suf") for a in away_team.columns]
print(home_team)
print(away_team)

team_stats = home_team.append(away_team)#.sort_values("date")
#this column will be used to calculate features for simulation
team_stats_raw = team_stats.copy()
# 将数据保存到 CSV 文件中
team_stats_raw.to_csv('kaggle/team_stats_raw.csv', index=False, encoding='utf-8')

# 现在，我们有了一个数据库，可以创建预测功能。他们将是：
# 球队在世界杯周期的平均进球数。
# 球队最近5场比赛的平均进球数。
# 球队在世界杯周期中的平均进球数。
# 球队在最近5场比赛中的平均进球数。
# 球队在世界杯周期中所面临的平均国际足联排名。
# 该队在过去5场比赛中的平均国际足联排名。
# 国际足联积分在循环赛中获胜。
# FIFA积分在最近5场比赛中获胜。
# 循环的平均游戏点数。
# 最近5场比赛的平均得分。
# 在循环赛中按等级划分的平均比赛点数。
# 在最近5场比赛中，按级别划分的平均比赛点数。
stats_val = []

for index, row in team_stats.iterrows():
    team = row["team"]
    date = row["date"]
    past_games = team_stats.loc[(team_stats["team"] == team) & (team_stats["date"] < date)].sort_values(by=['date'],
                                                                                                        ascending=False)
    last5 = past_games.head(5)
    goals = past_games["score"].mean()
    goals_l5 = last5["score"].mean()
    goals_suf = past_games["suf_score"].mean()
    goals_suf_l5 = last5["suf_score"].mean()
    rank = past_games["rank_suf"].mean()
    rank_l5 = last5["rank_suf"].mean()
    if len(last5) > 0:
        points = past_games["total_points"].values[0] - past_games["total_points"].values[-1]  # qtd de pontos ganhos
        points_l5 = last5["total_points"].values[0] - last5["total_points"].values[-1]
    else:
        points = 0
        points_l5 = 0
    gp = past_games["team_points"].mean()
    gp_l5 = last5["team_points"].mean()
    gp_rank = past_games["points_by_rank"].mean()
    gp_rank_l5 = last5["points_by_rank"].mean()

    stats_val.append(
        [goals, goals_l5, goals_suf, goals_suf_l5, rank, rank_l5, points, points_l5, gp, gp_l5, gp_rank, gp_rank_l5])

stats_cols = ["goals_mean", "goals_mean_l5", "goals_suf_mean", "goals_suf_mean_l5", "rank_mean", "rank_mean_l5", "points_mean", "points_mean_l5", "game_points_mean", "game_points_mean_l5", "game_points_rank_mean", "game_points_rank_mean_l5"]
stats_df = pd.DataFrame(stats_val, columns=stats_cols)
full_df = pd.concat([team_stats.reset_index(drop=True), stats_df], axis=1, ignore_index=False)

home_team_stats = full_df.iloc[:int(full_df.shape[0]/2),:]
away_team_stats = full_df.iloc[int(full_df.shape[0]/2):,:]
print(home_team_stats)
print(away_team_stats)
#home_team_stats.columns[-12:]

home_team_stats = home_team_stats[home_team_stats.columns[-12:]]
away_team_stats = away_team_stats[away_team_stats.columns[-12:]]
print(home_team_stats)
print(away_team_stats)
#为了统一数据库，需要为每一列添加home和away后缀。之后，数据就可以进行合并了。
home_team_stats.columns = ['home_'+str(col) for col in home_team_stats.columns]
away_team_stats.columns = ['away_'+str(col) for col in away_team_stats.columns]

match_stats = pd.concat([home_team_stats, away_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)
full_df = pd.concat([df, match_stats.reset_index(drop=True)], axis=1, ignore_index=False)
#full_df.columns
#现在，为了量化游戏的重要性，创建了一个专栏，寻找游戏的竞争。
def find_friendly(x):
    if x == "Friendly":
        return 1
    else: return 0

full_df["is_friendly"] = full_df["tournament"].apply(lambda x: find_friendly(x))
full_df = pd.get_dummies(full_df, columns=["is_friendly"])
print(full_df.columns)

base_df = full_df[["date", "home_team", "away_team", "rank_home", "rank_away","home_score", "away_score","result", "rank_dif", "rank_change_home", "rank_change_away", 'home_goals_mean',
       'home_goals_mean_l5', 'home_goals_suf_mean', 'home_goals_suf_mean_l5',
       'home_rank_mean', 'home_rank_mean_l5', 'home_points_mean',
       'home_points_mean_l5', 'away_goals_mean', 'away_goals_mean_l5',
       'away_goals_suf_mean', 'away_goals_suf_mean_l5', 'away_rank_mean',
       'away_rank_mean_l5', 'away_points_mean', 'away_points_mean_l5','home_game_points_mean', 'home_game_points_mean_l5',
       'home_game_points_rank_mean', 'home_game_points_rank_mean_l5','away_game_points_mean',
       'away_game_points_mean_l5', 'away_game_points_rank_mean',
       'away_game_points_rank_mean_l5',
       'is_friendly_0', 'is_friendly_1']]

print(base_df.tail())
print(base_df.isna().sum())
#带有NA的游戏是指无法计算的游戏（从数据集开始的游戏）。这些将被丢弃。
base_df_no_fg = base_df.dropna()

# 将数据保存到 CSV 文件中
base_df_no_fg.to_csv('kaggle/base_df_no_fg.csv', index=False, encoding='utf-8')