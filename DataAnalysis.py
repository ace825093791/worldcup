import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 现在，我们需要分析所有创建的功能，并检查它们是否具有预测能力。
# 此外，如果他们没有，我们需要创造一些有，比如主客场球队的差异。
# 为了分析预测能力，我将把平局比赛分配为主队的失利，并将产生一个二元问题。
df = pd.read_csv("kaggle/base_df_no_fg.csv")

def no_draw(x):
    if x == 2:
        return 1
    else:
        return x

df["target"] = df["result"].apply(lambda x: no_draw(x))
# 将进行的分析:
# 小提琴和箱线图分析特征是否根据目标有不同的分布
# 散点图分析相关性

#小提琴图
data1 = df[list(df.columns[8:20].values) + ["target"]]
data2 = df[df.columns[20:]]
scaled = (data1[:-1] - data1[:-1].mean()) / data1[:-1].std()
scaled["target"] = data1["target"]
# violin1 = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")
# scaled = (data2[:-1] - data2[:-1].mean()) / data2[:-1].std()
# scaled["target"] = data2["target"]
# violin2 = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")
# plt.figure(figsize=(15,10))
# sns.violinplot(x="features", y="value", hue="target", data=violin1,split=True, inner="quart")
# plt.xticks(rotation=90)
#plt.show()
#通过这些图，我们发现秩差是数据的唯一好的分隔符。但是，我们可以创建一些特征来区分主客场球队之间的差异，并分析它们是否能够很好地分离数据。
# plt.figure(figsize=(15,10))
# sns.violinplot(x="features", y="value", hue="target", data=violin2,split=True, inner="quart")
# plt.xticks(rotation=90)
#plt.show()

dif = df.copy()
dif.loc[:, "goals_dif"] = dif["home_goals_mean"] - dif["away_goals_mean"]
dif.loc[:, "goals_dif_l5"] = dif["home_goals_mean_l5"] - dif["away_goals_mean_l5"]
dif.loc[:, "goals_suf_dif"] = dif["home_goals_suf_mean"] - dif["away_goals_suf_mean"]
dif.loc[:, "goals_suf_dif_l5"] = dif["home_goals_suf_mean_l5"] - dif["away_goals_suf_mean_l5"]
dif.loc[:, "goals_made_suf_dif"] = dif["home_goals_mean"] - dif["away_goals_suf_mean"]
dif.loc[:, "goals_made_suf_dif_l5"] = dif["home_goals_mean_l5"] - dif["away_goals_suf_mean_l5"]
dif.loc[:, "goals_suf_made_dif"] = dif["home_goals_suf_mean"] - dif["away_goals_mean"]
dif.loc[:, "goals_suf_made_dif_l5"] = dif["home_goals_suf_mean_l5"] - dif["away_goals_mean_l5"]

data_difs = dif.iloc[:, -8:]
scaled = (data_difs - data_difs.mean()) / data_difs.std()
scaled["target"] = data2["target"]

# violin = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")
# plt.figure(figsize=(10,10))
# sns.violinplot(x="features", y="value", hue="target", data=violin,split=True, inner="quart")
# plt.xticks(rotation=90)
#plt.show()

# 从这个图中我们可以看出，目标差异是一个很好的分隔符，而且目标也存在差异。球队进球和失球之间的差异并不是很好的区分因素。
# 现在，我们有5个特点:
# rank_dif
# goals_dif
# goals_dif_l5
# goals_suf_dif
# goals_suf_dif_l5
# 我们还可以创建其他的特征，比如点的差值，rank face的点的差值，rank face的差值。
dif.loc[:, "dif_points"] = dif["home_game_points_mean"] - dif["away_game_points_mean"]
dif.loc[:, "dif_points_l5"] = dif["home_game_points_mean_l5"] - dif["away_game_points_mean_l5"]
dif.loc[:, "dif_points_rank"] = dif["home_game_points_rank_mean"] - dif["away_game_points_rank_mean"]
dif.loc[:, "dif_points_rank_l5"] = dif["home_game_points_rank_mean_l5"] - dif["away_game_points_rank_mean_l5"]
dif.loc[:, "dif_rank_agst"] = dif["home_rank_mean"] - dif["away_rank_mean"]
dif.loc[:, "dif_rank_agst_l5"] = dif["home_rank_mean_l5"] - dif["away_rank_mean_l5"]
#此外，我们还可以计算排名的目标实现和损失，并检查这种差异。
dif.loc[:, "goals_per_ranking_dif"] = (dif["home_goals_mean"] / dif["home_rank_mean"]) - (dif["away_goals_mean"] / dif["away_rank_mean"])
dif.loc[:, "goals_per_ranking_suf_dif"] = (dif["home_goals_suf_mean"] / dif["home_rank_mean"]) - (dif["away_goals_suf_mean"] / dif["away_rank_mean"])
dif.loc[:, "goals_per_ranking_dif_l5"] = (dif["home_goals_mean_l5"] / dif["home_rank_mean"]) - (dif["away_goals_mean_l5"] / dif["away_rank_mean"])
dif.loc[:, "goals_per_ranking_suf_dif_l5"] = (dif["home_goals_suf_mean_l5"] / dif["home_rank_mean"]) - (dif["away_goals_suf_mean_l5"] / dif["away_rank_mean"])
data_difs = dif.iloc[:, -10:]
scaled = (data_difs - data_difs.mean()) / data_difs.std()
scaled["target"] = data2["target"]
# violin = pd.melt(scaled,id_vars="target", var_name="features", value_name="value")
#
# plt.figure(figsize=(15,10))
# sns.violinplot(x="features", y="value", hue="target", data=violin,split=True, inner="quart")
# plt.xticks(rotation=90)
#plt.show()
#由于这些值很低，小提琴图并不是一个很好的选择来分析特征在这种情况下是否真的分离了数据。我们将看到箱线图:
# plt.figure(figsize=(15,10))
# sns.boxplot(x="features", y="value", hue="target", data=violin)
# plt.xticks(rotation=90)
#plt.show()
#分差(满5局和最后5局)、排名差(满5局和最后5局)和排名差(满5局和最后5局)是很好的功能。
# 此外，一些生成的特征具有非常相似的分布，将使用散点图进行分析
# sns.jointplot(data = data_difs, x = 'goals_per_ranking_dif', y = 'goals_per_ranking_dif_l5', kind="reg")
# plt.show()

#通过排名所面对的净胜球差异和它最近5场比赛的版本有着非常相似的分布。
# 因此，我们将只使用完整版本(goals_per_ranking_dif)。
sns.jointplot(data = data_difs, x = 'dif_rank_agst', y = 'dif_rank_agst_l5', kind="reg")
plt.show()

sns.jointplot(data = data_difs, x = 'dif_points', y = 'dif_points_l5', kind="reg")
plt.show()

sns.jointplot(data = data_difs, x = 'dif_points_rank', y = 'dif_points_rank_l5', kind="reg")
plt.show()

# 对于所面对的rank，所面对的游戏点数以及所面对的平均游戏点数的差异，两个版本(完整和5局)并没有那么相似。所以，我们将两者都使用。
# 基于此，最终的特征是:
# rank_dif
# goals_dif
# goals_dif_l5
# goals_suf_dif
# goals_suf_dif_l5
# dif_rank_agst
# dif_rank_agst_l5
# goals_per_ranking_dif
# dif_points_rank
# dif_points_rank_l5
# is_friendly
def create_db(df):
    columns = ["home_team", "away_team", "target", "rank_dif", "home_goals_mean", "home_rank_mean", "away_goals_mean",
               "away_rank_mean", "home_rank_mean_l5", "away_rank_mean_l5", "home_goals_suf_mean", "away_goals_suf_mean",
               "home_goals_mean_l5", "away_goals_mean_l5", "home_goals_suf_mean_l5", "away_goals_suf_mean_l5",
               "home_game_points_rank_mean", "home_game_points_rank_mean_l5", "away_game_points_rank_mean",
               "away_game_points_rank_mean_l5", "is_friendly_0", "is_friendly_1"]

    base = df.loc[:, columns]
    base.loc[:, "goals_dif"] = base["home_goals_mean"] - base["away_goals_mean"]
    base.loc[:, "goals_dif_l5"] = base["home_goals_mean_l5"] - base["away_goals_mean_l5"]
    base.loc[:, "goals_suf_dif"] = base["home_goals_suf_mean"] - base["away_goals_suf_mean"]
    base.loc[:, "goals_suf_dif_l5"] = base["home_goals_suf_mean_l5"] - base["away_goals_suf_mean_l5"]
    base.loc[:, "goals_per_ranking_dif"] = (base["home_goals_mean"] / base["home_rank_mean"]) - (
                base["away_goals_mean"] / base["away_rank_mean"])
    base.loc[:, "dif_rank_agst"] = base["home_rank_mean"] - base["away_rank_mean"]
    base.loc[:, "dif_rank_agst_l5"] = base["home_rank_mean_l5"] - base["away_rank_mean_l5"]
    base.loc[:, "dif_points_rank"] = base["home_game_points_rank_mean"] - base["away_game_points_rank_mean"]
    base.loc[:, "dif_points_rank_l5"] = base["home_game_points_rank_mean_l5"] - base["away_game_points_rank_mean_l5"]

    model_df = base[["home_team", "away_team", "target", "rank_dif", "goals_dif", "goals_dif_l5", "goals_suf_dif",
                     "goals_suf_dif_l5", "goals_per_ranking_dif", "dif_rank_agst", "dif_rank_agst_l5",
                     "dif_points_rank", "dif_points_rank_l5", "is_friendly_0", "is_friendly_1"]]
    return model_df
model_db = create_db(df)
print(model_db)
# 将数据保存到 CSV 文件中
model_db.to_csv('kaggle/model_db.csv', index=False, encoding='utf-8')





















