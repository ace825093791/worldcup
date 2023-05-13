import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from collections.abc import Iterable
from operator import itemgetter
# 现在我们已经准备好了一个数据库，并且拥有具有预测能力的列，我们可以开始建模了。
# 将测试两个模型:随机森林和梯度增强。被选中的将是记忆最深刻的一个。

model_db = pd.read_csv("kaggle/model_db.csv")
X = model_db.iloc[:, 3:]
y = model_db[["target"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)
#梯度提升分类器
gb = GradientBoostingClassifier(random_state=5)
params = {"learning_rate": [0.01, 0.1, 0.5],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [3, 5],
            "max_depth":[3,5,10],
            "max_features":["sqrt"],
            "n_estimators":[100, 200]
         }

gb_cv = GridSearchCV(gb, params, cv = 3, n_jobs = -1, verbose = False)
gb_cv.fit(X_train.values, np.ravel(y_train))
gb = gb_cv.best_estimator_
print(gb)

params_rf = {"max_depth": [20],
                "min_samples_split": [10],
                "max_leaf_nodes": [175],
                "min_samples_leaf": [5],
                "n_estimators": [250],
                 "max_features": ["sqrt"],
                }
#随机森林分类器
rf = RandomForestClassifier(random_state=1)

rf_cv = GridSearchCV(rf, params_rf, cv = 3, n_jobs = -1, verbose = False)

rf_cv.fit(X_train.values, np.ravel(y_train))
rf = rf_cv.best_estimator_
print(rf)
#分析模型
def analyze(model):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test.values)[:, 1])  # test AUC
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="test")

    fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train.values)[:, 1])  # train AUC
    plt.plot(fpr_train, tpr_train, label="train")
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
    auc_train = roc_auc_score(y_train, model.predict_proba(X_train.values)[:, 1])
    plt.legend()
    plt.title('AUC score is %.2f on test and %.2f on training' % (auc_test, auc_train))
    plt.show()

    plt.figure(figsize=(15, 10))
    cm = confusion_matrix(y_test, model.predict(X_test.values))
    sns.heatmap(cm, annot=True, fmt="d")
#随机森林模型稍微好一点，但似乎不太适合。因此，我们将使用梯度提升模型。
analyze(gb)
analyze(rf)

#开始比赛仿真
#第一件事是创造FIFA世界杯比赛。为了做到这一点，我将在维基百科上找到球队和小组赛。
# dfs = pd.read_html(r"https://en.wikipedia.org/wiki/2022_FIFA_World_Cup#Teams")
# for i in range(len(dfs)):
#     df = dfs[i]
#     cols = list(df.columns.values)
#
#     if isinstance(cols[0], Iterable):
#         if any("Tie-breaking criteria" in c for c in cols):
#             start_pos = i + 1
#
#         if any("Match 46" in c for c in cols):
#             end_pos = i + 1
# matches = []
groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
group_count = 7

#table = {}
# TABLE -> TEAM, POINTS, WIN PROBS (CRITERIO DE DESEMPATE)
#table[groups[group_count]] = [[a.split(" ")[0], 0, []] for a in list(dfs[start_pos].iloc[:, 1].values)]
table = {'A': [['Qatar', 0, []],  ['Ecuador', 0, []],  ['Senegal', 0, []],  ['Netherlands', 0, []]],
         'B': [['England', 0, []],  ['Iran', 0, []],  ['United States', 0, []],  ['Wales', 0, []]],
         'C': [['Argentina', 0, []],  ['Saudi Arabia', 0, []],  ['Mexico', 0, []],  ['Poland', 0, []]],
         'D': [['France', 0, []],  ['Australia', 0, []],  ['Denmark', 0, []],  ['Tunisia', 0, []]],
         'E': [['Spain', 0, []],  ['Costa Rica', 0, []],  ['Germany', 0, []],  ['Japan', 0, []]],
         'F': [['Belgium', 0, []],  ['Canada', 0, []],  ['Morocco', 0, []],  ['Croatia', 0, []]],
         'G': [['Brazil', 0, []],  ['Serbia', 0, []],  ['Switzerland', 0, []],  ['Cameroon', 0, []]],
         'H': [['Portugal', 0, []],  ['Ghana', 0, []],  ['Uruguay', 0, []],  ['South Korea', 0, []]]}

#for i in range(start_pos + 1, end_pos, 1):
# for i in range(13, 67, 1):
#     if len(dfs[i].columns) == 3:
#         team_1 = dfs[i].columns.values[0]
#         team_2 = dfs[i].columns.values[-1]
#
#         matches.append((groups[group_count], team_1, team_2))
#     else:
#         group_count+=1
#         table[groups[group_count]] = [[a, 0, []] for a in list(dfs[i].iloc[:, 1].values)]
matches = [('A', 'Qatar', 'Ecuador'),
 ('A', 'Senegal', 'Netherlands'),
 ('A', 'Qatar', 'Senegal'),
 ('A', 'Netherlands', 'Ecuador'),
 ('A', 'Ecuador', 'Senegal'),
 ('A', 'Netherlands', 'Qatar'),
 ('B', 'England', 'Iran'),
 ('B', 'United States', 'Wales'),
 ('B', 'Wales', 'Iran'),
 ('B', 'England', 'United States'),
 ('B', 'Wales', 'England'),
 ('B', 'Iran', 'United States'),
 ('C', 'Argentina', 'Saudi Arabia'),
 ('C', 'Mexico', 'Poland'),
 ('C', 'Poland', 'Saudi Arabia'),
 ('C', 'Argentina', 'Mexico'),
 ('C', 'Poland', 'Argentina'),
 ('C', 'Saudi Arabia', 'Mexico'),
 ('D', 'Denmark', 'Tunisia'),
 ('D', 'France', 'Australia'),
 ('D', 'Tunisia', 'Australia'),
 ('D', 'France', 'Denmark'),
 ('D', 'Australia', 'Denmark'),
 ('D', 'Tunisia', 'France'),
 ('E', 'Germany', 'Japan'),
 ('E', 'Spain', 'Costa Rica'),
 ('E', 'Japan', 'Costa Rica'),
 ('E', 'Spain', 'Germany'),
 ('E', 'Japan', 'Spain'),
 ('E', 'Costa Rica', 'Germany'),
 ('F', 'Morocco', 'Croatia'),
 ('F', 'Belgium', 'Canada'),
 ('F', 'Belgium', 'Morocco'),
 ('F', 'Croatia', 'Canada'),
 ('F', 'Croatia', 'Belgium'),
 ('F', 'Canada', 'Morocco'),
 ('G', 'Switzerland', 'Cameroon'),
 ('G', 'Brazil', 'Serbia'),
 ('G', 'Cameroon', 'Serbia'),
 ('G', 'Brazil', 'Switzerland'),
 ('G', 'Serbia', 'Switzerland'),
 ('G', 'Cameroon', 'Brazil'),
 ('H', 'Uruguay', 'South Korea'),
 ('H', 'Portugal', 'Ghana'),
 ('H', 'South Korea', 'Ghana'),
 ('H', 'Portugal', 'Uruguay'),
 ('H', 'Ghana', 'Uruguay'),
 ('H', 'South Korea', 'Portugal')]

print(table)
#上面，我们还存储了球队在小组中的得分和每场比赛获胜的概率。当两队得分相同时，获胜概率的平均值将作为决胜局。
print(matches[:10])
#我将使用上一场比赛的统计数据作为参与比赛的每个团队的统计数据。比如，巴西对塞尔维亚，巴西的数据是他们上一场比赛的数据，塞尔维亚也是如此。
team_stats_raw = pd.read_csv("kaggle/team_stats_raw.csv")
def find_stats(team_1):
#team_1 = "Qatar"
    past_games = team_stats_raw[(team_stats_raw["team"] == team_1)].sort_values("date")
    last5 = team_stats_raw[(team_stats_raw["team"] == team_1)].sort_values("date").tail(5)

    team_1_rank = past_games["rank"].values[-1]
    team_1_goals = past_games.score.mean()
    team_1_goals_l5 = last5.score.mean()
    team_1_goals_suf = past_games.suf_score.mean()
    team_1_goals_suf_l5 = last5.suf_score.mean()
    team_1_rank_suf = past_games.rank_suf.mean()
    team_1_rank_suf_l5 = last5.rank_suf.mean()
    team_1_gp_rank = past_games.points_by_rank.mean()
    team_1_gp_rank_l5 = last5.points_by_rank.mean()

    return [team_1_rank, team_1_goals, team_1_goals_l5, team_1_goals_suf, team_1_goals_suf_l5, team_1_rank_suf, team_1_rank_suf_l5, team_1_gp_rank, team_1_gp_rank_l5]


def find_features(team_1, team_2):
    rank_dif = team_1[0] - team_2[0]
    goals_dif = team_1[1] - team_2[1]
    goals_dif_l5 = team_1[2] - team_2[2]
    goals_suf_dif = team_1[3] - team_2[3]
    goals_suf_dif_l5 = team_1[4] - team_2[4]
    goals_per_ranking_dif = (team_1[1] / team_1[5]) - (team_2[1] / team_2[5])
    dif_rank_agst = team_1[5] - team_2[5]
    dif_rank_agst_l5 = team_1[6] - team_2[6]
    dif_gp_rank = team_1[7] - team_2[7]
    dif_gp_rank_l5 = team_1[8] - team_2[8]

    return [rank_dif, goals_dif, goals_dif_l5, goals_suf_dif, goals_suf_dif_l5, goals_per_ranking_dif, dif_rank_agst,
            dif_rank_agst_l5, dif_gp_rank, dif_gp_rank_l5, 1, 0]

#现在，我们可以模拟了。

#由于该模型模拟的是1队是否会赢，因此需要创建一些标准来定义平局。此外，由于我们在世界杯上没有主场优势，我们的想法是预测两场比赛，改变1队和2队。
#概率平均值最高的队伍将被指定为获胜者。在小组赛阶段，如果“主队”作为1队赢了，作为2队输了，或者“主队”作为2队赢了，作为1队输了，那么在那场比赛中会被分配到平局。
advanced_group = []
last_group = ""

for k in table.keys():
    for t in table[k]:
        t[1] = 0
        t[2] = []

for teams in matches:
    draw = False
    team_1 = find_stats(teams[1])
    team_2 = find_stats(teams[2])

    features_g1 = find_features(team_1, team_2)
    features_g2 = find_features(team_2, team_1)

    probs_g1 = gb.predict_proba([features_g1])
    probs_g2 = gb.predict_proba([features_g2])

    team_1_prob_g1 = probs_g1[0][0]
    team_1_prob_g2 = probs_g2[0][1]
    team_2_prob_g1 = probs_g1[0][1]
    team_2_prob_g2 = probs_g2[0][0]

    team_1_prob = (probs_g1[0][0] + probs_g2[0][1]) / 2
    team_2_prob = (probs_g2[0][0] + probs_g1[0][1]) / 2

    if ((team_1_prob_g1 > team_2_prob_g1) & (team_2_prob_g2 > team_1_prob_g2)) | (
            (team_1_prob_g1 < team_2_prob_g1) & (team_2_prob_g2 < team_1_prob_g2)):
        draw = True
        for i in table[teams[0]]:
            if i[0] == teams[1] or i[0] == teams[2]:
                i[1] += 1

    elif team_1_prob > team_2_prob:
        winner = teams[1]
        winner_proba = team_1_prob
        for i in table[teams[0]]:
            if i[0] == teams[1]:
                i[1] += 3

    elif team_2_prob > team_1_prob:
        winner = teams[2]
        winner_proba = team_2_prob
        for i in table[teams[0]]:
            if i[0] == teams[2]:
                i[1] += 3

    for i in table[teams[0]]:  # adding criterio de desempate (probs por jogo)
        if i[0] == teams[1]:
            i[2].append(team_1_prob)
        if i[0] == teams[2]:
            i[2].append(team_2_prob)

    if last_group != teams[0]:
        if last_group != "":
            print("\n")
            print("Group %s advanced: " % (last_group))

            for i in table[last_group]:  # adding crieterio de desempate
                i[2] = np.mean(i[2])

            final_points = table[last_group]
            final_table = sorted(final_points, key=itemgetter(1, 2), reverse=True)
            advanced_group.append([final_table[0][0], final_table[1][0]])
            for i in final_table:
                print("%s -------- %d" % (i[0], i[1]))
        print("\n")
        print("-" * 10 + " Starting Analysis for Group %s " % (teams[0]) + "-" * 10)

    if draw == False:
        print("Group %s - %s vs. %s: Winner %s with %.2f probability" % (
        teams[0], teams[1], teams[2], winner, winner_proba))
    else:
        print("Group %s - %s vs. %s: Draw" % (teams[0], teams[1], teams[2]))
    last_group = teams[0]

print("\n")
print("Group %s advanced: " % (last_group))

for i in table[last_group]:  # adding crieterio de desempate
    i[2] = np.mean(i[2])

final_points = table[last_group]
final_table = sorted(final_points, key=itemgetter(1, 2), reverse=True)
advanced_group.append([final_table[0][0], final_table[1][0]])
for i in final_table:
    print("%s -------- %d" % (i[0], i[1]))

#小组赛阶段预测不会有意外，也可能是巴西和瑞士或者法国和丹麦之间的平局。
#对于季后赛阶段，我将预测并以图形方式展示它，就像这里做的一样。
advanced = advanced_group
playoffs = {"Round of 16": [], "Quarter-Final": [], "Semi-Final": [], "Final": []}
for p in playoffs.keys():
    playoffs[p] = []

actual_round = ""
next_rounds = []

for p in playoffs.keys():
    if p == "Round of 16":
        control = []
        for a in range(0, len(advanced * 2), 1):
            if a < len(advanced):
                if a % 2 == 0:
                    control.append((advanced * 2)[a][0])
                else:
                    control.append((advanced * 2)[a][1])
            else:
                if a % 2 == 0:
                    control.append((advanced * 2)[a][1])
                else:
                    control.append((advanced * 2)[a][0])

        playoffs[p] = [[control[c], control[c + 1]] for c in range(0, len(control) - 1, 1) if c % 2 == 0]

        for i in range(0, len(playoffs[p]), 1):
            game = playoffs[p][i]

            home = game[0]
            away = game[1]
            team_1 = find_stats(home)
            team_2 = find_stats(away)

            features_g1 = find_features(team_1, team_2)
            features_g2 = find_features(team_2, team_1)

            probs_g1 = gb.predict_proba([features_g1])
            probs_g2 = gb.predict_proba([features_g2])

            team_1_prob = (probs_g1[0][0] + probs_g2[0][1]) / 2
            team_2_prob = (probs_g2[0][0] + probs_g1[0][1]) / 2

            if actual_round != p:
                print("-" * 10)
                print("Starting simulation of %s" % (p))
                print("-" * 10)
                print("\n")

            if team_1_prob < team_2_prob:
                print("%s vs. %s: %s advances with prob %.2f" % (home, away, away, team_2_prob))
                next_rounds.append(away)
            else:
                print("%s vs. %s: %s advances with prob %.2f" % (home, away, home, team_1_prob))
                next_rounds.append(home)

            game.append([team_1_prob, team_2_prob])
            playoffs[p][i] = game
            actual_round = p

    else:
        playoffs[p] = [[next_rounds[c], next_rounds[c + 1]] for c in range(0, len(next_rounds) - 1, 1) if c % 2 == 0]
        next_rounds = []
        for i in range(0, len(playoffs[p])):
            game = playoffs[p][i]
            home = game[0]
            away = game[1]
            team_1 = find_stats(home)
            team_2 = find_stats(away)

            features_g1 = find_features(team_1, team_2)
            features_g2 = find_features(team_2, team_1)

            probs_g1 = gb.predict_proba([features_g1])
            probs_g2 = gb.predict_proba([features_g2])

            team_1_prob = (probs_g1[0][0] + probs_g2[0][1]) / 2
            team_2_prob = (probs_g2[0][0] + probs_g1[0][1]) / 2

            if actual_round != p:
                print("-" * 10)
                print("Starting simulation of %s" % (p))
                print("-" * 10)
                print("\n")

            if team_1_prob < team_2_prob:
                print("%s vs. %s: %s advances with prob %.2f" % (home, away, away, team_2_prob))
                next_rounds.append(away)
            else:
                print("%s vs. %s: %s advances with prob %.2f" % (home, away, home, team_1_prob))
                next_rounds.append(home)
            game.append([team_1_prob, team_2_prob])
            playoffs[p][i] = game
            actual_round = p
#画球队晋级图
# import networkx as nx
# from networkx.drawing.nx_pydot import graphviz_layout
#
# plt.figure(figsize=(15, 10))
# G = nx.balanced_tree(2, 3)
#
# labels = []
#
# for p in playoffs.keys():
#     for game in playoffs[p]:
#         label = f"{game[0]}({round(game[2][0], 2)}) \n {game[1]}({round(game[2][1], 2)})"
#         labels.append(label)
#
# labels_dict = {}
# labels_rev = list(reversed(labels))
#
# for l in range(len(list(G.nodes))):
#     labels_dict[l] = labels_rev[l]
#
# pos = graphviz_layout(G, prog='twopi')
# labels_pos = {n: (k[0], k[1] - 0.08 * k[1]) for n, k in pos.items()}
# center = pd.DataFrame(pos).mean(axis=1).mean()
#
# nx.draw(G, pos=pos, with_labels=False, node_color=range(15), edge_color="#bbf5bb", width=10, font_weight='bold',
#         cmap=plt.cm.Greens, node_size=5000)
# nx.draw_networkx_labels(G, pos=labels_pos, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5, alpha=1),
#                         labels=labels_dict)
# texts = ["Round \nof 16", "Quarter \n Final", "Semi \n Final", "Final\n"]
# pos_y = pos[0][1] + 55
# for text in reversed(texts):
#     pos_x = center
#     pos_y -= 75
#     plt.text(pos_y, pos_x, text, fontsize=18)
#
# plt.axis('equal')
# plt.show()

#这就是最后的模拟!巴西赢得了第六个冠军!希望我的预测是正确的。
#分析一下可能出现的麻烦也很好。比利时战胜了德国，最后被葡萄牙击败。阿根廷对荷兰的比赛非常紧张，荷兰的传球优势接近1%。
#同样的情况也发生在法国和英格兰之间，英格兰通过。我认为英格兰进入决赛是模拟比赛中最大的意外。
#更新:数据库更新了各国国家队在世界杯前的最后一场友谊赛，因此，一些模拟也发生了变化。
#法国队在四分之一决赛中击败了英格兰队，在半决赛中被葡萄牙队击败!葡萄牙进入决赛是一个巨大的惊喜!

#结语
#这样做的目的是通过机器学习来模拟我喜欢的东西(足球世界杯)来提高我的知识。
#我认为创造出我们可以在现实生活中看到结果的模型是很神奇的，这就是将要发生的事情!
#总的来说，我认为这个模型的预测就像看足球的人的常识一样。在模拟中没有什么大的惊喜。
#在小组赛中看到不知名球队的比赛也很好，比如伊朗对阵威尔士，或者塞内加尔对阵厄瓜多尔。
#我认为在这样的比赛中，这种模式对投注有很好的指导作用，因为大多数人对二线国家队的了解并不多。
