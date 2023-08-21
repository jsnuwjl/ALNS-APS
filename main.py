import os
import pandas as pd
import numpy as np
from itertools import groupby, pairwise, permutations, filterfalse
from random import shuffle, seed, sample
import time
from multiprocessing import Pool
import datetime
import zipfile


seed(0)
SMALL_COLOR = ['量子红', '量子红-Y', '冰玫粉', '冰玫粉-Y', '蒂芙尼蓝', '星漫绿', '星漫绿-Y', '琉璃红', '夜荧黄', '黄绿荧', '薄荷贝绿', '烟雨青', '幻光紫', '广交红', '闪电橙', '脉冲蓝', '天际灰', '火焰橙', '幻光紫', '幻光紫-Y', '琉璃红', '松花黄', '松花黄-Y']
BIG_COLOR = ['白云蓝', '极地白', '极地白-Y', '幻影银', '幻影银(出租车)', '极速银', '极速银-Y', '极速银(出租车)', '夜影黑', '夜影黑-Y', '自由灰', '自由灰-Y', '素雅灰', '素雅灰-Y', '天青色', '天青色-Y', '珍珠白', '全息银']
ATTRIBUTE = ['计划日期', '车型', '天窗', '外色描述BAK', '大颜色', '双色车', '小颜色', '石墨电池', '车辆等级描述', '电池特征']
BATCH_LIMIT = {'小颜色': [15, 30], '双色车': [1, 4], '大颜色': [15, 9999], '石墨电池': [1, 1]}
GAP_LIMIT = {'小颜色': 60, '双色车': 60, '石墨电池': 30}


def update_div_dict():
    div_dict = {'小颜色': [30] * 2000, '大颜色': [30] * 2000, '双色车': [4] * 2000, '石墨电池': [1] * 2000}
    for dividend in range(0, 500):
        for col in ['小颜色', '大颜色', '双色车']:
            low, up = BATCH_LIMIT[col]
            divisor_range = np.arange(low, min(up + 1, 61))
            div, mod = divmod(dividend, divisor_range)
            is_available = np.where((mod == 0) | ((mod >= low) & (mod <= up)), 1, 0)
            div_ceil = div + np.where(mod > 0, 1, 0)
            out = sorted(zip(is_available, div_ceil, mod, divisor_range), key=lambda x: [-x[0], x[1], x[2], -x[3]])
            div_dict[col][dividend] = out[0][-1]
    return div_dict


def flatten(nested_list):
    result = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            result.extend(flatten(sublist))
        else:
            result.append(sublist)
    return result


def prepare_data():
    div_dict = update_div_dict()
    df = pd.read_excel('B榜-工厂智能排产算法赛题 .xlsx', sheet_name='原数据')
    # df = pd.read_csv("A.csv", encoding="gbk")
    df['外色描述BAK'] = df['外色描述'].str.strip('-Y')
    df['小颜色'] = np.where(df['外色描述'].isin(SMALL_COLOR), df['外色描述BAK'], 'other')
    df['大颜色'] = np.where(df['外色描述'].isin(BIG_COLOR), df['外色描述BAK'], 'other')
    df['双色车'] = np.where(df['外色描述'].str.contains('/'), df['外色描述BAK'], 'other')
    df["石墨电池"] = np.where(df["电池特征"].str.contains("石墨"), df["电池特征"], "other")
    df = df.sort_values(by=ATTRIBUTE).reset_index(drop=True)
    df['num'] = df.groupby(ATTRIBUTE)['生产订单号-ERP'].transform('count').values
    df['index'] = df.groupby(ATTRIBUTE)['生产订单号-ERP'].transform('rank').values
    df['split'] = np.ceil(df['index'] / 60)
    for col in ['大颜色', '小颜色', '双色车', '石墨电池']:
        df['split'] = np.where(df[col] != 'other',
                               np.ceil(df['index'] / df['num'].apply(lambda x: div_dict[col][int(x)])),
                               df['split'])
    df['num'] = 1
    return df.drop(['index'], axis=1)


class APS:
    def __init__(self, df):
        self.df = df
        self.attribute = ATTRIBUTE + ["split"]
        self.objective_artificial = {'switch': {'车型': 383, '天窗': 606, '外色描述BAK': 1906, '车辆等级描述': 1395, '电池特征': 1613},
                                     'gap': {'小颜色': 0.398, '双色车': 0.558, '石墨电池': 1.0},
                                     'num': {'小颜色': 0.226, '双色车': 1.0, '大颜色': 0.467, '石墨电池': 1.0}}
        self.group, self.data = {}, {}
        self.car_path, self.car_path_check = self.init_car_path()
        self.path, self.score = self.generate_path()

    def init_car_path(self):
        car_params = self.df.groupby('计划日期')['车型'].unique().to_list()
        car_params = [list(x) for x in car_params]
        n_length = sum([len(x) for x in car_params])
        path = [[] for _ in car_params]
        while sum([len(_) for _ in path]) < n_length:
            paths = []
            score = []
            for date_id, car_list in enumerate(car_params):
                for p in car_list:
                    if p in path[date_id]:
                        continue
                    for j in range(len(path[date_id]) + 1):
                        path_ = path[:date_id] + [path[date_id][:j] + [p] + path[date_id][j:]] + path[date_id + 1:]
                        paths.append(path_)
                        score.append([len(list(groupby(flatten(path_)))), len(path[date_id])])
            out = sorted(zip(score, paths))
            out = list(filterfalse(lambda x: x[0] != out[-1][0], out))
            shuffle(out)
            score, path = out[0]

        while True:
            score_before = len(list(groupby(flatten(path))))
            # single optimize
            for i in range(len(path)):
                if i < len(path) - 1:
                    if path[i][-1] == path[i + 1][0]:
                        continue
                paths = [path[:i] + [list(_)] + path[i + 1:] for _ in permutations(path[i])]
                shuffle(paths)
                score = [len(list(groupby(flatten(raw_list)))) for raw_list in paths]
                path = paths[score.index(min(score))]
            # two optimize
            for i in range(len(path) - 2):
                paths = []
                if (path[i][-1] == path[i + 1][0]) & (path[i + 1][-1] == path[i + 2][0]):
                    continue
                xx = list(permutations(path[i]))
                yy = list(permutations(path[i + 1]))
                for x in xx:
                    for y in yy:
                        if x[-1] == y[0]:
                            paths.append(path[:i] + [list(x)] + [list(y)] + path[i + 2:])
                shuffle(paths)
                score = [len(list(groupby(flatten(raw_list)))) for raw_list in paths]
                path = paths[score.index(min(score))]
            score_after = len(list(groupby(flatten(path))))
            if score_before == score_after:
                break
        return path, [x[-1] for x in path][:-1] + [x[0] for x in path][1:]

    def generate_path(self):
        df = self.df.groupby(self.attribute, as_index=False)["num"].sum()
        for var in self.attribute:
            self.group[var] = np.array(df[var])
            self.data[var] = np.array(df[[var, "num"]])
        path_dict = {date: {} for date in df["计划日期"].unique()}
        for (date, car), g_data in df.groupby(["计划日期", "车型"]):
            path_dict[date][car] = g_data.index.tolist()
        # path = [[path_dict[date][car] for car in self.car_path[date_id]] for date_id, date in enumerate(path_dict)]
        path = [[sample(path_dict[date][car], 1) for car in self.car_path[date_id]] for date_id, date in enumerate(path_dict)]
        score = self.statistic(path)
        while len(flatten(path)) < len(df):
            for date_id, date in enumerate(path_dict):
                if len(flatten(path[date_id])) == sum([len(path_dict[date][car]) for car in path_dict[date]]):
                    continue
                score = []
                paths = []
                for car_id, car in enumerate(self.car_path[date_id]):
                    if len(path[date_id][car_id]) == len(path_dict[date][car]):
                        continue
                    for p in path_dict[date][car]:
                        if p in path[date_id][car_id]:
                            continue
                        # path_ = path[date_id][:car_id] + [path[date_id][car_id] + [p]] + path[date_id][car_id + 1:]
                        # path_ = path[:date_id] + [path_] + path[date_id + 1:]
                        # score.append(self.statistic(path_))
                        # paths.append(path_)
                        for j in range(len(path[date_id][car_id]) + 1):
                            path_ = path[date_id][:car_id] + [path[date_id][car_id][:j] + [p] + path[date_id][car_id][j:]] + path[date_id][car_id + 1:]
                            path_ = path[:date_id] + [path_] + path[date_id + 1:]
                            score.append(self.statistic(path_))
                            paths.append(path_)
                out = sorted(zip(score, paths))
                out = list(filterfalse(lambda x: x[0] != out[-1][0], out))
                shuffle(out)
                if out:
                    score, path = out[-1]
                    break
        return path, score

    def switch_num(self, path, name):
        return len(list(groupby(self.group[name][path]))) - 1

    def batch_num(self, path, name):
        batch_name, batch_num = [], []
        for k, g in groupby(self.data[name][path], key=lambda x: x[0]):
            if k != "other":
                batch_name.append(k)
                batch_num.append(sum([v[1] for v in g]))
        if batch_name:
            batch_num, batch_name = np.array(batch_num), np.array(batch_name)
            low, up = BATCH_LIMIT[name]
            return np.mean((batch_num >= low) & (batch_num <= up))
        else:
            return 1

    def batch_gap(self, path, name):
        batch_name, batch_num, his = [], [], []
        for k, g in groupby(self.data[name][path], key=lambda x: x[0]):
            his.append(batch_name)
            batch_name.append(k)
            batch_num.append(sum([v[1] for v in g]))
        batch_num, batch_name = np.array(batch_num), np.array(batch_name)
        gap = np.sum((batch_num >= GAP_LIMIT[name]) & (batch_name == 'other'))
        num_other1 = np.sum(batch_name == "other")
        num_other2 = len([1 for a, b in pairwise(batch_name) if (a != "other") & (b != "other")])
        return gap / (num_other1 + num_other2)

    def objective(self, path):
        path = flatten(path)
        switch = {key: self.switch_num(path, key) for key in
                  ['车型', '天窗', '外色描述BAK', '车辆等级描述', '电池特征']}
        gap, num = {}, {}
        for key in ['小颜色', '双色车', '大颜色', '石墨电池']:
            num[key] = self.batch_num(path, key)
        for key in ['小颜色', '双色车', '石墨电池']:
            gap[key] = self.batch_gap(path, key)
        objective = {"switch": switch, "gap": gap, "num": num}
        return objective

    def statistic(self, path):
        objective = self.objective(path)
        x = [objective["gap"]['双色车'], objective["gap"]['石墨电池']] + list(objective["num"].values())
        for k1 in objective:
            for k2 in objective[k1]:
                if k1 == 'switch':
                    objective[k1][k2] = 1 - objective[k1][k2] / self.objective_artificial[k1][k2]
                else:
                    objective[k1][k2] = objective[k1][k2] / self.objective_artificial[k1][k2] - 1
        car_path_check = [x[-1][-1] for x in path][:-1] + [x[0][0] for x in path][1:]
        return [list(self.group['车型'][flatten(car_path_check)]) == self.car_path_check,
                objective["switch"]["车型"],
                np.minimum(np.min(x), 0.5),
                4 * objective["switch"]["天窗"] + 2 * objective["switch"]["外色描述BAK"] +
                objective["gap"]['双色车'] + objective["gap"]['石墨电池'] +
                sum(objective["num"].values()),
                objective["switch"]["车辆等级描述"] + objective["switch"]["电池特征"] + objective["gap"]['小颜色']]

    def local_search(self, date_id, car_id):
        path = [0] + self.path[date_id][car_id] + [0]
        n = len(path)
        paths = [path]
        # 2-opt
        for i in range(0, n - 1):
            for j in range(i + 1, n - 1):
                paths.append(path[:i + 1] + path[j:i:-1] + path[j + 1:])
        # 2_h_opt
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                paths.append([path[0]] + [path[i]] + path[1:i] + path[i + 1:j] + path[j + 1:-1] + [path[j]] + [path[-1]])
                paths.append([path[0]] + [path[j]] + path[1:i] + path[i + 1:j] + path[j + 1:-1] + [path[i]] + [path[-1]])
        # relocate_move
        for i in range(1, n - 1):
            for k in range(0, min(n - i - 2, 20)):
                for j in range(i + 1 + k, n):
                    if (i == j) or (j == i - 1):
                        continue
                    if i < j:
                        paths.append(path[:i] + path[i + 1 + k:j] + path[i:i + 1 + k] + path[j:])
                    else:
                        paths.append(path[:j] + path[i:i + 1 + k] + path[j:i] + path[i + 1 + k:])
        # exchange move
        for i in range(1, n - 1):
            for k in range(0, min(n - i - 2, 20)):
                for j in range(i + k + 1, n - k - 1):
                    if i == j:
                        continue
                    paths.append(path[:i] + path[j:j + k + 1] + path[i + k + 1:j] + path[i:i + k + 1] + path[j + k + 1:])
        paths = list(set(tuple(path) for path in paths))
        paths = [self.path[:date_id] + [self.path[date_id][:car_id] + [list(path[1:-1])] + self.path[date_id][car_id + 1:]] + self.path[date_id + 1:] for path in paths]
        return paths

    def optimize(self):
        t1 = time.time()
        for date_id, _ in enumerate(self.path):
            paths = []
            for car_id, _ in enumerate(self.path[date_id]):
                if len(self.path[date_id][car_id]) > 1:
                    paths.extend(self.local_search(date_id, car_id))
            n_car = len(self.path[date_id])
            for car_path in list(permutations(range(n_car))):
                path_ = [self.path[date_id][car_id] for car_id in car_path]
                path_ = self.path[:date_id] + [path_] + self.path[date_id + 1:]
                paths.append(path_)
            for i in range(n_car - 2):
                for j in range(i + 2, n_car):
                    path_opt = self.path[date_id][:i] + [_[::-1] for _ in self.path[date_id][i:j][::-1]] + self.path[date_id][j:]
                    path_opt = self.path[:date_id] + [path_opt] + self.path[date_id + 1:]
                    paths.append(path_opt)
            score = [self.statistic(path) for path in paths]
            out = sorted(zip(score, paths))
            out = list(filterfalse(lambda x: x[0] != out[-1][0], out))
            if len(out) > 1:
                out = list(filterfalse(lambda x: x[1] == self.path, out))
            shuffle(out)
            self.score, self.path = out[-1]
            print(date_id, time.time() - t1, self.score)

    def result(self):
        df_summary = pd.DataFrame(self.group).iloc[flatten(self.path)].reset_index(drop=True)
        df_summary["rank"] = range(len(df_summary))
        df = self.df.merge(df_summary, how="left", left_on=self.attribute, right_on=self.attribute)
        df = df.sort_values(by=["rank", "生产订单号-ERP", '车辆等级描述', '电池特征']).reset_index(drop=True)
        df = df.drop(["rank", "小颜色", "大颜色", "双色车", "石墨电池", "外色描述BAK", "split", "num"], axis=1)
        return df


def main(plan_id):
    start_time = time.time()
    early_stop = 0
    df_order = prepare_data()
    aps = APS(df_order)
    while True:
        path_before, score_before = aps.path, aps.score
        aps.optimize()
        end_time = time.time()
        path_after, score_after = aps.path, aps.score
        if (end_time - start_time > 60 * 60 * 20) | (path_before == path_after):
            break
        if score_before == score_after:
            early_stop += 1
            if early_stop == 10:
                break
        else:
            early_stop = 1
    return score_after, aps.result(), plan_id


def run():
    n = min(os.cpu_count(), 60)
    pool = Pool(n)
    with pool:
        plan_result = pool.map(main, range(n))
    pool.close()
    pool.join()
    plan_result = sorted(plan_result)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    path = f"out/{now}"
    os.makedirs(path)
    os.chdir(path)
    print(plan_result[-1][0])
    plan_result[-1][1].to_csv("commit.csv", index=False)
    zip_file = zipfile.ZipFile('commit.zip', 'w')
    zip_file.write("commit.csv", compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()


if __name__ == '__main__':
    run()
