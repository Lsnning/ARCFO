import os
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class GetData:
    def __init__(self, dataset, essay_set_value):
        self.dataset = dataset
        self.essay_set_value = essay_set_value

        # 设置随机种子
        np.random.seed(42)  # 控制NumPy的随机操作
        random.seed(42)  # 控制Python标准库的随机操作

    def get_writing_task(self):
        # 根据essay_set_value返回相应的writing_task
        file_path = os.path.join('data', self.dataset, 'prompts', f'Essaay Set #{self.essay_set_value}.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            writing_task = file.read()
        return writing_task

    def get_rubric(self):
        # 根据essay_set_value返回相应的rubric
        file_path = os.path.join('data', self.dataset, 'scoring_rubric', f'Essaay Set #{self.essay_set_value}.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            rubric = file.read()
        return rubric

    def create_score_levels(self, df, batch_size, score_col='domain1_score'):
        """
        为数据集添加分数等级，根据分数实际范围划分为2个等级

        参数:
        df: DataFrame, 包含分数的数据集
        batch_size: int, 划分的等级数量
        score_col: str, 分数列名

        返回:
        DataFrame: 添加了score_level列的数据集
        """
        df = df.copy()
        min_score = df[score_col].min()
        max_score = df[score_col].max()
        score_range = max_score - min_score
        # 计算每个区间的大小
        interval = score_range / batch_size

        def get_score_level(score, batch_size):
            # 计算每个区间的大小
            for i in range(batch_size):
                if score <= min_score + (i + 1) * interval:
                    return f'level_{i + 1}'
            return f'level_{batch_size}'  # 处理最大值的情况

        df['score_level'] = df[score_col].apply(lambda x: get_score_level(x, batch_size))
        return df

    def create_batch(self, data, batch_size):
        """
        从当前数据中创建一个固定大小的batch

        参数:
        data: DataFrame, 当前可用的训练数据
        batch_size: int, 期望的batch大小(固定为4)

        返回:
        tuple: (batch样本, 剩余样本)
        """
        if len(data) < batch_size:
            return None, None

        # 为当前数据添加分数等级
        leveled_data = self.create_score_levels(data, batch_size)

        # 获取当前的等级分布
        level_counts = leveled_data['score_level'].value_counts()

        # 初始化选中的样本列表
        selected_samples = []
        remaining_data = leveled_data.copy()

        while len(selected_samples) < batch_size and not remaining_data.empty:
            # 获取当前剩余数据的等级分布
            current_level_counts = remaining_data['score_level'].value_counts()

            if len(selected_samples) == 0:
                # 第一个样本：优先选择样本数最多的等级
                selected_level = current_level_counts.index[0]
            else:
                # 后续样本：尽可能选择不同等级
                # 获取已选样本的等级
                used_levels = set([s['score_level'].iloc[0] for s in selected_samples])
                # 获取未使用的等级
                available_levels = set(current_level_counts.index) - used_levels

                if available_levels:
                    # 如果还有未使用的等级，优先使用
                    selected_level = list(available_levels)[0]
                else:
                    # 如果所有等级都已使用，选择样本数最多的等级
                    selected_level = current_level_counts.index[0]

            # 从选中的等级中随机抽取一个样本
            level_samples = remaining_data[remaining_data['score_level'] == selected_level]
            selected_sample = level_samples.sample(n=1, random_state=42)
            selected_samples.append(selected_sample)

            # 从剩余数据中移除已选样本
            remaining_data = remaining_data.drop(selected_sample.index)

        if len(selected_samples) == batch_size:
            batch = pd.concat(selected_samples)
            # 移除score_level列
            batch = batch.drop('score_level', axis=1)
            remaining_data = remaining_data.drop('score_level', axis=1)
            return batch, remaining_data

        return None, None

    def get_dataset(self, batch_size):
        # 读取Excel数据集文件

        train_data = pd.read_excel(os.path.join('data', 'ASAP', 'dataset', f'essay_set_{self.essay_set_value}', 'train_data.xlsx'))
        val_data = pd.read_excel(os.path.join('data', 'ASAP', 'dataset', f'essay_set_{self.essay_set_value}', 'val_data.xlsx'))
        test_data = pd.read_excel(os.path.join('data', 'ASAP', 'dataset', f'essay_set_{self.essay_set_value}', 'test_data.xlsx'))

        # 创建训练集batches
        train_batches = []
        remaining_train_data = train_data.copy()
        while True:
            batch, remaining_train_data = self.create_batch(remaining_train_data, batch_size)
            if batch is None:
                break
            train_batches.append(batch)

        return train_batches, val_data, test_data


