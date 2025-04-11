# -*- coding: utf-8 -*-
# 增加输出验证集和测试集的评分结果，修改训练集、验证集和测试集的划分
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from getData import GetData
import optimizers
import utils
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ASAP')
    parser.add_argument('--essay_set', default=1, type=int)
    parser.add_argument('--model', default='gpt-4o-mini')  # "gpt-4o-mini"/"glm-4-air"/"deepseek-v3"/...
    parser.add_argument('--temperature', default=0.0, type=float)   # LLM的温度
    parser.add_argument('--batch_size', default=4, type=int)  # 每次迭代处理的作文数量
    parser.add_argument('--max_tokens', default=7000, type=int)  # LLM单次对话的最大token数
    args = parser.parse_args()
    return args


def plot_line_chart(data_list, save_path):
    """
    绘制折线图并保存
    参数:
    data_list: 列表，包含要绘制的数据
    save_path: 字符串，图片保存路径（例如：'path/to/chart.png'）
    """
    try:
        # 创建x轴数据
        x = np.arange(len(data_list))

        # 设置全局字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.unicode_minus'] = False
        # 创建图形
        plt.figure(figsize=(12, 6))
        # 绘制折线图
        plt.plot(x, data_list, marker='o', linewidth=2, markersize=8)
        # 设置标题和轴标签（使用相同的字体大小）
        plt.xlabel('Batch', fontsize=20)
        plt.ylabel('Improvement(%)', fontsize=20)
        # 设置刻度标签的字体大小
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        # 调整图表边距
        plt.tight_layout()
        # 保存图形
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存
        print(f"图表已保存至: {save_path}")
    except Exception as e:
        print(f"绘制图表时出错: {str(e)}")

def optimize_scoring_criteria(train_data, val_data, initial_rubric, optimizer, outDir):
    """
    优化评分标准的主要流程

    Args:
        train_data: 训练集数据
        val_data: 验证集数据
        initial_rubric: 初始评分标准
        optimizer: 优化器
        outDir: 输出文件路径

    Returns:
        tuple: (best_rubric, best_val_kappa) 最优评分标准和对应的验证集kappa值
    """

    logOutDir = os.path.join(outDir, 'log.txt')
    logValDir = os.path.join(outDir, 'valdata_score.xlsx')
    figDir = os.path.join(outDir, 'optimization_curve.jpg')

    # 对验证集进行初始评分
    val_kappa, val_essays, val_human_scores, val_llm_scores, val_comments = optimizer.evaluate(
        initial_rubric,
        val_data['essay'],
        val_data['domain1_score'],
        log=False
    )

    # 创建验证集结果DataFrame并保存
    val_results_df = pd.DataFrame({
        'essay': val_essays,
        'human_score': val_human_scores,
        'llm_score_batch_0': val_llm_scores,
        'comments_batch_0': val_comments
    })
    val_results_df.to_excel(logValDir, index=False)
    with open(logOutDir, 'a', encoding='utf-8') as outf:
        outf.write(f'初始验证集Kappa值：{val_kappa}\n')

    # 记录迭代优化过程中的val_kappa、相较于初始的验证性能(kappa)提升的百分比、最佳的评分标准、batch_index
    val_kappas = [val_kappa]
    perfBoost = [0]

    rubric = initial_rubric
    batch_index = 0

    # 开始优化过程，重置token计数并记录开始时间
    utils.optimization_tokens = 0  # 重置优化过程的token计数
    opt_start_time = datetime.now()

    with tqdm(enumerate(train_data), total=len(train_data), file=sys.stdout, desc='Training') as pbar:  ############
        for i, train_batch in pbar:
            # print(f'\n第{i+1}个batch的评分标准：\n{rubric}')
            with open(logOutDir, 'a', encoding='utf-8') as outf:
                outf.write(f"\n========== batch {i+1} ==========\n")
                outf.write(f'评分标准：\n{rubric}\n')
                outf.write(f'\n1.训练集评分\n')

            # 对当前batch进行评分
            print("\n1.训练集评分...")
            batch_llm_scores, batch_comments = optimizer.evaluate_batch(
                train_batch['essay'].tolist(),
                train_batch['domain1_score'].tolist(),
                rubric,
                log=True)
            # 获取评分错误的作文
            error_text, error_human_scores, error_llm_scores, error_comments = optimizer.sample_error_essay(
                train_batch['essay'].tolist(),
                train_batch['domain1_score'].tolist(),
                batch_llm_scores,
                batch_comments)

            if error_text:
                print("2.识别问题并提出改进评分标准的建议...")
                # 分析整个batch的评分差异并获取改进建议
                with open(logOutDir, 'a', encoding='utf-8') as outf:
                    outf.write(f'\n2.识别问题并提出改进评分标准的建议\n')
                suggestions = optimizer.cal_loss(
                    rubric,
                    error_text,
                    error_human_scores,
                    error_llm_scores,
                    error_comments)
                # 更新评分标准并测试
                print("3.更新评分标准...")
                with open(logOutDir, 'a', encoding='utf-8') as outf:
                    outf.write(f'\n3.更新评分标准\n')
                updated_rubric = optimizer.update_rubric(rubric, suggestions)
                # 使用更新后的评分标准评估验证集
                print("4.验证更新后的评分标准...")
                updated_val_kappa, _, _, val_llm_scores, val_comments = optimizer.evaluate(
                    updated_rubric,
                    val_data['essay'],
                    val_data['domain1_score'],
                    log=False
                )
                # 保存本次验证结果
                val_results_df[f'llm_score_batch_{i + 1}'] = val_llm_scores
                val_results_df[f'comments_batch_{i + 1}'] = val_comments
                val_results_df.to_excel(logValDir, index=False)
                val_kappas.append(updated_val_kappa)
                perfBoost.append((updated_val_kappa - val_kappas[0])/val_kappas[0]*100)
                # 如果验证集效果更好，则更新最佳结果
                if updated_val_kappa > val_kappas[batch_index]:
                    with open(logOutDir, 'a', encoding='utf-8') as outf:
                        outf.write(f'验证Kappa值：{updated_val_kappa}，比初始提升{perfBoost[-1]}%，大于当前最好的Kappa值{val_kappas[batch_index]}，因此该更新后的评分标准将被采纳。\n')
                    rubric = updated_rubric
                    batch_index = i + 1
                else:
                    with open(logOutDir, 'a', encoding='utf-8') as outf:
                        outf.write(f'验证Kappa值：{updated_val_kappa}，比初始提升{perfBoost[-1]}%，小于当前最好的kappa值{val_kappas[batch_index]}，因此该更新后的评分标准将不被采纳。\n')
            else:  # 如果没有评分错误的作文，则直接使用更新后的评分标准评估验证集
                with open(logOutDir, 'a', encoding='utf-8') as outf:
                    outf.write(f'\n2.没有识别到评分错误的作文，未更新评分标准\n')
                val_results_df[f'llm_score_batch_{i + 1}'] = val_results_df[f'llm_score_batch_{batch_index}']
                val_results_df[f'comments_batch_{i + 1}'] = val_results_df[f'comments_batch_{batch_index}']
                val_results_df.to_excel(logValDir, index=False)
                val_kappas.append(val_kappas[batch_index])
                perfBoost.append(perfBoost[batch_index])

    with open(logOutDir, 'a', encoding='utf-8') as outf:
        outf.write(f"\n======== 迭代优化结束 ========\n")
        outf.write(f'每个batch更新完评分标准的验证结果：{val_kappas}\n')
        outf.write(f'每个batch更新完评分标准的验证结果相较于初始性能提升的百分比：{perfBoost}\n')
        outf.write(f'最优的评分标准：\n{rubric}\n')
        outf.write(f"优化总时长: {(datetime.now()-opt_start_time).seconds}s\n")
        outf.write(f"优化过程消耗tokens: {utils.optimization_tokens}\n")
    # 绘制迭代过程折线图
    plot_line_chart(perfBoost, figDir)
    return rubric


if __name__ == '__main__':
    args = get_args()   # 实验参数
    config = vars(args)
    print(config)

    outDir = os.path.join('out7', f'essay_set_{args.essay_set}', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # 输出日志和评分结果的路径
    logOutDir = os.path.join(outDir, 'log.txt')
    logValDir = os.path.join(outDir, 'valdata_score.xlsx')
    logTestDir = os.path.join(outDir, 'testdata_score.xlsx')

    with open(logOutDir, 'a', encoding='utf-8') as outf:
        outf.write(json.dumps(config) + '\n')

    # 获取作文数据集及其相关信息
    data = GetData(args.dataset, args.essay_set)
    train_set, val_set, test_set = data.get_dataset(args.batch_size)
    writing_task = data.get_writing_task()
    rubric = data.get_rubric()
    optimizer = optimizers.ScoCrOpt(config, writing_task, logOutDir)

    # 对测试集进行初始评分，并保存测试集的评分结果到excel
    print("对测试集进行初始评分...")
    test_kappa, test_essays, test_human_scores, test_llm_scores, test_comments = optimizer.evaluate(
        rubric,
        test_set['essay'],
        test_set['domain1_score'],
        log=False
    )
    test_results_df = pd.DataFrame({
        'essay': test_essays,
        'human_score': test_human_scores,
        'initial_llm_score': test_llm_scores,
        'initial_comments': test_comments
    })
    test_results_df.to_excel(logTestDir, index=False)
    print(f"初始测试集Kappa值: {test_kappa}")
    with open(logOutDir, 'a', encoding='utf-8') as outf:
        outf.write(f'初始测试集Kappa值：{test_kappa}\n')


    # 开始迭代优化
    print("开始迭代优化...")
    optimized_rubric = optimize_scoring_criteria(
        train_set,
        val_set,
        rubric,
        optimizer,
        outDir
    )

    with open(os.path.join(outDir, 'optimized_rubric.txt'), 'w', encoding='utf-8') as file:
        file.write(str(optimized_rubric))


    # 使用最优评分标准对测试集进行评分
    print("对测试集进行最终评分...")
    final_test_kappa, _, _, final_test_scores, final_test_comments = optimizer.evaluate(
        rubric,
        test_set['essay'],
        test_set['domain1_score'],
        log=False
    )
    # 保存测试集结果
    test_results_df['final_llm_score'] = final_test_scores
    test_results_df['final_comments'] = final_test_comments
    test_results_df.to_excel(logTestDir, index=False)

    with open(logOutDir, 'a', encoding='utf-8') as outf:
        outf.write(
            f'初始测试集Kappa值: {test_kappa}，最终测试集kappa值：{final_test_kappa}，提升: {final_test_kappa - test_kappa}\n')


    print("DONE!")

