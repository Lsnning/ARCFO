# -*- coding: utf-8 -*-
import re
import json
from tqdm import tqdm
import utils
from sklearn.metrics import cohen_kappa_score
import concurrent.futures
from json_repair import repair_json

class ScoCrOpt:
    """ ScoCrOpt: Scoring criteria optimization"""

    def __init__(self, args, writing_task, out_dir):
        self.opt = args
        self.writing_task = writing_task
        self.out_log = out_dir
        self.max_tokens = self.opt['max_tokens']
        self.batch_size = self.opt['batch_size']

    def parse_json(self, text):
        try:
            # First remove any outer quotes and unescape if needed
            text = text.strip()
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]

            # Remove any initial newlines/whitespace
            text = text.lstrip()

            try:
                # Try direct JSON parsing first
                return json.loads(text)
            except:
                # If that fails, try to extract JSON content
                match = re.search(r'({.*})', text, re.DOTALL)
                if match:
                    json_text = match.group(1)
                    # Clean up the extracted JSON
                    json_text = json_text.replace('\n', '')
                    json_text = json_text.replace('\t', '')
                    # Remove any escaped quotes
                    json_text = json_text.replace('\\"', '"')
                    json_text = json_text.replace("\\'", "'")

                    try:
                        return json.loads(json_text)
                    except:
                        # Try repairing malformed JSON as last resort
                        try:
                            json_text = repair_json(json_text)
                            return json.loads(json_text)
                        except Exception as e:
                            print(f"Failed to repair JSON: {e}")
                            return []
                return []
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return []

    def parse_res(self, text, key):
        """ Parse the returned json format to list."""
        try:
            json_pred = self.parse_json(text)
            if key:
                json_pred = json_pred[key]
                return json_pred
        except Exception as e:
            print(e)

    def sample_error_essay(self, essays, human_scores, LLM_scores, LLM_comments):
        """ Sample error essays from the given essays, human_scores, and LLM_scores"""
        error_essays, error_human_scores, error_LLM_scores, error_LLM_comments = [], [], [], []
        for i, (l, p) in enumerate(zip(human_scores, LLM_scores)):
            if l != p:
                error_essays.append(essays[i])
                error_human_scores.append(l)
                error_LLM_scores.append(p)
                error_LLM_comments.append(LLM_comments[i])
        return error_essays, error_human_scores, error_LLM_scores, error_LLM_comments

    def _create_batch_scoring_prompt(self, score_rubic, essays):
        """创建批量评分提示"""
        essay_list = "\n\n".join([f"#Essay {i + 1}:\n{essay}" for i, essay in enumerate(essays)])
        ## score_prompt1给set1-6评分
        score_prompt1 = f"""
        You are an experienced English teacher. Please evaluate each essay following this systematic process:
    
        Step 1: For each criterion in the scoring rubric, you must:
        - Quote specific text evidence that relates to this criterion
        - Explain how the evidence demonstrates the criterion
        - Determine the score level at which this criterion is met
    
        Step 2: Consider the overall response:
        - Review all criteria evaluations together
        - Determine if the evidence consistently supports a particular score level
        - Identify any conflicts between criteria scores and resolve them
    
        Step 3: Provide final score and justification that references the specific criteria
        Please return results in this format:
        {{
            "evaluations": [
                {{
                    "essay_number": essay number,
                    "criteria_analysis": [
                        {{
                            "criterion": "Driterion 1",
                            "evidence": ["Quote 1(Brief Overview)", "Quote 2(Brief Overview)"], 
                            "explanation": "How evidence demonstrates criterion",
                            "criterion_level": score for this criterion
                        }},            
                        // Add more criteria as needed
                    ],
                    "score": final score,
                    "justification": "Explain how criteria led to final score"
                }}
                // Add more essays as needed
            ]
        }}

        <writing_task>
        {self.writing_task}
        </writing_task>

        <scoring_criteria>
        {score_rubic}
        </scoring_criteria>

        <essays>
        {essay_list}
        </essays>
        """
        ## score_prompt2给set7和8评分
        score_prompt2 = f"""
        You are an experienced English teacher. Please evaluate each essay following this systematic process:

        Step 1: For each dimension in the scoring rubric, you must:
        - Quote specific text evidence that relates to this dimension
        - Explain how the evidence demonstrates the dimension
        - Determine the score level at which this dimension is met

        Step 2: Consider the overall response:
        - Review all dimension evaluations together
        - Determine if the evidence consistently supports a particular score level
        - Identify any conflicts between dimension scores and resolve them

        Step 3: Provide final score and justification that references the specific dimensions

        Note: Personal identifying information in the essays has been replaced with strings starting with @ (e.g., "@PERSON", "@ORGANIZATION"). This anonymization should not affect scoring.

        Please return results in this format:
        {{
        "evaluations": [
        {{
            "essay_number": essay number,
            "dimension_analysis": [
                {{
                    "dimension": "Dimension 1 name",
                    "evidence": ["Quote 1(Brief Overview)", "Quote 2(Brief Overview)"], 
                    "explanation": "How evidence demonstrates dimension",
                    "dimension_level": score for this dimension
                }},
                // ... analysis for other dimensions
            ]
        }}
        // ... evaluations for other essays
         ]
        }}
        <writing_task>
        {self.writing_task}
        </writing_task>

        <scoring_criteria>
        {score_rubic}
        </scoring_criteria>

        <essays>
        {essay_list}
        </essays>
        """
        # 根据essay_set选择相应的评分提示
        if self.opt['essay_set'] == 7 or self.opt['essay_set'] == 8:
            score_prompt = score_prompt2
        else:
            score_prompt = score_prompt1
        score_prompt = '\n'.join([line.lstrip() for line in score_prompt.split('\n')])
        return score_prompt

    def evaluate_batch(self, batch_essays, batch_human_scores, rubric, batch_idx=0, log=False):
        """单个batch的作文评分
        Args:
            batch_idx: batch的索引
            batch_essays: 该batch中的作文列表
            batch_human_scores: 该batch中的人工评分列表
            rubric: 评分标准
            log: 是否记录日志
        Returns:
            tuple: (batch_llm_scores, analyses) 模型评分和分析结果
        """
        batch_prompt = self._create_batch_scoring_prompt(rubric, batch_essays)
        while True:
            try:
                response = utils.chat(self.opt['model'], batch_prompt,
                                      temperature=self.opt['temperature'],
                                      max_tokens=self.opt['max_tokens'])[0]
                evaluations = self.parse_res(response, 'evaluations')

                if len(evaluations) != len(batch_essays):
                    raise ValueError("评估数量与文章数量不匹配")

                # 处理不同essay_set的评分逻辑
                if self.opt['essay_set'] in [7, 8]:
                    dimensions_scores = []
                    analyses = []
                    for evaluation in evaluations:
                        dimensions_score = []
                        analyse = []
                        for dimension_eval in evaluation['dimension_analysis']:
                            dimensions_score.append(dimension_eval['dimension_level'])
                            # 构建分析内容
                            analysis = {
                                'dimension': dimension_eval['dimension'],
                                'evidence': dimension_eval['evidence'],
                                'explanation': dimension_eval['explanation']
                            }
                            analyse.append(analysis)
                        dimensions_scores.append(dimensions_score)
                        analyses.append(analyse)
                    if self.opt['essay_set'] == 7:
                        batch_llm_scores = [sum(scores) * 2 for scores in dimensions_scores]
                    else:  # essay_set == 8
                        batch_llm_scores = [sum(scores) * 2 + 2 * scores[-1] for scores in dimensions_scores]
                else:
                    batch_llm_scores, analyses = self.parse_scores_and_analysis(evaluations)
                # 记录日志
                if log:
                    with open(self.out_log, 'a', encoding='utf-8') as file:
                        start_idx = batch_idx * self.batch_size
                        for i, essay in enumerate(batch_essays):
                            print(f"Essay_{start_idx + i + 1}:", file=file)
                            print(f"Essay: {essay}", file=file)
                            print(f"Human Score: {batch_human_scores[i]}", file=file)
                            print(f"LLM Score: {batch_llm_scores[i]}", file=file)
                            print(f"Analysis: {analyses[i]}\n\n", file=file)
                            if self.opt['essay_set'] in [7, 8]:
                                print(f"Dimension Scores: {dimensions_scores[i]}\n\n", file=file)
                return batch_llm_scores, analyses
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

    def evaluate(self, rubric, essays, human_scores, log=False):
        """评估多篇作文并计算 kappa 值，使用线程池处理多个batch
        Args:
            rubric (dict): 评分标准
            essays (list): 作文列表
            human_scores (list): 人工评分列表
            log (bool): 是否记录日志
        Returns:
            tuple: (kappa, essays, human_scores, llm_scores, analyses)
        """
        essays = list(essays)
        human_scores = list(human_scores)

        # 将essays按batch_size分组
        batches = [essays[i:i + self.batch_size] for i in range(0, len(essays), self.batch_size)]
        human_scores_batches = [human_scores[i:i + self.batch_size] for i in
                                range(0, len(human_scores), self.batch_size)]
        # 用于存储所有结果
        all_results = [None] * len(batches)
        # 使用线程池处理多个batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有batch任务
            future_to_batch = {
                executor.submit(
                    self.evaluate_batch,
                    batch_essays,
                    batch_human_scores,
                    rubric,
                    batch_idx,
                    log
                ): batch_idx
                for batch_idx, (batch_essays, batch_human_scores)
                in enumerate(zip(batches, human_scores_batches))
            }
            # 使用tqdm显示进度
            with tqdm(
                    concurrent.futures.as_completed(future_to_batch),
                    total=len(batches),
                    desc='Processing batches'
            ) as pbar:
                for future in pbar:
                    batch_idx = future_to_batch[future]
                    try:
                        batch_llm_scores, batch_analyses = future.result()
                        all_results[batch_idx] = (batch_llm_scores, batch_analyses)
                    except Exception as e:
                        print(f'Batch {batch_idx} generated an exception: {e}')
                        # 重试失败的batch
                        while True:
                            try:
                                batch_llm_scores, batch_analyses = self.evaluate_batch(
                                    batches[batch_idx],
                                    human_scores_batches[batch_idx],
                                    rubric,
                                    batch_idx,
                                    log
                                )
                                all_results[batch_idx] = (batch_llm_scores, batch_analyses)
                                break
                            except Exception as e:
                                print(f'Retry failed for batch {batch_idx}: {e}')


        # 合并所有结果
        all_llm_scores = []
        all_analyses = []
        for batch_result in all_results:
            batch_llm_scores, batch_analyses = batch_result
            all_llm_scores.extend(batch_llm_scores)
            all_analyses.extend(batch_analyses)
        # 计算整体kappa值
        kappa = cohen_kappa_score(human_scores, all_llm_scores, weights='quadratic')
        return kappa, essays, human_scores, all_llm_scores, all_analyses

    def cal_loss(self, score_rubic, essays, human_scores, llm_scores, analyses):
        """ Analyze the possible difference between the scoring rubric drawn up by the LLM and that of a human teacher. And then generate optimization suggestions for the scoring rubric. """
        scoring_results = '\n'.join([
                                        f'Essay {idx + 1}:\nEssay：{essay}\nHuman_score: {human_scores[idx]}\nLLM_score：{llm_scores[idx]}\nLLM_analyses：{analyses[idx]}\n'
                                        for idx, essay in enumerate(essays)])
        loss_prompt = f"""
        Conduct an in-depth analysis of the LLM's essay scoring results based on the scoring criteria and provide suggestions for optimizing the scoring standards. The optimization objective is to enhance the objectivity and consistency of LLM scoring by developing precise scoring criteria, reducing subjective arbitrariness, and providing clearer and more actionable standards for future LLM scoring. The analysis will focus on the following aspects:
        1.Blurred Scoring Boundary Analysis
            (1) Identify fuzzy areas between different score levels
            (2) Locate scoring points lacking clear indicators
            (3) Discover descriptions that may lead to ambiguity
        2. Optimization of Scoring Criteria
            (1) Add specific scoring points for each score level
            (2) Supplement qualitative descriptions and quantitative indicators
            (3) Provide typical and boundary case explanations
        Please provide your analysis and suggestions in the following JSON format:
            {{
            "analysis": {{
                "unclear_boundaries": [
                    {{
                        "score_range": "Current score range being analyzed",
                        "current_description": "Original description of current scoring criteria",
                        "identified_issues": [
                            "Specific fuzzy boundary issue 1",
                            "Specific fuzzy boundary issue 2",
                            ...
                        ],
                        "impact_examples": [
                            "Example 1 of potential scoring inconsistency caused by blurred boundaries",
                            "Example 2 of potential scoring inconsistency caused by blurred boundaries",
                            ...
                        ]
                    }},
                    //... other potentially ambiguous score ranges
                ],
                "missing_indicators": [
                    {{
                        "scoring_point": "Scoring dimension requiring additional indicators",
                        "current_state": "Current state description of the scoring dimension",
                        "needed_indicators": [
                            "Suggested specific indicator 1",
                            "Suggested specific indicator 2",
                            ...
                        ],
                        "example_cases": [
                            "Example 1 illustrating scoring problems due to missing indicators",
                            "Example 2 illustrating scoring problems due to missing indicators",
                            ...
                        ]
                    }},
                    //... other missing indicators scoring points
                ],
                "ambiguous_descriptions": [
                    {{
                        "criteria_text": "Scoring criteria description with potential ambiguity",
                        "interpretation_issues": [
                            "Possible misinterpretation or ambiguous explanation 1",
                            "Possible misinterpretation or ambiguous explanation 2",
                            ...
                        ],
                        "problematic_cases": [
                            "Specific scoring case causing ambiguity 1",
                            "Specific scoring case causing ambiguity 2",
                            ...
                        ]
                    }},
                    // ... other ambiguous descriptions
                ]
            }},
            "refinement_suggestions": [
                    {{
                        "score_level": "Score level requiring optimization",
                        "current_criteria": {{
                            "points": ["Current scoring point 1", "Current scoring point 2", ...],
                            "indicators": ["Current scoring indicator"],
                            "examples": ["Current scoring example"]
                        }},
                        "suggested_change": {{
                            "point": 
                                {{
                                    "current_text": "Current scoring point description",
                                    "refined_text": "Improved description",
                                    "rationale": "Reason for modification"
                                }},
                            "indicator":
                                {{
                                    "aspect": "Scoring dimension",
                                    "proposed": [
                                        "Suggested specific indicator 1",
                                        "Suggested specific indicator 2",
                                        ...
                                    ],
                                    "measurement": "Description of assessment method"
                                }},
                            "example":
                                {{
                                    "case_type": "Case type (typical/boundary)",
                                    "description": "Specific case description",
                                    "key_features": [
                                        "Case key feature 1",
                                        "Case key feature 2"
                                    ],
                                    "scoring_explanation": "Scoring explanation"
                                }}
                        }},
                        "expected_impact": "Description of expected improvement effect"
                    }}
                // ...other potentially optimizable score levels
                ]
        }}

        <writing task>
        {self.writing_task}
        </writing task>

        <scoring criteria>
        {score_rubic}
        </scoring criteria>

        <scoring results>
        {scoring_results}
        </scoring results>
        """

        while True:
            try:
                res = utils.chat(self.opt['model'], loss_prompt,
                                 temperature=self.opt['temperature'],
                                 max_tokens=self.opt['max_tokens'])[0]
                res = self.parse_json(res)

                # Log the detailed analysis results
                with open(self.out_log, 'a', encoding='utf-8') as file:
                    # Handle quantitative analysis
                    if 'analysis' in res:
                        print(f'分析评分标准中需要改进的方面:\n {res["analysis"]}', file=file)

                    suggestions = []
                    if 'refinement_suggestions' in res:
                        for refinement in res['refinement_suggestions']:
                            suggestion = {
                                "score_level": refinement["score_level"],
                                "current_text": refinement["suggested_change"]['point']['current_text'],
                                "refined_text": refinement["suggested_change"]['point']['refined_text'],
                                "aspect": refinement["suggested_change"]['indicator']['aspect'],
                                "proposed": refinement["suggested_change"]['indicator']['proposed'],
                                "measurement": refinement["suggested_change"]['indicator']['measurement'],
                                "scoring_explanation": refinement["suggested_change"]['example']['scoring_explanation']
                            }
                            suggestions.append(suggestion)

                    suggestions_text = '\n'.join([
                        f"Suggestion {i + 1}: {{\n" +
                        f"  score_level: {sug['score_level']}\n" +
                        f"  current_text: {sug['current_text']}\n" +
                        f"  refined_text: {sug['refined_text']}\n" +
                        f"  aspect: {sug['aspect']}\n" +
                        f"  proposed: {sug['proposed']}\n" +
                        f"  measurement: {sug['measurement']}\n" +
                        f"  scoring_explanation: {sug['scoring_explanation']}\n}}"
                        for i, sug in enumerate(suggestions)
                    ])
                    print(f'评分等级修改建议：\n {res["refinement_suggestions"]}', file=file)
                break
            except Exception as e:
                print(f"Error in processing loss analysis: {e}")
                continue
        return suggestions_text

    def update_rubric(self, score_rubic, suggestions):
        update_prompt = f"""    
        Update the scoring criteria following these strict requirements:
     1. For each proposed modification:
       a) Check that new wording maintains original scoring intent
       b) Ensure changes improve clarity without altering core requirements

    2. For each criterion being modified:
       a) Keep all unmodified aspects exactly as they are
       b) Maintain consistent language style and structure
       c) Preserve scoring level distinctions

    3. After each modification:
       a) Verify no unintended changes to other criteria
       b) Check that scoring levels remain clearly distinguished

    Your response must follow this JSON format:
    {{
        "updated_criteria": "complete updated scoring criteria text",
        "alignment_check": [
            {{
                "criterion": "Criterion identifier",
                "original_text": "Original text",
                "modified_text": "Modified text",
                "level_distinctions": "Explanation of how scoring levels remain distinctly separated",
                "intent_preservation": "Whether the original scoring intent is maintained",
                "clarity_improvement": "Whether the standard's clarity has been enhanced"
            }}
        ],
        "validation_summary": {{
            "total_criteria": "Total number of scoring criteria",
            "modified_criteria": "Number of criteria modified",
            "validation_status": "Validation result (Pass/Fail)"
        }},
        "modification_rationale": "Overall rationale and purpose of the modifications",
        "quality_metrics": {{
            "precision": "Numerical score indicating modification precision",
            "comprehensiveness": "Extent of criteria coverage after modification",
            "consistency": "Degree of maintaining original scoring framework"
        }}
    }}

        <current_scoring_criteria>
        {score_rubic}
        </current_scoring_criteria>

        <modification_suggestions>
        {suggestions}
        </modification_suggestions>
        """
        update_prompt = '\n'.join([line.lstrip() for line in update_prompt.split('\n')])
        while True:
            try:
                updated_rubric = utils.chat(self.opt['model'], update_prompt,
                                            temperature=self.opt['temperature'],
                                            max_tokens=self.opt['max_tokens'])[0]
                updated_rubric = self.parse_res(updated_rubric, 'updated_criteria')
                break
            except Exception as e:
                print(e)
        with open(self.out_log, 'a', encoding='utf-8') as file:
            print(f'提供的修改建议：\n {suggestions}', file=file)
            print(f'改进后的评分标准：\n{updated_rubric}', file=file)
        return updated_rubric

    def parse_scores_and_analysis(self, evaluations):
        """Parse scores and analysis from the new structured evaluation format."""
        batch_llm_scores = []
        analyses = []

        for eval in evaluations:
            # Extract criterion scores
            if ('criteria_analysis' in eval) or ('criterion_analysis' in eval):
                # New format
                final_score = eval['score']
                analysis = f"Justification: {eval['justification']}\n"
                for criterion in eval['criteria_analysis']:
                    analysis += f"\nCriterion: {criterion.get('criterion', 'Unknown')}\n"
                    analysis += f"Evidence: {criterion.get('evidence', 'No evidence')}\n"
                    analysis += f"Explanation: {criterion.get('explanation', 'No explanation')}\n"
                    analysis += f"Criterion level: {criterion.get('criterion_level', 'No criterion level')}\n"
            else:
                # Legacy format compatibility
                final_score = eval['score']
                analysis = eval['analysis']

            batch_llm_scores.append(int(final_score))
            analyses.append(analysis)

        return batch_llm_scores, analyses

