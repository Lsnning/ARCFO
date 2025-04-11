# -*- coding: utf-8 -*-
import time
import requests
import config
import traceback

optimization_tokens = 0  # 专门用于统计优化过程的token消耗

def chat(model, prompt, temperature=0, n=1, top_p=1, max_tokens=4095, timeout=300):
    if 'gpt' in model:
        url = 'https://api.openai.com/v1/chat/completions'
        key = config.GPT_KEY
    elif 'glm' in model:
        url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
        key = config.ZHIPU_KEY
    elif 'deepseek-chat' in model:
        url = 'https://api.deepseek.com/v1'
        key = config.DEEPSEEK_KEY
    else:
        raise ValueError(f"Unknown model: {model}")

    global optimization_tokens
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    retries = 0
    while True:
        try:
            r = requests.post(url,
                              headers={
                                  "Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"
                              },
                              json=payload,
                              timeout=timeout
                              )
            if r.status_code != 200:
                retries += 1
                time.sleep(1)
                if retries >= 10:
                    with open('error-log.txt', 'a', encoding='utf-8') as outf:
                        outf.write(f"\n\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                        outf.write(f"response:{r}, {r.reason}")
                        outf.write(f"prompt:\n{prompt}")
                    # exit()
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
            if retries >= 10:
                with open('error-log.txt', 'a', encoding='utf-8') as outf:
                    outf.write(f"\n\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                    outf.write(f"超时未响应！")
                    outf.write(f"prompt:\n{prompt}")
                # exit()
    r = r.json()
    tokens_used = r['usage']['total_tokens']

    # 检查调用栈以确定是否在优化过程中
    stack = traceback.extract_stack()
    is_optimization = any('optimize_scoring_criteria' in frame.name for frame in stack)
    if is_optimization:
        optimization_tokens += tokens_used
    return [choice['message']['content'] for choice in r['choices']]

if __name__ == '__main__':
    r = chat("gpt-4o-mini", "hello！")
    print(r)


