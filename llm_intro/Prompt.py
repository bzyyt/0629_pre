import json
import math
import os
import pickle
import re
import time
import traceback
import warnings

import gradio as gr
import jinja2
import openai
import tiktoken  # 用于 prompt_token_num()
from dotenv import load_dotenv
from jinja2 import meta

load_dotenv()

# 所有请求共用同一模型配置；首次请求时才初始化客户端。
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
client: openai.OpenAI | None = None


def get_client() -> openai.OpenAI:
    global client
    if client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        if not api_key or not base_url or not MODEL_NAME:
            raise ValueError(
                "请配置 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL 和有效的模型名称"
            )
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return client


def check_api_settings():
    try:
        get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "测试"}],
            max_tokens=1,
        )
        print("API 设置成功！！")
    except Exception as e:
        print(f"API 可能有问题，请检查：{e}")


class OpenAIModel:
    """
    封装OpenAI API调用和缓存机制的类。

    用于调用OpenAI API，处理响应，并缓存结果以提高效率。

    属性:
        cache_file (str): 缓存文件的路径
        cache_dict (dict): 内存中的缓存字典
    """

    def __init__(self, cache_file="openai_cache"):
        """
        初始化OpenAI模型对象，设置缓存文件路径并加载缓存。

        参数:
            cache_file (str): 缓存文件的路径，默认为"openai_cache"
        """
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()  # 加载缓存

    def save_cache(self):
        """
        将当前缓存保存到文件中。
        """
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_dict, f)
        except OSError as e:
            warnings.warn(f"缓存保存失败，评估结果仍保留在内存中：{e}", stacklevel=2)

    def load_cache(self):
        """
        从文件加载缓存。损坏或不可读时提示错误并使用空缓存，不无限重试。

        返回:
            dict: 加载的缓存字典，如果文件不存在则返回空字典
        """
        try:
            with open(self.cache_file, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict):
                raise ValueError("缓存内容必须是字典")
            return cache
        except FileNotFoundError:
            return {}
        except Exception as e:
            warnings.warn(f"缓存读取失败，将使用空缓存继续：{e}", stacklevel=2)
            return {}

    def set_cache_file(self, file_name):
        """
        设置缓存文件名并重新加载缓存。

        参数:
            file_name (str): 新的缓存文件路径
        """
        self.cache_file = file_name
        self.cache_dict = self.load_cache()

    def get_response(self, content):
        """
        获取模型完成的文本。每次重新采样，并将最新回复记录到缓存。

        参数:
            content (str): 提供给模型的输入内容

        返回:
            str: 模型生成的回复文本，如果出错则返回None
        """
        # 如果选择检查缓存，则会导致同问题不同trial的结果相同，这与实际想表达的内容不符，故注释
        # if content in self.cache_dict:
        #     return self.cache_dict[content]
        for _ in range(3):  # 尝试三次
            try:
                # 调用模型生成内容
                response = get_client().chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": content}],
                    temperature=1.0,
                )
                completion = response.choices[0].message.content
                self.cache_dict[content] = completion
                return completion
            except Exception as e:
                print(e, "\n")
                time.sleep(1)
        return None

    def is_valid_key(self):
        """
        检查API密钥是否有效。

        返回:
            bool: 如果API密钥有效则返回True，否则返回False
        """
        for _ in range(4):  # 尝试四次
            try:
                get_client().chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": "hi there"}],
                    temperature=1.0,
                    max_tokens=1,
                )
                return True
            except Exception:
                traceback.print_exc()
                time.sleep(1)
        return False

    def prompt_token_num(self, prompt):
        """
        计算prompt的token数量。

        参数:
            prompt (str): 要计算token数量的prompt

        返回:
            int: token的数量
        """
        try:
            # 仅为长度估算，不代表服务端模型实际使用的 Token 数。
            encoding = tiktoken.get_encoding(
                "cl100k_base"
            )  # 这是 GPT-3.5-turbo 所使用的编码器
            # 将 prompt 编码成 token，并返回 token 数量
            tokens = encoding.encode(prompt)
            return len(tokens)
        except Exception as e:
            print(f"计算 token 数量时出错: {e}")
            return None

    def two_stage_completion(self, question, content):
        """
        两阶段完成：首先获取推理（注意，这里并非推理模型的思维链，而是直接输入 content 后得到的回复），再获取最终答案。

        参数:
            question (str): 原始问题
            content (str): 提供给模型的输入内容

        返回:
            dict: 包含prompt、推理过程和答案的字典
        """
        rationale = self.get_response(content)
        if not rationale:
            return {"prompt": content, "rationale": None, "answer": None}

        ans = self.get_response(
            content=f"Q:{question}\nA:{rationale}\nThe answer to the original question is (a number only): "
        )
        return {"prompt": content, "rationale": rationale, "answer": ans}


# 初始化模型
my_model = OpenAIModel()


questions = [
    "一位艺术家使用红色和蓝色瓷砖制作马赛克。蓝色瓷砖的数量恰好是红色瓷砖的3倍。如果他使用了57块红色瓷砖，那么整个马赛克共使用多少块瓷砖？",
    "一位农民正在为当地市场装苹果。他有120个苹果，并希望将它们均匀分配到篮子中。如果他决定留15个苹果给家人，每个篮子最多能装7个苹果，那么他最少需要多少个篮子才能将苹果带到市场？",
    "一个花园有5块矩形地块，排列成一条直线。中间3块地的面积各为24平方米，所有地块的宽度均为4米。第一块地的长度是中间一块地的两倍，最后一块地的长度是中间一块地的一半。那么所有地块的总面积是多少平方米？",
    "一个农贸市场出售两种类型的苹果混合袋：A型袋子包含4个红苹果和6个绿苹果，B型袋子包含8个红苹果和4个绿苹果。一位顾客购买了一袋A型和一袋B型的苹果，将全部苹果混合后等概率抽取一个，选到绿苹果的概率是多少？请将答案保留到小数点后两位。",
    "一位园丁按照两朵红色花跟着一朵黄色花的模式种花。如果园丁想保持这种模式，并且有35个连续的空位来种花，那么园丁会种多少朵红色花？",
    "杰森正在为马拉松训练。星期一，他跑了5英里。之后的每一天，他的跑步距离比前一天增加10%。如果杰森按照这个模式继续跑步，那么他在星期五将跑多少英里？",
    "16棵植物排列成一条直线，每棵植物占据一个半径为0.5米的圆形区域，相邻区域恰好相切。从第一棵植物区域的最左端到最后一棵植物区域的最右端，这一排占据的总长度是多少米？",
    "威尔逊博士正在设计一个几何花园，花园中的花朵围绕着中央的喷泉排列成同心圆。每一圈比里面一圈多6朵花，形成一个六边形的图案。最里面一圈有6朵花。如果威尔逊博士种足够的花，形成15圈（包括最里面一圈），那么这个花园总共需要多少朵花？",
    "一个图书馆最上层书架的最大承重是最下层书架的一半。如果最上层书架最多能承重15磅，那么最下层书架最多能承重多少磅？",
    "一份饼干的配方需要3杯面粉、2杯糖和1杯巧克力片。如果马克想要做三倍量的饼干，但只有4杯糖，那么他还需要多少杯糖？",
    "一家宠物店的店主正在制作定制鸟舍。每个鸟舍外部需要0.75升木材清漆。如果店主有一罐10升的木材清漆，那么他在需要更多清漆之前最多可以制作多少个鸟舍？",
    "一个农场有鸡和牛。总共有30个头，88条腿。农场上有多少头牛？",
    "一个地方图书馆正在组织一场旧书义卖会，以筹集资金购买新书。他们以每本2美元的价格卖出120本儿童书，以每本3美元的价格卖出75本小说，并以每本1.50美元的价格卖出了小说两倍数量的杂志。他们还以每本0.50美元的价格卖出与书籍和杂志总数相等的书签。那么图书馆总共筹集了多少钱？",
    "一个当地的农民正在为市场准备混合水果篮，每个篮子包含3个苹果、5个橙子和2个香蕉。苹果的价格是每个0.50美元，橙子每个0.30美元，香蕉每个0.25美元。如果农民为当地市场准备了120个篮子，并以每个5.00美元的价格出售每个篮子，那么卖完所有篮子后，农民将获得多少利润？",
    "玛丽亚有24个苹果，想将它们均匀分给她的6个朋友。如果每个朋友还要再给老师2个苹果，那么每个朋友剩下多少苹果？",
    "莉拉正在计划一个花园，想要种三种花：雏菊、郁金香和玫瑰。她想要的雏菊数量是郁金香的两倍，郁金香的数量是玫瑰的三倍。如果她总共要种60朵花，那么她计划种多少朵玫瑰？",
    "一个花园有三种开花植物。第一种每株有12朵花，第二种每株有8朵花，第三种每株有15朵花。如果第一种植物的数量是第二种植物的两倍，第三种植物的数量是第一种植物的一半，并且花园中有16株第二种植物，那么花园里一共有多少朵花？",
    "在一个棋盘游戏中，从一个方格转移到另一个方格的费用是你要落在的方格号码的硬币数。第一个方格是1号，第二个方格是2号，以此类推。如果一个玩家从5号方格移动到9号方格，再到14号方格，最后到20号方格，他总共花费了多少枚硬币？",
    "一个景观公司在两个公园种植树木。在A公园，他们种了5排，每排6棵树。在B公园，他们种了3排，每排7棵树。然而，B公园的4棵树没有成活，必须移除。移除之后，总共剩下多少棵树？",
    "欧拉博士正在计划一场数学比赛，他决定将参与者分成几组。为了保证公平，每组必须有相同数量的参与者。如果欧拉博士可以选择将参与者分成4人、5人或6人的组，并且参与者总数少于100人，那么他最多可以有多少参与者，确保无论怎么分组都不会有剩余？",
    "一个农民为万圣节种植南瓜。他种了8排，每排15棵南瓜植株。每棵植株平均产出3个南瓜。收获后，农民将20%的南瓜卖给当地市场，剩下的在他的农场摊位上出售。如果每个南瓜卖4美元，农民通过销售南瓜总共赚了多少钱？",
    "一个公园有三条绿化带，种植的树木数量依次构成公比为2的等比数列。如果三条绿化带总共种植了77棵树，那么第一条绿化带种植了多少棵树？",
    "一群朋友正在收集可回收的罐子。玛雅收集的罐子是利亚姆的两倍。利亚姆收集了15个罐子。如果佐伊比玛雅多收集了5个罐子，并且这群朋友想把罐子平分给4家慈善机构，每家会收到多少个罐子？",
    "在一场科学比赛中，每个团队需要制作一个模型火箭。有6个团队，每个团队需要一套材料。材料包括火箭的主体管、引擎和降落伞。主体管每个12.50美元，引擎每个18.75美元，降落伞每个6.25美元。购买所有团队的材料后，总费用为225美元。制作一支火箭的材料费用是多少？",
    "艾米丽有一个小菜园，种植了番茄、胡萝卜和黄瓜。她的番茄植株数量是黄瓜植株的两倍，而胡萝卜植株比番茄少5棵。如果艾米丽有4棵黄瓜植株，那么她总共有多少棵菜园植物？",
    "在一个小村庄，当地裁缝制作外套和裤子。制作一件外套需要3码布料，而制作一条裤子需要2码布料。他接到了一份剧院制作的订单，要求的裤子数量是外套的两倍，而剧院要求了4件外套。如果布料的价格是每码15美元，那么剧院在这个订单上需要花费多少布料费用？",
    "一个小镇每年增加的人口数量相同。如果2010年小镇的人口是5000人，2020年是8000人，那么如果这种增长趋势继续，到2025年小镇的人口会是多少？",
    "一位数学老师正在组织一场测验比赛，并决定用铅笔作为奖品。每位参与者将获得2支铅笔，而得分超过80%的学生将额外获得3支铅笔。如果班上有30名学生，其中1/5的学生得分超过80%，那么老师需要准备多少支铅笔？",
    "一个长方形的花园被120米的围栏包围。如果花园的长度是其宽度的三倍，那么花园的面积是多少平方米？",
    "一个长10米、宽15米的花园将用方形瓷砖铺设。每块瓷砖的边长为25厘米。如果每块瓷砖的价格是3美元，而铺设瓷砖的人工费用是每平方米8美元，那么铺设整个花园的总费用是多少？",
]
answers = [
    228,
    15,
    132,
    0.45,
    24,
    7.3205,
    16,
    720,
    30,
    2,
    13,
    14,
    862.5,
    180,
    2,
    6,
    752,
    43,
    47,
    60,
    1440,
    11,
    20,
    37.5,
    15,
    420,
    9500,
    78,
    675,
    8400,
]


# 使用gradio进行自定义prompt操作


def reset_prompt(chatbot):
    """
    Reset按钮点击处理：重置prompt

    参数:
        chatbot (List): 聊天记录

    返回:
        Tuple: 更新后的聊天记录和清空的提示词文本
    """
    gr.Info("已清除提示词")
    chatbot.extend(
        [
            {"role": "user", "content": "清除提示词"},
            {"role": "assistant", "content": "提示词已成功重置"},
        ]
    )
    return chatbot, "", None, "0"


def compile_prompt(prompt: str) -> jinja2.Template:
    """校验问题变量及模板语法，禁止未定义变量被静默渲染为空字符串。"""
    environment = jinja2.Environment(undefined=jinja2.StrictUndefined)
    syntax = environment.parse(prompt)
    if "question" not in meta.find_undeclared_variables(syntax):
        raise ValueError("你需要在提示词中包含占位符 {{question}}。")
    return environment.from_string(prompt)


def assign_prompt(chatbot, prompt, example_number):
    """设置成功后保存模板文本；任何校验失败都清除旧模板状态。"""
    gr.Info("正在分配提示词")
    token_num = my_model.prompt_token_num(prompt)
    template = None
    try:
        if token_num is None:
            raise ValueError("Token 数估算失败，请检查后重新设置提示词。")
        if token_num > 1024:
            raise ValueError("提示词太长（估算超过1024个token）。")
        if example_number is None or not 1 <= example_number <= len(questions):
            raise ValueError(f"请选择一个1到{len(questions)}之间的示例编号。")
        compiled = compile_prompt(prompt)
        prompt_ex = compiled.render(question=questions[int(example_number) - 1])
        # Gradio State 保存可复制的文本，不保存 Jinja2 编译对象。
        template = prompt
        chatbot.extend(
            [
                {"role": "user", "content": "分配提示词"},
                {
                    "role": "assistant",
                    "content": "提示词已成功分配\n\n自定义提示词示例：",
                },
                {"role": "assistant", "content": prompt_ex},
            ]
        )
    except (jinja2.TemplateError, ValueError, TypeError, ArithmeticError) as e:
        message = f"提示词无效：{e}"
        gr.Warning(message)
        chatbot.append({"role": "assistant", "content": message})
    return (
        chatbot,
        prompt,
        template,
        "估算失败" if token_num is None else str(token_num),
    )


def parse_numeric_answer(text: str) -> float | None:
    """整条最终回复必须是一个有限数值；支持千位分隔符和科学计数法。"""
    pattern = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    text = text.strip()
    if re.fullmatch(pattern, text) is None:
        return None
    number = float(text.replace(",", ""))
    return number if math.isfinite(number) else None


def is_correct_answer(number: float, ground_truth: float) -> bool:
    """只容忍微小浮点误差，不替模型舍入题目要求保留的小数位数。"""
    return math.isclose(number, ground_truth, rel_tol=1e-9, abs_tol=1e-9)


def evaluation_error(chatbot, message):
    """校验失败时清空旧结果及导航状态。"""
    gr.Warning(message)
    chatbot.append({"role": "assistant", "content": message})
    return (
        chatbot,
        [],
        message,
        0,
        gr.Slider(value=1, minimum=1, maximum=3, step=1, interactive=False),
        gr.Slider(value=1, minimum=1, maximum=2, step=1, interactive=False),
        "",
    )


def show_result(results, evaluated_count, trial_number, question_number):
    """使用已完成评估的题数定位结果，与下一次评估的题数无关。"""
    if not results or evaluated_count < 1:
        return "暂无结果"
    if not 1 <= trial_number <= 3 or not 1 <= question_number <= evaluated_count:
        return "结果编号超出范围"
    index = (int(trial_number) - 1) * evaluated_count + int(question_number) - 1
    if not 0 <= index < len(results):
        return "结果编号超出范围"
    return results[index]


def assess_prompt(chatbot, template, test_num):
    """
    Test按钮点击处理：评估自定义prompt

    参数:
        chatbot (List): 聊天记录
        template: 已校验的模板文本
        test_num (int): 要测试的问题数量

    返回:
        Tuple: 更新后的聊天记录、结果列表、结果统计和UI组件
    """
    if template is None:
        return evaluation_error(chatbot, "提示词未设置，请先点击 Set Prompt。")
    if (
        not isinstance(test_num, (int, float))
        or not math.isfinite(test_num)
        or int(test_num) != test_num
        or not 1 <= test_num <= len(questions)
    ):
        return evaluation_error(
            chatbot, f"评估题数必须是1到{len(questions)}之间的整数。"
        )
    test_num = int(test_num)
    try:
        compiled = compile_prompt(template)
        # 在请求 API 前渲染全部题目，避免模板只在某道题上报错而中断评估。
        prompts = [compiled.render(question=q) for q in questions[:test_num]]
    except (jinja2.TemplateError, ValueError, TypeError, ArithmeticError) as e:
        return evaluation_error(chatbot, f"提示词渲染失败：{e}")

    gr.Info("正在评估提示词")
    ans_template = "提示词和问题：\n\n{{question}}\n\n--------------------\n\n解题过程：\n\n{{rationale}}\n\n--------------------\n\n最终答案\n\n{{answer}}"
    res_list = []
    total_count = test_num
    environment = jinja2.Environment()
    ans_template = environment.from_string(ans_template)
    trial_num = 3
    trials = [[] for _ in range(trial_num)]
    res_stats_str = (
        "判分规则：最终回复必须只有一个数字；相对和绝对容差均为1e-9。\n"
        "请求失败或格式无效均计为未答对，并单独统计。\n"
    )

    for i in range(trial_num):
        gr.Info(f"开始第{i + 1}次测试")
        accurate_count = 0
        request_failures = 0
        format_failures = 0
        for idx, example in enumerate(questions[:test_num]):
            test_res = ""
            result = my_model.two_stage_completion(example, prompts[idx])

            if not result["answer"]:
                trials[i].append(0)
                request_failures += 1
                res_list.append(
                    f"第{i + 1}次测试，问题 {idx + 1}：请求失败或回复为空，计为未答对。\n\n"
                )
                continue

            number = parse_numeric_answer(result["answer"])
            if number is None:
                format_failures += 1
                verdict = "格式无效：最终回复须为单个数字"
                trials[i].append(0)
            elif is_correct_answer(number, answers[idx]):
                accurate_count += 1
                trials[i].append(1)
                verdict = "正确"
            else:
                trials[i].append(0)
                verdict = "错误"

            test_res += f"第{i + 1}次测试\n\n"
            test_res += f"问题 {idx + 1}:\n" + "-" * 20
            test_res += f"""\n\n{ans_template.render(question=result["prompt"], rationale=result["rationale"], answer=result["answer"])}\n"""
            test_res += f"\n判定：{verdict}；标准答案：{answers[idx]}\n"
            test_res += "\n" + "<" * 6 + "=" * 30 + ">" * 6 + "\n\n"
            res_list.append(test_res)

        res_stats_str += f"第{i + 1}次测试，正确数：{accurate_count}，总数：{total_count}，准确率：{accurate_count / total_count * 100}%，请求失败或空回复：{request_failures}，格式无效：{format_failures}\n"
        my_model.save_cache()

    voting_acc = 0
    for i in range(total_count):
        count = 0
        for j in range(trial_num):
            if trials[j][i] == 1:
                count += 1
        if count >= 2:
            voting_acc += 1

    res_stats_str += f"三次中至少两次答对的题目占比：{voting_acc / total_count * 100}%"
    chatbot.extend(
        [
            {"role": "user", "content": "测试"},
            {
                "role": "assistant",
                "content": "测试完成。结果可以在“结果”和“结果统计”中找到。",
            },
            {"role": "assistant", "content": "测试结果"},
            {"role": "assistant", "content": "".join(res_list)},
            {"role": "assistant", "content": "结果统计"},
            {"role": "assistant", "content": res_stats_str},
        ]
    )

    return (
        chatbot,
        res_list,
        res_stats_str,
        total_count,
        gr.Slider(
            value=1,
            minimum=1,
            maximum=3,
            step=1,
            interactive=True,
        ),
        gr.Slider(
            value=1,
            minimum=1,
            maximum=max(2, total_count),
            step=1,
            interactive=total_count > 1,
        ),
        res_list[0],
    )


def save_prompt(chatbot, prompt):
    """
    Save按钮点击处理：保存提示词

    参数:
        chatbot (List): 聊天记录
        prompt (str): 用户输入的提示词

    返回:
        List: 更新后的聊天记录
    """
    gr.Info("正在保存提示词")
    prompt_dict = {"prompt": prompt}
    with open("prompt.json", "w") as f:
        json.dump(prompt_dict, f)
    chatbot.extend(
        [
            {"role": "user", "content": "保存提示词"},
            {"role": "assistant", "content": "提示词已保存为prompt.json"},
        ]
    )
    return chatbot


# Gradio界面
with gr.Blocks() as demo:
    my_magic_prompt = "任务：\n解决以下数学问题。\n\n问题：{{question}}\n\n答案："
    my_magic_prompt = my_magic_prompt.strip("\n")
    template = gr.State(None)
    res_list = gr.State(list())
    evaluated_count = gr.State(0)

    # 组件
    with gr.Tab(label="Console"):
        with gr.Group():
            example_num_box = gr.Dropdown(
                label="Demo Example (Please choose one example for demo)",
                value=1,
                info=questions[0],
                choices=[i + 1 for i in range(len(questions))],
                filterable=False,
            )
            prompt_textbox = gr.Textbox(
                label="Custom Prompt",
                placeholder=f"在这里输入你的自定义提示词。例如：\n\n{my_magic_prompt}",
                value="",
                info="请包含 {{question}}；编辑后需要重新点击 Set Prompt。",
            )
            with gr.Row():
                set_button = gr.Button(value="Set Prompt")
                reset_button = gr.Button(value="Clear Prompt")
            prompt_token_num = gr.Textbox(
                label="Number of prompt tokens",
                value="0",
                interactive=False,
                info="自定义提示词的Token数量估算，不含渲染后的问题。",
            )
        with gr.Group():
            test_num = gr.Slider(
                label="Number of examples used for evaluation",
                minimum=1,
                maximum=len(questions),
                step=1,
                value=1,
            )
            assess_button = gr.Button(value="Evaluate")
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        trial_no = gr.Slider(
                            label="Trial ID",
                            value=1,
                            minimum=1,
                            maximum=3,
                            step=1,
                            interactive=False,
                        )
                        ques_no = gr.Slider(
                            label="Question ID",
                            value=1,
                            minimum=1,
                            maximum=2,
                            step=1,
                            interactive=False,
                        )
                    res = gr.Textbox(
                        label="Result",
                        value="",
                        placeholder="暂无结果",
                        interactive=False,
                    )
                with gr.Column():
                    res_stats = gr.Textbox(label="Result Stats", interactive=False)
            save_button = gr.Button(value="Save Custom Prompt")
    with gr.Tab(label="Log"):
        chatbot = gr.Chatbot(label="Log")

    # 事件处理
    example_num_box.input(
        lambda example_number: gr.Dropdown(
            label="Example (Please choose one example for demo)",
            value=example_number,
            info=questions[example_number - 1],
            choices=[i + 1 for i in range(len(questions))],
        ),
        inputs=[example_num_box],
        outputs=[example_num_box],
    )

    trial_no.input(
        show_result,
        inputs=[res_list, evaluated_count, trial_no, ques_no],
        outputs=[res],
    )
    ques_no.input(
        show_result,
        inputs=[res_list, evaluated_count, trial_no, ques_no],
        outputs=[res],
    )
    set_button.click(
        assign_prompt,
        inputs=[chatbot, prompt_textbox, example_num_box],
        outputs=[chatbot, prompt_textbox, template, prompt_token_num],
    )
    reset_button.click(
        reset_prompt,
        inputs=[chatbot],
        outputs=[chatbot, prompt_textbox, template, prompt_token_num],
    )
    prompt_textbox.input(
        lambda: (None, "待重新设置"), outputs=[template, prompt_token_num]
    )
    assess_button.click(
        assess_prompt,
        inputs=[chatbot, template, test_num],
        outputs=[chatbot, res_list, res_stats, evaluated_count, trial_no, ques_no, res],
    )
    save_button.click(save_prompt, inputs=[chatbot, prompt_textbox], outputs=[chatbot])

if __name__ == "__main__":
    check_api_settings()
    demo.queue().launch(debug=True)
