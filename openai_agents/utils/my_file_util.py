import re
from datetime import datetime
from pathlib import Path


def save_story_to_file(content, user_prompt, output_dir):
    """
    将生成的故事保存到本地文件

    Args:
        content: 生成的故事内容
        user_prompt: 用户的初始提示

    Returns:
        保存的文件路径
    """
    # 清理模型输出中的思考过程
    cleaned_story = remove_thinking_process(content)

    # 创建保存目录
    save_dir = Path(output_dir)
    save_dir.mkdir(exist_ok=True)

    # 生成文件名 (使用时间戳确保唯一性)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{user_prompt}_{timestamp}.txt"

    # 完整的文件路径
    file_path = save_dir / filename

    # 写入故事内容，包括用户提示和时间戳
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"用户提示: {user_prompt}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("\n==================\n\n")
        f.write(cleaned_story)

    return file_path


def remove_thinking_process(text):
    """
    移除文本中<think>标签之间的内容

    Args:
        text: 原始文本内容

    Returns:
        清理后的文本
    """
    # 检查文本中是否存在<think>标签
    if "<think>" in text and "</think>" in text:
        print("检测到思考过程，正在清理...")
        # 使用正则表达式找到并移除所有<think>...</think>内容
        cleaned_text = re.sub(r'(?s)<think>.*?</think>', '', text)
        # 移除可能产生的多余空行
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        return cleaned_text.strip()
    else:
        # 如果没有思考过程标签，返回原始文本
        return text
