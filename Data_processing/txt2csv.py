import os
import re
import pandas as pd


def merge_txt_to_csv(input_folder, original_folder, output_folder):
    """
    批量处理文件夹中的所有TXT文件，并根据文件名将其合并到对应的CSV文件中。
    """
    # 获取输入文件夹中的所有txt文件
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    for txt_file in txt_files:
        # 构建txt文件和对应的csv文件路径
        txt_path = os.path.join(input_folder, txt_file)
        csv_name = os.path.splitext(txt_file)[0] + '.csv'
        csv_path = os.path.join(original_folder, csv_name)

        # 检查CSV文件是否存在
        if not os.path.exists(csv_path):
            print(f"找不到对应的CSV文件：{csv_path}")
            continue

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 读取TXT文件内容
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        speakers = []
        texts = []

        # 处理TXT文件的每一行
        for line in lines:
            match = re.match(r"^([^:：]+)[:：]\s*(.*)$", line)  # 支持中英文冒号
            if match:
                spk, content = match.groups()
                speakers.append(spk.strip())
                texts.append(content.strip())
            else:
                speakers.append("")
                texts.append(line.strip())

        # 检查行数是否匹配
        if len(df) != len(speakers):
            print(f"行数不匹配：CSV {len(df)} 行，TXT {len(speakers)} 行。将按最短长度对齐。")
            min_len = min(len(df), len(speakers))
            df = df.iloc[:min_len]
            speakers = speakers[:min_len]
            texts = texts[:min_len]

        # 更新CSV内容
        if 'speaker' in df.columns:
            df['speaker'] = speakers
        else:
            df.insert(0, 'speaker', speakers)

        if 'text' in df.columns:
            df['text'] = texts
        else:
            df['text'] = texts

        # 保存更新后的CSV文件
        output_file = os.path.join(output_folder, csv_name)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"文件 {csv_name} 已更新并保存到：{output_file}")


if __name__ == "__main__":
    # 示例路径：请修改为你自己文件夹的路径
    merge_txt_to_csv(
        input_folder="path_to_txt_files",  # TXT文件夹路径
        original_folder="path_to_csv_files",  # 原始CSV文件夹路径
        output_folder="path_to_output"  # 输出更新后的CSV文件夹路径
    )
