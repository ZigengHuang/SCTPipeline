import pandas as pd
import random
import os

# 设置随机种子，确保每次运行随机结果一致
random.seed(42)


def csv_to_txt_batch(input_folder, output_folder):
    """
    批量将CSV文件转换为TXT格式
    每个CSV将生成一个同名TXT文件，内容格式为：Role：Transcription
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        print("未找到任何CSV文件。")
        return

    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        output_file = os.path.join(output_folder, base_name + '.txt')

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"读取文件 {csv_file} 失败: {e}")
            continue

        # --- 随机分配角色 ---
        df['speaker'] = df['speaker'].astype(str)
        unique_speakers = df['speaker'].unique()

        roles = ['Patient', 'Doctor']
        speaker_map = {uid: random.choice(roles) for uid in unique_speakers}
        df['Role'] = df['speaker'].map(speaker_map)

        # --- 拼接内容 ---
        dialogue_lines = []
        for _, row in df.iterrows():
            text_col = 'transcription' if 'transcription' in df.columns else 'text'
            transcription = str(row.get(text_col, '')).strip()
            # 保留空行以保持原行数
            if transcription or transcription == '':
                line = f"{row['Role']}：{transcription}\n"
                dialogue_lines.append(line)

        # --- 保存为TXT ---
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(dialogue_lines)

        print(f"成功生成TXT文件: {output_file}")
        print(f"--- 映射关系: {speaker_map}")

    print("\n批量转换完成！")


if __name__ == "__main__":
    csv_to_txt_batch(
        input_folder="path_to_csv_folder",  # 输入CSV文件夹路径
        output_folder="path_to_output_txt"  # 输出TXT文件夹路径
    )
