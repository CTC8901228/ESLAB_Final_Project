import numpy as np
import os

def save_matrix_as_npy(matrix, char, folder):
    """
    將 NumPy 矩陣保存成 npy 檔案

    Parameters:
    - matrix: 要保存的 NumPy 矩陣
    - char: 用來構建檔案名稱的字元
    - folder: 存放檔案的資料夾名稱
    """
    # 確保資料夾存在
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 構建檔案路徑和名稱
    filename = os.path.join(folder, f"{char}.npy")

    # 保存矩陣
    np.save(filename, matrix)
    print(f"矩陣已成功保存為 {filename}")

'''
# 使用範例
matrix_to_save = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 從鍵盤輸入一個字元
user_input = input("請輸入一個字元：")

# 指定存放檔案的資料夾
save_folder = "BBB"  # 更換成你想要的資料夾名稱

# 呼叫函數保存矩陣
save_matrix_as_npy(matrix_to_save, user_input, save_folder)
'''


def load_matrix_from_npy(char, folder):
    """
    從 npy 檔案讀取數據並返回 NumPy 矩陣

    Parameters:
    - char: 檔案名稱的字元
    - folder: 存放檔案的資料夾名稱

    Returns:
    - matrix: 讀取的 NumPy 矩陣
    """
    # 構建檔案路徑和名稱
    filename = os.path.join(folder, f"{char}.npy")

    # 讀取矩陣
    matrix = np.load(filename)
    print(f"成功從 {filename} 讀取矩陣")
    return matrix

# 使用範例
# 使用者輸入一個字元
user_input = input("請輸入一個字元：")

# 指定存放檔案的資料夾
load_folder = "BBB"  # 更換成你的資料夾名稱

# 呼叫函數讀取矩陣
loaded_matrix = load_matrix_from_npy(user_input, load_folder)
print("讀取的矩陣:")
print(loaded_matrix)

