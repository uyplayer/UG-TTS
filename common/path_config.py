import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_dir = os.path.join(project_dir, 'logs')
log_file = os.path.join(project_dir, 'logs/digital_twin_stable_1.log')



"""-----------------------结果输出路径---------------------------"""
data_dir = os.path.join(project_dir, 'data')



if __name__ == "__main__":
    print(data_dir)
    print(log_dir)
