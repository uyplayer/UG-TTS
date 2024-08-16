import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_dir, 'logs')
log_file = os.path.join(project_dir, 'logs/UG-TTS.log')
data_dir = os.path.join(project_dir, 'data')

if __name__ == "__main__":
    print(project_dir)
    print(log_dir)
    print(log_file)
    print(data_dir)
