import subprocess

def run_scripts():
    # List of command lines to execute
    commands = [
        #['python', 'main_ml.py', 'gpt4', 'bert'],
        #['python', 'main_ml.py', 'gpt4', 'bart'],
        #['python', 'main_ml.py', 'gpt4', 'roberta'],
        #['python', 'main_ml.py', 'gpt4_finetuned', 'bert'],
        #['python', 'main_ml.py', 'gpt4_finetuned', 'bart'],
        #['python', 'main_ml.py', 'gpt4_finetuned', 'roberta'],
        ['python', 'main_ml.py', 'llama_8b', 'bert'],
        ['python', 'main_ml.py', 'llama_8b', 'bart'],
        ['python', 'main_ml.py', 'llama_8b', 'roberta'],
        #['python', 'main_ml.py', 'llama_8b_finetuned', 'bert'],
        #['python', 'main_ml.py', 'llama_8b_finetuned', 'bart'],
        #['python', 'main_ml.py', 'llama_8b_finetuned', 'roberta']
        ['python', 'main_ml.py', 'llama_8b_finetuned', 'tudo']
    ]
    
    for command in commands:
        subprocess.run(command)

if __name__ == "__main__":
    run_scripts()
