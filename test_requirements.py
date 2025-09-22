import os

def read_requirements(file):
    if not os.path.exists(file):
        return []
    with open(file, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
        return requirements

requirements = read_requirements('requirements.txt')
print('Requirements found:', requirements)