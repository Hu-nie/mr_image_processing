import os
from tqdm import tqdm
if __name__ == "__main__":
    root_dir = "D:/raw/2012/"
    i=0
    for (root, dirs, files) in tqdm(os.walk(root_dir)):
        print("# root : " + root)
        i+=1