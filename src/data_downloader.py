import os
import urllib.request
import zipfile

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{os.path.basename(dest_path)} already exists, skipping download.")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            print(f"\rProgress: {percent:.1f}% ({read_so_far}/{total_size} bytes)", end="")
        else:
            print(f"\rDownloaded {read_so_far} bytes", end="")
            
    urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
    print("\nDownload complete.")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

def main():
    base_url = "https://physionet.org/files/challenge-2012/1.0.0/"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    files_to_download = [
        ("Outcomes-a.txt", "Outcomes-a.txt"),
        ("Outcomes-b.txt", "Outcomes-b.txt"),
        ("set-a.zip", "set-a.zip"),
        ("set-b.zip", "set-b.zip")
    ]
    
    for filename, save_name in files_to_download:
        url = base_url + filename
        dest = os.path.join(data_dir, save_name)
        download_file(url, dest)
        
        if filename.endswith(".zip"):
            extract_target = os.path.join(data_dir, filename.replace(".zip", ""))
            if not os.path.exists(extract_target):
                extract_zip(dest, data_dir)
            else:
                print(f"Directory {extract_target} already exists, skipping extraction.")

if __name__ == "__main__":
    main()
