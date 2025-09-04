import os
import sys
import subprocess

def download_files():
    """
    Downloads the models and demo files.
    """
    import gdown
    from huggingface_hub import hf_hub_download

    print("Menyiapkan folder model")
    models_dir = "src/models"
    os.makedirs(models_dir, exist_ok=True)
    print("folder berhasil dibuat!")

    repo_id_detection = "IceKhoffi/chicken-object-detection-yolov11s"
    filename_detection = "yolov8n.pt"

    repo_id_classifier = "IceKhoffi/chicken-vocalization-classifier"
    filename_classifier = "Chicken_CNN_Disease_Detection_Model.pth"

    print(f"\nMengunduh model object detection: {repo_id_detection}")
    try:
        hf_hub_download(repo_id=repo_id_detection, filename=filename_detection, local_dir=models_dir, local_dir_use_symlinks=False)
        print("Model object detection berhasil diunduh.")
    except Exception as e:
        print(f"Error menggunduh model detection: {e}")

    print(f"\nMengunduh model classifier: {repo_id_classifier}...")
    try:
        hf_hub_download(repo_id=repo_id_classifier, filename=filename_classifier, local_dir=models_dir, local_dir_use_symlinks=False)
        print("Model classifier berhasil diunduh.")
    except Exception as e:
        print(f"Error menggunduh classifier model: {e}")
    
    print("\nMengunduh folder demo dari Google Drive")
    try:
        subprocess.run([sys.executable, "-m", "gdown", "--folder", "https://drive.google.com/drive/folders/1tt-jrv7diZwoweLVERikgooJATtJBol2?usp=drive_link", "-O", "demo"], check=True)
        print("Demo folder berhasil diunduh")
    except subprocess.CalledProcessError:
        print("Error saat mengunduh folder demo.")
    except FileNotFoundError:
        print("Error library 'gdown' tidak terinstall mohon pip install gdown")

    print("\nSelesai!")

def main():
    download_files()

if __name__ == "__main__":
    main()