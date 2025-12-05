# from google_drive_downloader import GoogleDriveDownloader as gdd
from googledrivedownloader import download_file_from_google_drive
def download_weights():
    
    download_file_from_google_drive(file_id="1--7p9rRJy7WU4OmomkzM8i0veetZctTT",
                                        dest_path ="weight/modeldense1.h5" )
