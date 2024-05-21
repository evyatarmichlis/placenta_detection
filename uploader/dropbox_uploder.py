import dropbox
import dropbox.files

with open("dropbox_token.txt","r") as f:
    TOKEN = f.read()

import os


class DropboxUploader:
    def __init__(self):
        self.dbx = dropbox.Dropbox(TOKEN)
    def upload_to_dropbox(self, file_path, dropbox_folder):
        try:
            file_name = os.path.basename(file_path)
            dropbox_path = os.path.join(dropbox_folder, file_name).replace("\\", "/")

            try:
                self.dbx.files_get_metadata(dropbox_path)
                raise Exception(f"File '{file_name}' already exists in the folder.")
            except dropbox.exceptions.ApiError as err:
                # If the error is not 'path/not_found', re-raise the exception
                if err.error.is_path() and err.error.get_path().is_not_found():
                    pass
                else:
                    raise

            # Read the file content
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Upload the file
            self.dbx.files_upload(file_content, dropbox_path)
            print(f"File '{file_path}' uploaded to Dropbox at '{dropbox_path}'.")

        except Exception as error:
            print(f"An error occurred: {error}")

    def upload_csv_to_dropbox(self, csv_path, dropbox_folder):
        self.upload_to_dropbox(csv_path, dropbox_folder)


uploader = DropboxUploader()

# Upload an image
image_path = r"C:\Users\Evyatar\PycharmProjects\placenta_detection\Images\color_images\mask-image_2024-02-08_12-35-29.jpg"
dropbox_image_folder = '/images'
uploader.upload_to_dropbox(image_path, dropbox_image_folder)

# Upload a CSV file
csv_path = r"C:\Users\Evyatar\PycharmProjects\placenta_detection\Images\csv_data\raw_depth_bipartite_data_2024-02-08_13-09-01.csv"
dropbox_csv_folder = '/csv_files'
uploader.upload_csv_to_dropbox(csv_path, dropbox_csv_folder)
