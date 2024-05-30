import dropbox
import dropbox.files
import os
import requests

class DropboxUploader:
    def __init__(self, token_file, refresh_token, client_id, client_secret):
        self.token_file = token_file
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = self.load_access_token()
        self.dbx = self.get_dropbox_client()

    def load_access_token(self):
        with open(self.token_file, "r") as f:
            return f.read().strip()

    def save_access_token(self, access_token):
        with open(self.token_file, 'w') as f:
            f.write(access_token)

    def refresh_access_token(self):
        url = "https://api.dropbox.com/oauth2/token"
        data = {
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        response = requests.post(url, data=data)
        if response.status_code == 200:
            new_token_data = response.json()
            new_access_token = new_token_data['access_token']
            self.save_access_token(new_access_token)
            self.access_token = new_access_token
        else:
            raise Exception(f"Failed to refresh access token: {response.text}")

    def get_dropbox_client(self):
        try:
            dbx = dropbox.Dropbox(self.access_token)
            # Test the token to ensure it is still valid
            dbx.users_get_current_account()
        except dropbox.exceptions.AuthError:
            # If token is expired, refresh it
            self.refresh_access_token()
            dbx = dropbox.Dropbox(self.access_token)
        return dbx

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

if __name__ == "__main__":
    token_file = "dropbox_token.txt"
    refresh_token = "ZDhC-N3mwtMAAAAAAAAAAXKuN-TzjTfa_kvRtnVSrcNJ0CHZTQUSol62CXEBRu31"
    client_id = "prrl5bsxyc65kxw"
    client_secret = "hxw3omlgoj9uulz"

    uploader = DropboxUploader(token_file, refresh_token, client_id, client_secret)

    # Upload an image
    image_path = r"C:\Users\Evyatar\PycharmProjects\placenta_detection\Images\color_images\mask-image_2024-02-08_12-35-29.jpg"
    dropbox_image_folder = '/images'
    uploader.upload_to_dropbox(image_path, dropbox_image_folder)

    # Upload a CSV file
    csv_path = r"C:\Users\Evyatar\PycharmProjects\placenta_detection\Images\csv_data\raw_depth_bipartite_data_2024-02-08_13-09-01.csv"
    dropbox_csv_folder = '/csv_files'
    uploader.upload_csv_to_dropbox(csv_path, dropbox_csv_folder)
