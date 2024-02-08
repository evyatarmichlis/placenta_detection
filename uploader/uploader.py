import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import consts


class GoogleDriveUploader:
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    def __init__(self):
        self.creds = self.authenticate()

    def authenticate(self):
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", self.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as error:
                    print(f"An error occurred: {error}")
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", self.SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", self.SCOPES)
                creds = flow.run_local_server(port=0)

            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return creds

    def upload_to_drive(self, image_path, folder_id=consts.Folders.placenta_main_folder):
        try:
            service = build("drive", "v3", credentials=self.creds)
            existing_files = service.files().list(
                q=f"name='{os.path.basename(image_path)}' and '{folder_id}' in parents",
                fields="files(id)"
            ).execute().get("files", [])

            if existing_files:
                raise Exception(f"File '{os.path.basename(image_path)}' already exists in the folder.")

            file_metadata = {"name": os.path.basename(image_path), "parents": [folder_id]}
            media = MediaFileUpload(image_path, mimetype="image/jpeg")

            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()

            print(f"File '{os.path.basename(image_path)}' uploaded to Google Drive. File ID: {uploaded_file['id']}")

        except Exception as error:
            print(f"An error occurred: {error}")

    def upload_csv_to_drive(self, csv_path, folder_id=consts.Folders.placenta_main_folder):
        try:
            service = build("drive", "v3", credentials=self.creds)
            existing_files = service.files().list(
                q=f"name='{os.path.basename(csv_path)}' and '{folder_id}' in parents",
                fields="files(id)"
            ).execute().get("files", [])

            if existing_files:
                raise Exception(f"File '{os.path.basename(csv_path)}' already exists in the folder.")

            file_metadata = {"name": os.path.basename(csv_path), "parents": [folder_id]}
            media = MediaFileUpload(csv_path, mimetype="text/csv")

            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()

            print(f"File '{os.path.basename(csv_path)}' uploaded to Google Drive. File ID: {uploaded_file['id']}")

        except Exception as error:
            print(f"An error occurred: {error}")

if __name__ == "__main__":
    uploader = GoogleDriveUploader()
    image_paths = r"C:\Users\Evyatar\PycharmProjects\placenta\placenta_example3.jpg"
    uploader.upload_to_drive(image_paths)
