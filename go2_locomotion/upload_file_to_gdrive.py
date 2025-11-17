import argparse
import os
import requests
import zipfile
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv

load_dotenv()

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    """Authenticate and return Google Drive service object."""
    token_path = os.getenv("TOKEN_FILE_PATH", "token.json")
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            credentials_path = os.getenv("CREDENTIALS_FILE_PATH", "credentials.json")
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

def upload_file(service, file_path, drive_filename=None, folder_id=None):
    """Upload a file to Google Drive."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    
    # Use provided filename or default to original filename
    filename = drive_filename if drive_filename else os.path.basename(file_path)
    
    file_metadata = {
        'name': filename
    }
    
    # If folder_id is provided, upload to that folder
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(file_path, resumable=True)
    
    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,webViewLink'
        ).execute()
        
        print(f"✅ File uploaded successfully!")
        print(f"   Name: {file.get('name')}")
        print(f"   ID: {file.get('id')}")
        print(f"   Link: {file.get('webViewLink')}")
        return file
        
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        return None

def send_discord_notification(webhook_url, message):
    """Send notification to Discord channel"""
    data = {"content": message}
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("✅ Discord notification sent!")
        else:
            print(f"❌ Discord notification failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Discord notification failed: {e}")

def zip_log_folder(log_folder_path, zip_filename):
    """Zip the log folder"""
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(log_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Get relative path for the zip
                    arcname = os.path.relpath(file_path, os.path.dirname(log_folder_path))
                    zipf.write(file_path, arcname)
        print(f"✅ Log folder zipped: {zip_filename}")
        return True
    except Exception as e:
        print(f"❌ Error zipping log folder: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload a file to Google Drive')
    parser.add_argument('file_path', help='Path to the file to upload')
    parser.add_argument('--name', '-n', help='Name for the file on Google Drive (optional)')
    parser.add_argument('--folder_id', '-f', help='Google Drive folder ID to upload to (optional)')
    
    args = parser.parse_args()
    
    try:
        # Authenticate and get service
        service = authenticate_google_drive()
        
        # Upload file
        upload_file(service, args.file_path, args.name, args.folder_id)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()